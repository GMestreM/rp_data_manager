from database import Base
from sqlalchemy import (
    Column, String, Date, DateTime, Integer, Numeric, Float, Boolean,
    ForeignKey,
)
from datetime import date, datetime

# Define database schema
class AssetHeader(Base):
    __tablename__ = 'asset_header'
    
    id = Column(Integer, primary_key=True)
    isin = Column(String())
    name = Column(String())
    shortname = Column(String())
    currency = Column(String(3))
    country = Column(String(50))
    
class AssetPrices(Base):
    __tablename__ = 'asset_prices'
    
    id_asset = Column(Integer, ForeignKey('asset_header.id'), primary_key=True)
    date = Column(Date(), primary_key=True)
    timestamp = Column(DateTime(), default = datetime.now, primary_key=True)
    open_price = Column(Float())
    high_price = Column(Float())
    low_price = Column(Float())
    close_price = Column(Float())
    volume = Column(Numeric())
    change_pct = Column(Float())
    
    
class ModelHeader(Base):
    __tablename__ = 'model_header'
    
    id = Column(Integer, primary_key=True)
    flag = Column(Boolean())
    assets = Column(String())
    window_size = Column(Integer())
    constr_dict = Column(String())
    rp_model = Column(String())
    risk_measure = Column(String())
    
class ModelWeights(Base):
    __tablename__ = 'model_weights'
    
    id_model = Column(Integer, ForeignKey('model_header.id'), primary_key=True)
    id_asset = Column(Integer, ForeignKey('asset_header.id'), primary_key=True)
    date = Column(Date(), primary_key=True)
    timestamp = Column(DateTime(), default = datetime.now, primary_key=True)
    weight = Column(Float())