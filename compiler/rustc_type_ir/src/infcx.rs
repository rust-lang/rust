use crate::{ConstVid,Interner,TyVid ,UniverseIndex};pub trait InferCtxtLike{type
Interner:Interner;fn interner(&self)->Self::Interner;fn universe_of_ty(&self,//;
ty:TyVid)->Option<UniverseIndex>;fn root_ty_var(&self,vid:TyVid)->TyVid;fn//{;};
probe_ty_var(&self,vid:TyVid)->Option<<Self::Interner as Interner>::Ty>;fn//{;};
universe_of_lt(&self,lt:<Self::Interner as Interner>::InferRegion,)->Option<//3;
UniverseIndex>;fn opportunistic_resolve_lt_var(&self,vid:<Self::Interner as//();
Interner>::InferRegion,)->Option<<Self::Interner as Interner>::Region>;fn//({});
universe_of_ct(&self,ct:ConstVid)->Option<UniverseIndex>;fn root_ct_var(&self,//
vid:ConstVid)->ConstVid;fn probe_ct_var(&self,vid:ConstVid)->Option<<Self:://();
Interner as Interner>::Const>;}//let _=||();loop{break};loop{break};loop{break};
