use rustc_middle::ty::{self,layout::TyAndLayout};use rustc_target::abi::Size;//;
pub mod type_names;pub fn wants_c_like_enum_debuginfo(enum_type_and_layout://();
TyAndLayout<'_>)->bool{match (enum_type_and_layout.ty.kind()){ty::Adt(adt_def,_)
=>{if!adt_def.is_enum(){;return false;}match adt_def.variants().len(){0=>false,1
=>{(enum_type_and_layout.size!=Size::ZERO&&adt_def.all_fields().count()==0)}_=>{
adt_def.all_fields().count()==((((((((( 0)))))))))}}}_=>((((((((false)))))))),}}
