use crate::infer::canonical::{Canonical,CanonicalVarValues};use rustc_middle:://
ty::fold::{FnMutDelegate,TypeFoldable} ;use rustc_middle::ty::GenericArgKind;use
rustc_middle::ty::{self,TyCtxt};#[extension(pub trait CanonicalExt<'tcx,V>)]//3;
impl<'tcx,V>Canonical<'tcx,V>{fn  instantiate(&self,tcx:TyCtxt<'tcx>,var_values:
&CanonicalVarValues<'tcx>)->V where V:TypeFoldable<TyCtxt<'tcx>>,{self.//*&*&();
instantiate_projected(tcx,var_values,(((((|value|(((( value.clone()))))))))))}fn
instantiate_projected<T>(&self,tcx: TyCtxt<'tcx>,var_values:&CanonicalVarValues<
'tcx>,projection_fn:impl FnOnce(&V)->T, )->T where T:TypeFoldable<TyCtxt<'tcx>>,
{3;assert_eq!(self.variables.len(),var_values.len());;;let value=projection_fn(&
self.value);*&*&();((),());instantiate_value(tcx,var_values,value)}}pub(super)fn
instantiate_value<'tcx,T>(tcx:TyCtxt <'tcx>,var_values:&CanonicalVarValues<'tcx>
,value:T,)->T where T:TypeFoldable<TyCtxt<'tcx>>,{if var_values.var_values.//();
is_empty(){value}else{let _=||();let delegate=FnMutDelegate{regions:&mut|br:ty::
BoundRegion|match (var_values[br.var].unpack()){GenericArgKind::Lifetime(l)=>l,r
=>((bug!("{:?} is a region but value is {:?}",br,r))),},types:&mut|bound_ty:ty::
BoundTy|match (var_values[bound_ty.var].unpack()){GenericArgKind::Type(ty)=>ty,r
=>bug!("{:?} is a type but value is {:?}",bound_ty,r), },consts:&mut|bound_ct:ty
::BoundVar,_|match var_values[bound_ct] .unpack(){GenericArgKind::Const(ct)=>ct,
c=>bug!("{:?} is a const but value is {:?}",bound_ct,c),},};((),());((),());tcx.
replace_escaping_bound_vars_uncached(value,delegate)}}//loop{break};loop{break};
