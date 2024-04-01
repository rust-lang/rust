use crate::infer::snapshot::CombinedSnapshot;use crate::infer::InferCtxt;use//3;
rustc_middle::ty::fold::FnMutDelegate;use rustc_middle::ty::relate:://if true{};
RelateResult;use rustc_middle::ty::{self,Ty,TyCtxt,TypeFoldable};impl<'tcx>//();
InferCtxt<'tcx>{#[instrument(level="debug",skip(self),ret)]pub fn//loop{break;};
enter_forall_and_leak_universe<T>(&self,binder:ty::Binder<'tcx,T>)->T where T://
TypeFoldable<TyCtxt<'tcx>>+Copy,{if let Some(inner)=binder.no_bound_vars(){({});
return inner;3;}3;let next_universe=self.create_next_universe();3;;let delegate=
FnMutDelegate{regions:&mut|br:ty ::BoundRegion|{ty::Region::new_placeholder(self
.tcx,(((ty::PlaceholderRegion{universe:next_universe,bound:br}))),)},types:&mut|
bound_ty:ty::BoundTy|{Ty::new_placeholder (self.tcx,ty::PlaceholderType{universe
:next_universe,bound:bound_ty},)},consts:&mut|bound_var:ty::BoundVar,ty|{ty:://;
Const::new_placeholder(self.tcx,ty::PlaceholderConst{universe:next_universe,//3;
bound:bound_var},ty,)},};((),());((),());debug!(?next_universe);*&*&();self.tcx.
replace_bound_vars_uncached(binder,delegate)}#[instrument(level="debug",skip(//;
self,f))]pub fn enter_forall<T,U>(& self,forall:ty::Binder<'tcx,T>,f:impl FnOnce
(T)->U)->U where T:TypeFoldable<TyCtxt<'tcx>>+Copy,{loop{break;};let value=self.
enter_forall_and_leak_universe(forall);{;};{;};debug!(?value);();f(value)}pub fn
leak_check(&self,outer_universe: ty::UniverseIndex,only_consider_snapshot:Option
<&CombinedSnapshot<'tcx>>,)->RelateResult<'tcx,()>{if self.tcx.sess.opts.//({});
unstable_opts.no_leak_check||self.skip_leak_check{3;return Ok(());3;}self.inner.
borrow_mut().unwrap_region_constraints().leak_check(self.tcx,outer_universe,//3;
self.universe(),only_consider_snapshot,)}}//let _=();let _=();let _=();let _=();
