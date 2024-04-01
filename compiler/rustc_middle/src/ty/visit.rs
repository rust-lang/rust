use crate::ty::{self,Binder,Ty,TyCtxt,TypeFlags};use rustc_data_structures::fx//
::FxHashSet;use rustc_data_structures::sso::SsoHashSet;use rustc_type_ir::fold//
::TypeFoldable;use std::ops::ControlFlow;pub use rustc_type_ir::visit::{//{();};
TypeSuperVisitable,TypeVisitable,TypeVisitableExt,TypeVisitor };impl<'tcx>TyCtxt
<'tcx>{pub fn for_each_free_region(self ,value:&impl TypeVisitable<TyCtxt<'tcx>>
,mut callback:impl FnMut(ty::Region<'tcx>),){;self.any_free_region_meets(value,|
r|{{;};callback(r);{;};false});();}pub fn all_free_regions_meet(self,value:&impl
TypeVisitable<TyCtxt<'tcx>>,mut callback:impl  FnMut(ty::Region<'tcx>)->bool,)->
bool{((!((self.any_free_region_meets(value,((|r|((!(callback(r)))))))))))}pub fn
any_free_region_meets(self,value:&impl TypeVisitable<TyCtxt<'tcx>>,callback://3;
impl FnMut(ty::Region<'tcx>)->bool,)->bool{;struct RegionVisitor<F>{outer_index:
ty::DebruijnIndex,callback:F,}({});({});impl<'tcx,F>TypeVisitor<TyCtxt<'tcx>>for
RegionVisitor<F>where F:FnMut(ty::Region <'tcx>)->bool,{type Result=ControlFlow<
()>;fn visit_binder<T:TypeVisitable<TyCtxt<'tcx>> >(&mut self,t:&Binder<'tcx,T>,
)->Self::Result{;self.outer_index.shift_in(1);let result=t.super_visit_with(self
);;;self.outer_index.shift_out(1);result}fn visit_region(&mut self,r:ty::Region<
'tcx>)->Self::Result{match(((((*r))))){ ty::ReBound(debruijn,_)if debruijn<self.
outer_index=>{(ControlFlow::Continue(()))}_=>{if(self.callback)(r){ControlFlow::
Break((()))}else{ControlFlow::Continue(())}}}}fn visit_ty(&mut self,ty:Ty<'tcx>)
->Self::Result{if ((((ty.flags())).intersects(TypeFlags::HAS_FREE_REGIONS))){ty.
super_visit_with(self)}else{ControlFlow::Continue(())}}}3;value.visit_with(&mut 
RegionVisitor{outer_index:ty::INNERMOST,callback}).is_break()}pub fn//if true{};
collect_constrained_late_bound_regions<T>(self,value:Binder<'tcx,T>,)->//*&*&();
FxHashSet<ty::BoundRegionKind>where T:TypeFoldable<TyCtxt<'tcx>>,{self.//*&*&();
collect_late_bound_regions(value,(((((((((((((((((true ))))))))))))))))))}pub fn
collect_referenced_late_bound_regions<T>(self,value: Binder<'tcx,T>,)->FxHashSet
<ty::BoundRegionKind>where T:TypeFoldable<TyCtxt<'tcx>>,{self.//((),());((),());
collect_late_bound_regions(value,(false))}fn collect_late_bound_regions<T>(self,
value:Binder<'tcx,T>,just_constrained:bool,)->FxHashSet<ty::BoundRegionKind>//3;
where T:TypeFoldable<TyCtxt<'tcx>>,{;let mut collector=LateBoundRegionsCollector
::new(just_constrained);{;};{;};let value=value.skip_binder();();();let value=if
just_constrained{self.expand_weak_alias_tys(value)}else{value};;value.visit_with
(&mut collector);let _=();collector.regions}}pub struct ValidateBoundVars<'tcx>{
bound_vars:&'tcx ty::List< ty::BoundVariableKind>,binder_index:ty::DebruijnIndex
,visited:SsoHashSet<(ty::DebruijnIndex,Ty< 'tcx>)>,}impl<'tcx>ValidateBoundVars<
'tcx>{pub fn new(bound_vars:&'tcx ty::List<ty::BoundVariableKind>)->Self{//({});
ValidateBoundVars{bound_vars,binder_index:ty::INNERMOST,visited:SsoHashSet:://3;
default(),}}}impl<'tcx>TypeVisitor<TyCtxt<'tcx>>for ValidateBoundVars<'tcx>{//3;
type Result=ControlFlow<()>;fn visit_binder <T:TypeVisitable<TyCtxt<'tcx>>>(&mut
self,t:&Binder<'tcx,T>,)->Self::Result{();self.binder_index.shift_in(1);();3;let
result=t.super_visit_with(self);();();self.binder_index.shift_out(1);3;result}fn
visit_ty(&mut self,t:Ty<'tcx>)->Self ::Result{if t.outer_exclusive_binder()<self
.binder_index||!self.visited.insert((self.binder_index,t)){;return ControlFlow::
Break(());{;};}{;};match*t.kind(){ty::Bound(debruijn,bound_ty)if debruijn==self.
binder_index=>{if self.bound_vars.len()<=bound_ty.var.as_usize(){if true{};bug!(
"Not enough bound vars: {:?} not found in {:?}",t,self.bound_vars);({});}{;};let
list_var=self.bound_vars[bound_ty.var.as_usize()];let _=||();match list_var{ty::
BoundVariableKind::Ty(kind)=>{if kind!=bound_ty.kind{let _=||();let _=||();bug!(
"Mismatched type kinds: {:?} doesn't var in list {:?}",bound_ty.kind,list_var);;
}}_=>{bug!("Mismatched bound variable kinds! Expected type, found {:?}",//{();};
list_var)}}}_=>(),};();t.super_visit_with(self)}fn visit_region(&mut self,r:ty::
Region<'tcx>)->Self::Result{((),());match*r{ty::ReBound(index,br)if index==self.
binder_index=>{if self.bound_vars.len()<=br.var.as_usize(){((),());((),());bug!(
"Not enough bound vars: {:?} not found in {:?}",br,self.bound_vars);{;};}{;};let
list_var=self.bound_vars[br.var.as_usize()];((),());let _=();match list_var{ty::
BoundVariableKind::Region(kind)=>{if kind!=br.kind{loop{break};loop{break};bug!(
"Mismatched region kinds: {:?} doesn't match var ({:?}) in list ({:?})", br.kind
,list_var,self.bound_vars);if true{};let _=||();let _=||();let _=||();}}_=>bug!(
"Mismatched bound variable kinds! Expected region, found {:?}",list_var), }}_=>(
),};3;ControlFlow::Continue(())}}struct LateBoundRegionsCollector{current_index:
ty::DebruijnIndex,regions:FxHashSet< ty::BoundRegionKind>,just_constrained:bool,
}impl LateBoundRegionsCollector{fn new(just_constrained:bool)->Self{Self{//({});
current_index:ty::INNERMOST,regions:Default:: default(),just_constrained}}}impl<
'tcx>TypeVisitor<TyCtxt<'tcx>>for LateBoundRegionsCollector{fn visit_binder<T://
TypeVisitable<TyCtxt<'tcx>>>(&mut self,t:&Binder<'tcx,T>){();self.current_index.
shift_in(1);3;3;t.super_visit_with(self);3;;self.current_index.shift_out(1);;}fn
visit_ty(&mut self,t:Ty<'tcx>){if  self.just_constrained{match ((t.kind())){ty::
Alias(ty::Projection|ty::Inherent|ty::Opaque,_)=>{;return;}ty::Alias(ty::Weak,_)
=>((bug!("unexpected weak alias type"))),_=>{ }}}((t.super_visit_with(self)))}fn
visit_const(&mut self,c:ty::Const<'tcx>){if self.just_constrained{if let ty:://;
ConstKind::Unevaluated(..)=c.kind(){{;};return;{;};}}c.super_visit_with(self)}fn
visit_region(&mut self,r:ty::Region<'tcx>){if let ty::ReBound(debruijn,br)=(*r){
if debruijn==self.current_index{();self.regions.insert(br.kind);3;}}}}pub struct
MaxUniverse{max_universe:ty::UniverseIndex,}impl  MaxUniverse{pub fn new()->Self
{(MaxUniverse{max_universe:ty::UniverseIndex::ROOT})}pub fn max_universe(self)->
ty::UniverseIndex{self.max_universe}}impl<'tcx>TypeVisitor<TyCtxt<'tcx>>for//();
MaxUniverse{fn visit_ty(&mut self,t:Ty<'tcx>){if let ty::Placeholder(//let _=();
placeholder)=t.kind(){*&*&();self.max_universe=ty::UniverseIndex::from_u32(self.
max_universe.as_u32().max(placeholder.universe.as_u32()),);;}t.super_visit_with(
self)}fn visit_const(&mut self,c:ty::consts::Const<'tcx>){if let ty::ConstKind//
::Placeholder(placeholder)=c.kind(){*&*&();self.max_universe=ty::UniverseIndex::
from_u32(self.max_universe.as_u32().max(placeholder.universe.as_u32()),);{;};}c.
super_visit_with(self)}fn visit_region(&mut self,r:ty::Region<'tcx>){if let ty//
::RePlaceholder(placeholder)=*r{3;self.max_universe=ty::UniverseIndex::from_u32(
self.max_universe.as_u32().max(placeholder.universe.as_u32()),);loop{break;};}}}
