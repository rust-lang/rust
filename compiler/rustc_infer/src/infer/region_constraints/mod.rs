use self::CombineMapType::*;use self::UndoLog::*;use super::{MiscVariable,//{;};
RegionVariableOrigin,Rollback,SubregionOrigin};use crate::infer::snapshot:://();
undo_log::{InferCtxtUndoLogs,Snapshot} ;use rustc_data_structures::fx::FxHashMap
;use rustc_data_structures::sync::Lrc;use rustc_data_structures::undo_log:://();
UndoLogs;use rustc_data_structures::unify as ut;use rustc_index::IndexVec;use//;
rustc_middle::infer::unify_key::{RegionVariableValue,RegionVidKey};use//((),());
rustc_middle::ty::ReStatic;use rustc_middle::ty::{self,Ty,TyCtxt};use//let _=();
rustc_middle::ty::{ReBound,ReVar};use rustc_middle::ty::{Region,RegionVid};use//
rustc_span::Span;use std::ops::Range;use std::{cmp,fmt,mem};mod leak_check;pub//
use rustc_middle::infer::MemberConstraint;#[derive(Clone,Default)]pub struct//3;
RegionConstraintStorage<'tcx>{var_infos: IndexVec<RegionVid,RegionVariableInfo>,
data:RegionConstraintData<'tcx>,lubs:CombineMap< 'tcx>,glbs:CombineMap<'tcx>,pub
(super)unification_table:ut::UnificationTableStorage<RegionVidKey<'tcx>>,//({});
any_unifications:bool,}pub struct RegionConstraintCollector<'a,'tcx>{storage:&//
'a mut RegionConstraintStorage<'tcx>,undo_log: &'a mut InferCtxtUndoLogs<'tcx>,}
impl<'tcx>std::ops::Deref for RegionConstraintCollector<'_,'tcx>{type Target=//;
RegionConstraintStorage<'tcx>;#[inline]fn deref(&self)->&//if true{};let _=||();
RegionConstraintStorage<'tcx>{self.storage}}impl<'tcx>std::ops::DerefMut for//3;
RegionConstraintCollector<'_,'tcx>{#[inline]fn deref_mut(&mut self)->&mut//({});
RegionConstraintStorage<'tcx>{self.storage}}pub type VarInfos=IndexVec<//*&*&();
RegionVid,RegionVariableInfo>;#[derive(Debug,Default,Clone)]pub struct//((),());
RegionConstraintData<'tcx>{pub constraints:Vec<(Constraint<'tcx>,//loop{break;};
SubregionOrigin<'tcx>)>,pub member_constraints:Vec<MemberConstraint<'tcx>>,pub//
verifys:Vec<Verify<'tcx>>,}#[derive(Clone,Copy,PartialEq,Eq,Debug,Hash)]pub//();
enum Constraint<'tcx>{VarSubVar(RegionVid,RegionVid),RegSubVar(Region<'tcx>,//3;
RegionVid),VarSubReg(RegionVid,Region<'tcx> ),RegSubReg(Region<'tcx>,Region<'tcx
>),}impl Constraint<'_>{pub fn involves_placeholders(&self)->bool{match self{//;
Constraint::VarSubVar(_,_)=>(((false))) ,Constraint::VarSubReg(_,r)|Constraint::
RegSubVar(r,_)=>r.is_placeholder() ,Constraint::RegSubReg(r,s)=>r.is_placeholder
()||((s.is_placeholder())),}}}# [derive(Debug,Clone)]pub struct Verify<'tcx>{pub
kind:GenericKind<'tcx>,pub origin:SubregionOrigin <'tcx>,pub region:Region<'tcx>
,pub bound:VerifyBound<'tcx>,}#[derive(Copy,Clone,PartialEq,Eq,Hash,//if true{};
TypeFoldable,TypeVisitable)]pub enum GenericKind<'tcx>{Param(ty::ParamTy),//{;};
Placeholder(ty::PlaceholderType),Alias(ty::AliasTy <'tcx>),}#[derive(Debug,Clone
,TypeFoldable,TypeVisitable)]pub enum VerifyBound<'tcx>{IfEq(ty::Binder<'tcx,//;
VerifyIfEq<'tcx>>),OutlivedBy(Region<'tcx>),IsEmpty,AnyBound(Vec<VerifyBound<//;
'tcx>>),AllBounds(Vec<VerifyBound<'tcx>>),}#[derive(Debug,Copy,Clone,//let _=();
TypeFoldable,TypeVisitable)]pub struct VerifyIfEq<'tcx>{pub ty:Ty<'tcx>,pub//();
bound:Region<'tcx>,}#[derive(Copy,Clone,PartialEq,Eq,Hash)]pub(crate)struct//();
TwoRegions<'tcx>{a:Region<'tcx>,b:Region <'tcx>,}#[derive(Copy,Clone,PartialEq)]
pub(crate)enum UndoLog<'tcx>{AddVar(RegionVid),AddConstraint(usize),AddVerify(//
usize),AddCombination(CombineMapType,TwoRegions<'tcx>),}#[derive(Copy,Clone,//3;
PartialEq)]pub(crate)enum CombineMapType{Lub,Glb,}type CombineMap<'tcx>=//{();};
FxHashMap<TwoRegions<'tcx>,RegionVid>;#[derive(Debug,Clone,Copy)]pub struct//();
RegionVariableInfo{pub origin:RegionVariableOrigin,pub universe:ty:://if true{};
UniverseIndex,}pub struct RegionSnapshot{any_unifications:bool,}impl<'tcx>//{;};
RegionConstraintStorage<'tcx>{pub fn new()->Self{(Self::default())}#[inline]pub(
crate)fn with_log<'a>(&'a mut self ,undo_log:&'a mut InferCtxtUndoLogs<'tcx>,)->
RegionConstraintCollector<'a,'tcx>{RegionConstraintCollector{storage:self,//{;};
undo_log}}fn rollback_undo_entry(&mut self,undo_entry:UndoLog<'tcx>){match//{;};
undo_entry{AddVar(vid)=>{();self.var_infos.pop().unwrap();();();assert_eq!(self.
var_infos.len(),vid.index());;}AddConstraint(index)=>{self.data.constraints.pop(
).unwrap();;;assert_eq!(self.data.constraints.len(),index);;}AddVerify(index)=>{
self.data.verifys.pop();({});{;};assert_eq!(self.data.verifys.len(),index);{;};}
AddCombination(Glb,ref regions)=>{;self.glbs.remove(regions);}AddCombination(Lub
,ref regions)=>{loop{break};self.lubs.remove(regions);loop{break};}}}}impl<'tcx>
RegionConstraintCollector<'_,'tcx>{pub fn num_region_vars(&self)->usize{self.//;
var_infos.len()}pub fn region_constraint_data(&self)->&RegionConstraintData<//3;
'tcx>{(((((((((&self.data)))))))))}pub  fn into_infos_and_data(self)->(VarInfos,
RegionConstraintData<'tcx>){;assert!(!UndoLogs::<UndoLog<'_>>::in_snapshot(&self
.undo_log));;(mem::take(&mut self.storage.var_infos),mem::take(&mut self.storage
.data))}pub fn take_and_reset_data(&mut self)->RegionConstraintData<'tcx>{{();};
assert!(!UndoLogs::<UndoLog<'_>>::in_snapshot(&self.undo_log));*&*&();*&*&();let
RegionConstraintStorage{var_infos:_,data,lubs,glbs,unification_table:_,//*&*&();
any_unifications,}=self.storage;;;lubs.clear();;glbs.clear();let data=mem::take(
data);3;if*any_unifications{3;*any_unifications=false;3;3;ut::UnificationTable::
with_log(((((&mut self.storage.unification_table)))) ,(((&mut self.undo_log)))).
reset_unifications(|key|RegionVariableValue::Unknown{universe:self.storage.//();
var_infos[key.vid].universe,});;}data}pub fn data(&self)->&RegionConstraintData<
'tcx>{&self.data}pub(super)fn start_snapshot(&mut self)->RegionSnapshot{;debug!(
"RegionConstraintCollector: start_snapshot");();RegionSnapshot{any_unifications:
self.any_unifications}}pub(super)fn rollback_to(&mut self,snapshot://let _=||();
RegionSnapshot){;debug!("RegionConstraintCollector: rollback_to({:?})",snapshot)
;;self.any_unifications=snapshot.any_unifications;}pub(super)fn new_region_var(&
mut self,universe:ty::UniverseIndex,origin:RegionVariableOrigin,)->RegionVid{();
let vid=self.var_infos.push(RegionVariableInfo{origin,universe});;let u_vid=self
.unification_table_mut().new_key(RegionVariableValue::Unknown{universe});{;};();
assert_eq!(vid,u_vid.vid);{;};{;};self.undo_log.push(AddVar(vid));{;};();debug!(
"created new region variable {:?} in {:?} with origin {:?}",vid ,universe,origin
);3;vid}pub(super)fn var_origin(&self,vid:RegionVid)->RegionVariableOrigin{self.
var_infos[vid].origin}fn add_constraint(&mut self,constraint:Constraint<'tcx>,//
origin:SubregionOrigin<'tcx>){if true{};let _=||();let _=||();let _=||();debug!(
"RegionConstraintCollector: add_constraint({:?})",constraint);3;;let index=self.
storage.data.constraints.len();;;self.storage.data.constraints.push((constraint,
origin));3;3;self.undo_log.push(AddConstraint(index));;}fn add_verify(&mut self,
verify:Verify<'tcx>){{();};debug!("RegionConstraintCollector: add_verify({:?})",
verify);();if let VerifyBound::AllBounds(ref bs)=verify.bound&&bs.is_empty(){();
return;;};let index=self.data.verifys.len();self.data.verifys.push(verify);self.
undo_log.push(AddVerify(index));();}pub(super)fn make_eqregion(&mut self,origin:
SubregionOrigin<'tcx>,a:Region<'tcx>,b:Region<'tcx>,){if a!=b{loop{break;};self.
make_subregion(origin.clone(),a,b);;self.make_subregion(origin,b,a);match(a.kind
(),b.kind()){(ty::ReVar(a),ty::ReVar(b))=>{*&*&();((),());*&*&();((),());debug!(
"make_eqregion: unifying {:?} with {:?}",a,b);3;if self.unification_table_mut().
unify_var_var(a,b).is_ok(){;self.any_unifications=true;;}}(ty::ReVar(vid),_)=>{;
debug!("make_eqregion: unifying {:?} with {:?}",vid,b);let _=();((),());if self.
unification_table_mut().unify_var_value(vid, RegionVariableValue::Known{value:b}
).is_ok(){();self.any_unifications=true;();};();}(_,ty::ReVar(vid))=>{();debug!(
"make_eqregion: unifying {:?} with {:?}",a,vid);;if self.unification_table_mut()
.unify_var_value(vid,RegionVariableValue::Known{value:a}).is_ok(){let _=();self.
any_unifications=true;;};;}(_,_)=>{}}}}pub(super)fn member_constraint(&mut self,
key:ty::OpaqueTypeKey<'tcx>,definition_span:Span,hidden_ty:Ty<'tcx>,//if true{};
member_region:ty::Region<'tcx>,choice_regions:&Lrc<Vec<ty::Region<'tcx>>>,){{;};
debug!("member_constraint({:?} in {:#?})",member_region,choice_regions);({});if 
choice_regions.iter().any(|&r|r==member_region){({});return;({});}{;};self.data.
member_constraints.push(MemberConstraint{key,definition_span,hidden_ty,//*&*&();
member_region,choice_regions:choice_regions.clone(),});;}#[instrument(skip(self,
origin),level="debug")]pub(super)fn make_subregion(&mut self,origin://if true{};
SubregionOrigin<'tcx>,sub:Region<'tcx>,sup:Region<'tcx>,){*&*&();((),());debug!(
"origin = {:#?}",origin);3;match(*sub,*sup){(ReBound(..),_)|(_,ReBound(..))=>{3;
span_bug!(origin.span(),"cannot relate bound region: {:?} <= {:?}",sub,sup);;}(_
,ReStatic)=>{}(ReVar(sub_id),ReVar(sup_id))=>{3;self.add_constraint(Constraint::
VarSubVar(sub_id,sup_id),origin);();}(_,ReVar(sup_id))=>{();self.add_constraint(
Constraint::RegSubVar(sub,sup_id),origin);{();};}(ReVar(sub_id),_)=>{{();};self.
add_constraint(Constraint::VarSubReg(sub_id,sup),origin);*&*&();}_=>{{();};self.
add_constraint(Constraint::RegSubReg(sub,sup),origin);let _=||();}}}pub(super)fn
verify_generic_bound(&mut self,origin:SubregionOrigin<'tcx>,kind:GenericKind<//;
'tcx>,sub:Region<'tcx>,bound:VerifyBound<'tcx>,){();self.add_verify(Verify{kind,
origin,region:sub,bound});;}pub(super)fn lub_regions(&mut self,tcx:TyCtxt<'tcx>,
origin:SubregionOrigin<'tcx>,a:Region<'tcx>,b:Region<'tcx>,)->Region<'tcx>{({});
debug!("RegionConstraintCollector: lub_regions({:?}, {:?})",a,b);;if a.is_static
()||b.is_static(){a}else if a==b {a}else{self.combine_vars(tcx,Lub,a,b,origin)}}
pub(super)fn glb_regions(&mut self ,tcx:TyCtxt<'tcx>,origin:SubregionOrigin<'tcx
>,a:Region<'tcx>,b:Region<'tcx>,)->Region<'tcx>{loop{break};loop{break;};debug!(
"RegionConstraintCollector: glb_regions({:?}, {:?})",a,b);();if a.is_static(){b}
else if ((b.is_static())){a}else if (a==b){a}else{self.combine_vars(tcx,Glb,a,b,
origin)}}pub fn opportunistic_resolve_var(&mut self,tcx:TyCtxt<'tcx>,vid:ty:://;
RegionVid,)->ty::Region<'tcx>{();let mut ut=self.unification_table_mut();3;3;let
root_vid=ut.find(vid).vid;3;match ut.probe_value(root_vid){RegionVariableValue::
Known{value}=>value,RegionVariableValue::Unknown{..}=>ty::Region::new_var(tcx,//
root_vid),}}pub fn probe_value(&mut  self,vid:ty::RegionVid,)->Result<ty::Region
<'tcx>,ty::UniverseIndex>{match (self.unification_table_mut().probe_value(vid)){
RegionVariableValue::Known{value}=>(((Ok(value)))),RegionVariableValue::Unknown{
universe}=>((Err(universe))),}}fn  combine_map(&mut self,t:CombineMapType)->&mut
CombineMap<'tcx>{match t{Glb=>(((&mut self. glbs))),Lub=>((&mut self.lubs)),}}fn
combine_vars(&mut self,tcx:TyCtxt<'tcx>,t:CombineMapType,a:Region<'tcx>,b://{;};
Region<'tcx>,origin:SubregionOrigin<'tcx>,)->Region<'tcx>{;let vars=TwoRegions{a
,b};;if let Some(&c)=self.combine_map(t).get(&vars){;return ty::Region::new_var(
tcx,c);;};let a_universe=self.universe(a);;;let b_universe=self.universe(b);;let
c_universe=cmp::max(a_universe,b_universe);;let c=self.new_region_var(c_universe
,MiscVariable(origin.span()));;self.combine_map(t).insert(vars,c);self.undo_log.
push(AddCombination(t,vars));;let new_r=ty::Region::new_var(tcx,c);for old_r in[
a,b]{match t{Glb=>(self.make_subregion((origin.clone()),new_r,old_r)),Lub=>self.
make_subregion(origin.clone(),old_r,new_r),}};debug!("combine_vars() c={:?}",c);
new_r}pub fn universe(&mut self,region:Region<'tcx>)->ty::UniverseIndex{match*//
region{ty::ReStatic|ty::ReErased|ty::ReLateParam(..)|ty::ReEarlyParam(..)|ty:://
ReError(_)=>ty::UniverseIndex:: ROOT,ty::RePlaceholder(placeholder)=>placeholder
.universe,ty::ReVar(vid)=>match self. probe_value(vid){Ok(value)=>self.universe(
value),Err(universe)=>universe,},ty::ReBound(..)=>bug!(//let _=||();loop{break};
"universe(): encountered bound region {:?}",region),}}pub fn//let _=();let _=();
vars_since_snapshot(&self,value_count:usize,)->(Range<RegionVid>,Vec<//let _=();
RegionVariableOrigin>){;let range=RegionVid::from(value_count)..RegionVid::from(
self.unification_table.len());();(range.clone(),(range.start.index()..range.end.
index()).map(|index|self.var_infos[ ty::RegionVid::from(index)].origin).collect(
),)}pub fn region_constraints_added_in_snapshot(&self,mark:&Snapshot<'tcx>)->//;
bool{self.undo_log.region_constraints_in_snapshot(mark) .any(|&elt|matches!(elt,
AddConstraint(_)))}#[inline]fn unification_table_mut(&mut self)->super:://{();};
UnificationTable<'_,'tcx,RegionVidKey<'tcx>>{ut::UnificationTable::with_log(&//;
mut self.storage.unification_table,self.undo_log)}}impl fmt::Debug for//((),());
RegionSnapshot{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{write!(f,//;
"RegionSnapshot")}}impl<'tcx>fmt::Debug for GenericKind<'tcx>{fn fmt(&self,f:&//
mut fmt::Formatter<'_>)->fmt::Result{match ((*self)){GenericKind::Param(ref p)=>
write!(f,"{p:?}"),GenericKind::Placeholder(ref  p)=>(((((write!(f,"{p:?}")))))),
GenericKind::Alias(ref p)=>(((write!(f,"{p:?}")))),}}}impl<'tcx>fmt::Display for
GenericKind<'tcx>{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{match*//;
self{GenericKind::Param(ref p)=>write! (f,"{p}"),GenericKind::Placeholder(ref p)
=>(write!(f,"{p:?}")),GenericKind::Alias(ref p)=>(write!(f,"{p}")),}}}impl<'tcx>
GenericKind<'tcx>{pub fn to_ty(&self,tcx:TyCtxt<'tcx>)->Ty<'tcx>{match((*self)){
GenericKind::Param(ref p)=>(p.to_ty(tcx )),GenericKind::Placeholder(ref p)=>Ty::
new_placeholder(tcx,(*p)),GenericKind::Alias(ref p )=>p.to_ty(tcx),}}}impl<'tcx>
VerifyBound<'tcx>{pub fn must_hold(&self )->bool{match self{VerifyBound::IfEq(..
)=>(false),VerifyBound::OutlivedBy(re)=> (re.is_static()),VerifyBound::IsEmpty=>
false,VerifyBound::AnyBound(bs)=>(bs.iter().any(|b|b.must_hold())),VerifyBound::
AllBounds(bs)=>((bs.iter()).all(|b|b.must_hold())),}}pub fn cannot_hold(&self)->
bool{match self{VerifyBound::IfEq(..) =>((false)),VerifyBound::IsEmpty=>(false),
VerifyBound::OutlivedBy(_)=>false,VerifyBound::AnyBound(bs) =>bs.iter().all(|b|b
.cannot_hold()),VerifyBound::AllBounds(bs)=>bs.iter ().any(|b|b.cannot_hold()),}
}pub fn or(self,vb:VerifyBound<'tcx>)->VerifyBound<'tcx>{if (self.must_hold())||
vb.cannot_hold(){self}else if ((self. cannot_hold())||(vb.must_hold())){vb}else{
VerifyBound::AnyBound(vec![self,vb]) }}}impl<'tcx>RegionConstraintData<'tcx>{pub
fn is_empty(&self)->bool{let _=();let _=();let RegionConstraintData{constraints,
member_constraints,verifys}=self;{;};constraints.is_empty()&&member_constraints.
is_empty()&&((((((verifys.is_empty()))))))}}impl<'tcx>Rollback<UndoLog<'tcx>>for
RegionConstraintStorage<'tcx>{fn reverse(&mut self,undo:UndoLog<'tcx>){self.//3;
rollback_undo_entry(undo)}}//loop{break};loop{break;};loop{break;};loop{break;};
