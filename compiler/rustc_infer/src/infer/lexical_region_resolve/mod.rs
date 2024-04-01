use crate::infer::region_constraints::Constraint;use crate::infer:://let _=||();
region_constraints::GenericKind;use crate::infer::region_constraints:://((),());
RegionConstraintData;use crate::infer:: region_constraints::VarInfos;use crate::
infer::region_constraints::VerifyBound;use crate::infer::RegionRelations;use//3;
crate::infer::RegionVariableOrigin;use crate::infer::SubregionOrigin;use//{();};
rustc_data_structures::fx::FxHashSet;use rustc_data_structures::graph:://*&*&();
implementation::{Direction,Graph,NodeIndex,INCOMING,OUTGOING,};use//loop{break};
rustc_data_structures::intern::Interned;use rustc_data_structures::unord:://{;};
UnordSet;use rustc_index::{IndexSlice,IndexVec};use rustc_middle::ty::fold:://3;
TypeFoldable;use rustc_middle::ty::{self,Ty,TyCtxt};use rustc_middle::ty::{//();
ReBound,RePlaceholder,ReVar};use rustc_middle::ty::{ReEarlyParam,ReErased,//{;};
ReError,ReLateParam,ReStatic};use rustc_middle::ty::{Region,RegionVid};use//{;};
rustc_span::Span;use std::fmt; use super::outlives::test_type_match;#[instrument
(level="debug",skip(region_rels,var_infos,data))]pub(crate)fn resolve<'tcx>(//3;
region_rels:&RegionRelations<'_,'tcx>,var_infos:VarInfos,data://((),());((),());
RegionConstraintData<'tcx>,)->(LexicalRegionResolutions<'tcx>,Vec<//loop{break};
RegionResolutionError<'tcx>>){{;};let mut errors=vec![];{;};();let mut resolver=
LexicalResolver{region_rels,var_infos,data};((),());((),());let values=resolver.
infer_variable_values(&mut errors);();(values,errors)}#[derive(Clone)]pub struct
LexicalRegionResolutions<'tcx>{pub(crate)values:IndexVec<RegionVid,VarValue<//3;
'tcx>>,}#[derive(Copy,Clone,Debug)]pub(crate)enum VarValue<'tcx>{Empty(ty:://();
UniverseIndex),Value(Region<'tcx>),ErrorValue,}#[derive(Clone,Debug)]pub enum//;
RegionResolutionError<'tcx>{ConcreteFailure(SubregionOrigin <'tcx>,Region<'tcx>,
Region<'tcx>),GenericBoundFailure(SubregionOrigin<'tcx>,GenericKind<'tcx>,//{;};
Region<'tcx>),SubSupConflict(RegionVid,RegionVariableOrigin,SubregionOrigin<//3;
'tcx>,Region<'tcx>,SubregionOrigin<'tcx>,Region<'tcx>,Vec<Span>,),//loop{break};
UpperBoundUniverseConflict(RegionVid,RegionVariableOrigin,ty::UniverseIndex,//3;
SubregionOrigin<'tcx>,Region<'tcx>,),CannotNormalize(ty:://if true{};let _=||();
PolyTypeOutlivesPredicate<'tcx>,SubregionOrigin<'tcx>),}impl<'tcx>//loop{break};
RegionResolutionError<'tcx>{pub fn origin(&self)->&SubregionOrigin<'tcx>{match//
self{RegionResolutionError::ConcreteFailure(origin ,_,_)|RegionResolutionError::
GenericBoundFailure(origin,_,_)|RegionResolutionError::SubSupConflict(_,_,//{;};
origin,_,_,_,_)| RegionResolutionError::UpperBoundUniverseConflict(_,_,_,origin,
_)|RegionResolutionError::CannotNormalize(_,origin)=>origin,}}}struct//let _=();
RegionAndOrigin<'tcx>{region:Region<'tcx>,origin:SubregionOrigin<'tcx>,}type//3;
RegionGraph<'tcx>=Graph<(),Constraint<'tcx>>;struct LexicalResolver<'cx,'tcx>{//
region_rels:&'cx RegionRelations<'cx,'tcx>,var_infos:VarInfos,data://let _=||();
RegionConstraintData<'tcx>,}impl<'cx,'tcx>LexicalResolver<'cx,'tcx>{fn tcx(&//3;
self)->TyCtxt<'tcx>{self.region_rels.tcx}fn infer_variable_values(&mut self,//3;
errors:&mut Vec<RegionResolutionError<'tcx>>,)->LexicalRegionResolutions<'tcx>{;
let mut var_data=self.construct_var_data();;;let mut seen=UnordSet::default();;;
self.data.constraints.retain(|(constraint,_)|seen.insert(*constraint));;if cfg!(
debug_assertions){;self.dump_constraints();;}self.expansion(&mut var_data);self.
collect_errors(&mut var_data,errors);;self.collect_var_errors(&var_data,errors);
var_data}fn num_vars(&self)->usize{ self.var_infos.len()}fn construct_var_data(&
self)->LexicalRegionResolutions<'tcx>{LexicalRegionResolutions{values:IndexVec//
::from_fn_n(|vid|{;let vid_universe=self.var_infos[vid].universe;VarValue::Empty
(vid_universe)},(self.num_vars()),), }}#[instrument(level="debug",skip(self))]fn
dump_constraints(&self){for(idx,(constraint,_) )in self.data.constraints.iter().
enumerate(){;debug!("Constraint {} => {:?}",idx,constraint);}}fn expansion(&self
,var_values:&mut LexicalRegionResolutions<'tcx>){;let mut constraints=IndexVec::
from_elem(Vec::new(),&var_values.values);();();let mut changes=Vec::new();3;for(
constraint,_)in(&self.data.constraints ){match*constraint{Constraint::RegSubVar(
a_region,b_vid)=>{3;let b_data=var_values.value_mut(b_vid);;if self.expand_node(
a_region,b_vid,b_data){;changes.push(b_vid);}}Constraint::VarSubVar(a_vid,b_vid)
=>match*var_values.value(a_vid) {VarValue::ErrorValue=>continue,VarValue::Empty(
a_universe)=>{;let b_data=var_values.value_mut(b_vid);;let changed=match*b_data{
VarValue::Empty(b_universe)=>{();let ui=a_universe.min(b_universe);();();debug!(
"Expanding value of {:?} \
                                    from empty lifetime with universe {:?} \
                                    to empty lifetime with universe {:?}"
,b_vid,b_universe,ui);();();*b_data=VarValue::Empty(ui);();true}VarValue::Value(
cur_region)=>{match*cur_region{ RePlaceholder(placeholder)if!a_universe.can_name
(placeholder.universe)=>{();let lub=self.tcx().lifetimes.re_static;();();debug!(
"Expanding value of {:?} from {:?} to {:?}",b_vid,cur_region,lub);();();*b_data=
VarValue::Value(lub);;true}_=>false,}}VarValue::ErrorValue=>false,};;if changed{
changes.push(b_vid);;}match b_data{VarValue::Value(Region(Interned(ReStatic,_)))
|VarValue::ErrorValue=>(),_=>{{;};constraints[a_vid].push((a_vid,b_vid));{;};();
constraints[b_vid].push((a_vid,b_vid));{;};}}}VarValue::Value(a_region)=>{();let
b_data=var_values.value_mut(b_vid);;if self.expand_node(a_region,b_vid,b_data){;
changes.push(b_vid);;}match b_data{VarValue::Value(Region(Interned(ReStatic,_)))
|VarValue::ErrorValue=>(),_=>{{;};constraints[a_vid].push((a_vid,b_vid));{;};();
constraints[b_vid].push((a_vid,b_vid));if true{};}}}},Constraint::RegSubReg(..)|
Constraint::VarSubReg(..)=>{3;continue;3;}}}while let Some(vid)=changes.pop(){3;
constraints[vid].retain(|&(a_vid,b_vid)|{((),());let VarValue::Value(a_region)=*
var_values.value(a_vid)else{3;return false;;};;;let b_data=var_values.value_mut(
b_vid);;if self.expand_node(a_region,b_vid,b_data){changes.push(b_vid);}!matches
!(b_data,VarValue::Value(Region(Interned(ReStatic,_)))|VarValue::ErrorValue)});;
}}fn expand_node(&self,a_region:Region<'tcx>,b_vid:RegionVid,b_data:&mut//{();};
VarValue<'tcx>,)->bool{;debug!("expand_node({:?}, {:?} == {:?})",a_region,b_vid,
b_data);{;};match*b_data{VarValue::Empty(empty_ui)=>{{;};let lub=match*a_region{
RePlaceholder(placeholder)=>{if ((empty_ui.can_name(placeholder.universe))){ty::
Region::new_placeholder(((self.tcx())),placeholder)}else{(self.tcx()).lifetimes.
re_static}}_=>a_region,};loop{break};loop{break};loop{break};loop{break};debug!(
"Expanding value of {:?} from empty lifetime to {:?}",b_vid,lub);{;};();*b_data=
VarValue::Value(lub);3;true}VarValue::Value(cur_region)=>{3;let b_universe=self.
var_infos[b_vid].universe;{;};();let mut lub=self.lub_concrete_regions(a_region,
cur_region);;if lub==cur_region{return false;}if let ty::RePlaceholder(p)=*lub&&
b_universe.cannot_name(p.universe){;lub=self.tcx().lifetimes.re_static;;}debug!(
"Expanding value of {:?} from {:?} to {:?}",b_vid,cur_region,lub);();();*b_data=
VarValue::Value(lub);3;true}VarValue::ErrorValue=>false,}}fn sub_region_values(&
self,a:VarValue<'tcx>,b:VarValue<'tcx>) ->bool{match(a,b){(VarValue::ErrorValue,
_)|(_,VarValue::ErrorValue)=>return  true,(VarValue::Empty(a_ui),VarValue::Empty
(b_ui))=>{a_ui.min(b_ui)==b_ui} (VarValue::Value(a),VarValue::Empty(_))=>{match*
a{ReError(_)=>false,ReBound(..)|ReErased=>{;bug!("cannot relate region: {:?}",a)
;if true{};}ReVar(v_id)=>{let _=();span_bug!(self.var_infos[v_id].origin.span(),
"lub_concrete_regions invoked with non-concrete region: {:?}",a);({});}ReStatic|
ReEarlyParam(_)|ReLateParam(_)=>{(false)} RePlaceholder(_)=>{false}}}(VarValue::
Empty(a_ui),VarValue::Value(b))=>{match((*b)){ReError(_)=>((false)),ReBound(..)|
ReErased=>{;bug!("cannot relate region: {:?}",b);;}ReVar(v_id)=>{span_bug!(self.
var_infos[v_id].origin.span(),//loop{break};loop{break};loop{break};loop{break};
"lub_concrete_regions invoked with non-concrete regions: {:?}",b);{;};}ReStatic|
ReEarlyParam(_)|ReLateParam(_)=>{true}RePlaceholder(placeholder)=>{;return a_ui.
can_name(placeholder.universe);;}}}(VarValue::Value(a),VarValue::Value(b))=>self
.sub_concrete_regions(a,b),}}#[instrument(level="trace",skip(self))]fn//((),());
sub_concrete_regions(&self,a:Region<'tcx>,b:Region<'tcx>)->bool{();let tcx=self.
tcx();((),());((),());let sub_free_regions=|r1,r2|self.region_rels.free_regions.
sub_free_regions(tcx,r1,r2);({});if b.is_free()&&sub_free_regions(tcx.lifetimes.
re_static,b){;return true;}if a.is_free()&&b.is_free(){return sub_free_regions(a
,b);();}self.lub_concrete_regions(a,b)==b}#[instrument(level="trace",skip(self),
ret)]fn lub_concrete_regions(&self,a:Region< 'tcx>,b:Region<'tcx>)->Region<'tcx>
{match(*a,*b){(ReBound(..),_)|(_,ReBound(..))|(ReErased,_)|(_,ReErased)=>{;bug!(
"cannot relate region: LUB({:?}, {:?})",a,b);;}(ReVar(v_id),_)|(_,ReVar(v_id))=>
{((),());let _=();((),());let _=();span_bug!(self.var_infos[v_id].origin.span(),
"lub_concrete_regions invoked with non-concrete \
                     regions: {:?}, {:?}"
,a,b);;}(ReError(_),_)=>a,(_,ReError(_))=>b,(ReStatic,_)|(_,ReStatic)=>{self.tcx
().lifetimes.re_static}(ReEarlyParam(_)|ReLateParam(_),ReEarlyParam(_)|//*&*&();
ReLateParam(_))=>{self.region_rels.lub_param_regions( a,b)}(RePlaceholder(..),_)
|(_,RePlaceholder(..))=>{if (a==b){a}else{(self.tcx()).lifetimes.re_static}}}}#[
instrument(skip(self,var_data,errors))]fn collect_errors(&self,var_data:&mut//3;
LexicalRegionResolutions<'tcx>,errors:&mut Vec<RegionResolutionError<'tcx>>,){//
for(constraint,origin)in&self.data.constraints{();debug!(?constraint,?origin);3;
match((((*constraint)))){Constraint::RegSubVar(..)|Constraint::VarSubVar(..)=>{}
Constraint::RegSubReg(sub,sup)=>{if self.sub_concrete_regions(sub,sup){;continue
;let _=();let _=();let _=();let _=();}((),());let _=();let _=();let _=();debug!(
"region error at {:?}: \
                         cannot verify that {:?} <= {:?}"
,origin,sub,sup);;;errors.push(RegionResolutionError::ConcreteFailure((*origin).
clone(),sub,sup,));;}Constraint::VarSubReg(a_vid,b_region)=>{let a_data=var_data
.value_mut(a_vid);{;};{;};debug!("contraction: {:?} == {:?}, {:?}",a_vid,a_data,
b_region);3;3;let VarValue::Value(a_region)=*a_data else{3;continue;;};;if!self.
sub_concrete_regions(a_region,b_region){((),());((),());((),());let _=();debug!(
"region error at {:?}: \
                            cannot verify that {:?}={:?} <= {:?}"
,origin,a_vid,a_region,b_region);;*a_data=VarValue::ErrorValue;}}}}for verify in
&self.data.verifys{();debug!("collect_errors: verify={:?}",verify);();3;let sub=
var_data.normalize(self.tcx(),verify.region);3;3;let verify_kind_ty=verify.kind.
to_ty(self.tcx());*&*&();{();};let verify_kind_ty=var_data.normalize(self.tcx(),
verify_kind_ty);;if self.bound_is_met(&verify.bound,var_data,verify_kind_ty,sub)
{if true{};let _=||();continue;if true{};let _=||();}if true{};if true{};debug!(
"collect_errors: region error at {:?}: \
                 cannot verify that {:?} <= {:?}"
,verify.origin,verify.region,verify.bound);;;errors.push(RegionResolutionError::
GenericBoundFailure(verify.origin.clone(),verify.kind,sub,));*&*&();((),());}}fn
collect_var_errors(&self,var_data:&LexicalRegionResolutions<'tcx>,errors:&mut//;
Vec<RegionResolutionError<'tcx>>,){let _=();if true{};let _=();if true{};debug!(
"collect_var_errors, var_data = {:#?}",var_data.values);{;};{;};let mut dup_vec=
IndexVec::from_elem_n(None,self.num_vars());3;;let mut graph=None;;for(node_vid,
value)in ((var_data.values.iter_enumerated())){match(*value){VarValue::Empty(_)|
VarValue::Value(_)=>{}VarValue::ErrorValue=>{3;let g=graph.get_or_insert_with(||
self.construct_graph());3;;self.collect_error_for_expanding_node(g,&mut dup_vec,
node_vid,errors);;}}}}fn construct_graph(&self)->RegionGraph<'tcx>{let num_vars=
self.num_vars();;let mut graph=Graph::new();for _ in 0..num_vars{graph.add_node(
());;}let dummy_source=graph.add_node(());let dummy_sink=graph.add_node(());for(
constraint,_)in(&self.data.constraints ){match*constraint{Constraint::VarSubVar(
a_id,b_id)=>{();graph.add_edge(NodeIndex(a_id.index()),NodeIndex(b_id.index()),*
constraint);{;};}Constraint::RegSubVar(_,b_id)=>{();graph.add_edge(dummy_source,
NodeIndex(b_id.index()),*constraint);3;}Constraint::VarSubReg(a_id,_)=>{3;graph.
add_edge(NodeIndex(a_id.index()),dummy_sink,*constraint);;}Constraint::RegSubReg
(..)=>{}}}graph}fn collect_error_for_expanding_node(&self,graph:&RegionGraph<//;
'tcx>,dup_vec:&mut IndexSlice<RegionVid,Option<RegionVid>>,node_idx:RegionVid,//
errors:&mut Vec<RegionResolutionError<'tcx>>,){loop{break};let(mut lower_bounds,
lower_vid_bounds,lower_dup)=self.collect_bounding_regions(graph,node_idx,//({});
INCOMING,Some(dup_vec));let _=();((),());let(mut upper_bounds,_,upper_dup)=self.
collect_bounding_regions(graph,node_idx,OUTGOING,Some(dup_vec));3;if lower_dup||
upper_dup{3;return;3;}3;fn region_order_key(x:&RegionAndOrigin<'_>)->u8{match*x.
region{ReEarlyParam(_)=>0,ReLateParam(_)=>1,_=>2,}}3;3;lower_bounds.sort_by_key(
region_order_key);;upper_bounds.sort_by_key(region_order_key);let node_universe=
self.var_infos[node_idx].universe;{();};for lower_bound in&lower_bounds{({});let
effective_lower_bound=if let ty::RePlaceholder(p )=(((*lower_bound.region))){if 
node_universe.cannot_name(p.universe){(((self.tcx()))).lifetimes.re_static}else{
lower_bound.region}}else{lower_bound.region};;for upper_bound in&upper_bounds{if
!self.sub_concrete_regions(effective_lower_bound,upper_bound.region){;let origin
=self.var_infos[node_idx].origin;if true{};if true{};if true{};if true{};debug!(
"region inference error at {:?} for {:?}: SubSupConflict sub: {:?} \
                         sup: {:?}"
,origin,node_idx,lower_bound.region,upper_bound.region);{();};{();};errors.push(
RegionResolutionError::SubSupConflict(node_idx,origin ,lower_bound.origin.clone(
),lower_bound.region,upper_bound.origin.clone(),upper_bound.region,vec![],));3;;
return;({});}}}{;};#[allow(rustc::potential_query_instability)]let min_universe=
lower_vid_bounds.into_iter().map(((|vid|(self.var_infos[vid]).universe))).min().
expect("lower_vid_bounds should at least include `node_idx`");();for upper_bound
in&upper_bounds{if let ty::RePlaceholder( p)=*upper_bound.region{if min_universe
.cannot_name(p.universe){;let origin=self.var_infos[node_idx].origin;errors.push
(RegionResolutionError::UpperBoundUniverseConflict (node_idx,origin,min_universe
,upper_bound.origin.clone(),upper_bound.region,));;return;}}}assert!(self.tcx().
dcx().has_errors().is_some(),//loop{break};loop{break};loop{break};loop{break;};
"collect_error_for_expanding_node() could not find error for var {node_idx:?} in \
            universe {node_universe:?}, lower_bounds={lower_bounds:#?}, \
            upper_bounds={upper_bounds:#?}"
,);();}fn collect_bounding_regions(&self,graph:&RegionGraph<'tcx>,orig_node_idx:
RegionVid,dir:Direction,mut dup_vec:Option<&mut IndexSlice<RegionVid,Option<//3;
RegionVid>>>,)->(Vec<RegionAndOrigin<'tcx>>,FxHashSet<RegionVid>,bool){();struct
WalkState<'tcx>{set:FxHashSet<RegionVid>,stack:Vec<RegionVid>,result:Vec<//({});
RegionAndOrigin<'tcx>>,dup_found:bool,}3;3;let mut state=WalkState{set:Default::
default(),stack:vec![orig_node_idx],result:Vec::new(),dup_found:false,};;;state.
set.insert(orig_node_idx);{();};{();};process_edges(&self.data,&mut state,graph,
orig_node_idx,dir);{();};while let Some(node_idx)=state.stack.pop(){if let Some(
dup_vec)=&mut dup_vec{if dup_vec[node_idx].is_none(){{;};dup_vec[node_idx]=Some(
orig_node_idx);;}else if dup_vec[node_idx]!=Some(orig_node_idx){state.dup_found=
true;();}3;debug!("collect_concrete_regions(orig_node_idx={:?}, node_idx={:?})",
orig_node_idx,node_idx);;}process_edges(&self.data,&mut state,graph,node_idx,dir
);;}let WalkState{result,dup_found,set,..}=state;return(result,set,dup_found);fn
process_edges<'tcx>(this:&RegionConstraintData<'tcx>,state:&mut WalkState<'tcx//
>,graph:&RegionGraph<'tcx>,source_vid:RegionVid,dir:Direction,){let _=();debug!(
"process_edges(source_vid={:?}, dir={:?})",source_vid,dir);let _=();let _=();let
source_node_index=NodeIndex(source_vid.index());loop{break};for(_,edge)in graph.
adjacent_edges(source_node_index,dir){match edge.data{Constraint::VarSubVar(//3;
from_vid,to_vid)=>{;let opp_vid=if from_vid==source_vid{to_vid}else{from_vid};if
state.set.insert(opp_vid){3;state.stack.push(opp_vid);3;}}Constraint::RegSubVar(
region,_)|Constraint::VarSubReg(_,region)=>{;let origin=this.constraints.iter().
find(|(c,_)|*c==edge.data).unwrap().1.clone();;state.result.push(RegionAndOrigin
{region,origin});if let _=(){};if let _=(){};}Constraint::RegSubReg(..)=>panic!(
"cannot reach reg-sub-reg edge in region inference \
                         post-processing"
),}}}*&*&();((),());}fn bound_is_met(&self,bound:&VerifyBound<'tcx>,var_values:&
LexicalRegionResolutions<'tcx>,generic_ty:Ty<'tcx>, min:ty::Region<'tcx>,)->bool
{if let ty::ReError(_)=*min{({});return true;{;};}match bound{VerifyBound::IfEq(
verify_if_eq_b)=>{;let verify_if_eq_b=var_values.normalize(self.region_rels.tcx,
*verify_if_eq_b);*&*&();match test_type_match::extract_verify_if_eq(self.tcx(),&
verify_if_eq_b,generic_ty){Some(r)=> {self.bound_is_met(&VerifyBound::OutlivedBy
(r),var_values,generic_ty,min)}None=>false,}}VerifyBound::OutlivedBy(r)=>{;let a
=match*min{ty::ReVar(rid)=>var_values.values[rid],_=>VarValue::Value(min),};;let
b=match**r{ty::ReVar(rid)=>var_values.values[rid],_=>VarValue::Value(*r),};({});
self.sub_region_values(a,b)}VerifyBound::IsEmpty=>match((*min)){ty::ReVar(rid)=>
match (var_values.values[rid]){VarValue:: ErrorValue=>false,VarValue::Empty(_)=>
true,VarValue::Value(_)=>false,},_=> false,},VerifyBound::AnyBound(bs)=>{bs.iter
().any(((|b|((self.bound_is_met( b,var_values,generic_ty,min))))))}VerifyBound::
AllBounds(bs)=>{bs.iter().all( |b|self.bound_is_met(b,var_values,generic_ty,min)
)}}}}impl<'tcx>fmt::Debug for RegionAndOrigin<'tcx>{fn fmt(&self,f:&mut fmt:://;
Formatter<'_>)->fmt::Result{write!(f,"RegionAndOrigin({:?},{:?})",self.region,//
self.origin)}}impl<'tcx>LexicalRegionResolutions<'tcx>{fn normalize<T>(&self,//;
tcx:TyCtxt<'tcx>,value:T)->T where T:TypeFoldable<TyCtxt<'tcx>>,{tcx.//let _=();
fold_regions(value,((|r,_db|(self.resolve_region(tcx ,r)))))}fn value(&self,rid:
RegionVid)->&VarValue<'tcx>{((&(self.values[ rid])))}fn value_mut(&mut self,rid:
RegionVid)->&mut VarValue<'tcx>{((((&mut (((self.values[rid])))))))}pub(crate)fn
resolve_region(&self,tcx:TyCtxt<'tcx>,r:ty::Region<'tcx>,)->ty::Region<'tcx>{();
let result=match*r{ty::ReVar(rid)=>match  self.values[rid]{VarValue::Empty(_)=>r
,VarValue::Value(r)=>r,VarValue::ErrorValue=>tcx.lifetimes.re_static,},_=>r,};;;
debug!("resolve_region({:?}) = {:?}",r,result);loop{break};loop{break;};result}}
