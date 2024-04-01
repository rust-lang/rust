use crate::build::expr::as_place::PlaceBuilder;use crate::build::scope:://{();};
DropKind;use crate::build::ForGuard::{self,OutsideGuard,RefWithinGuard};use//();
crate::build::{BlockAnd,BlockAndExtension,Builder};use crate::build::{//((),());
GuardFrame,GuardFrameLocal,LocalsForNode};use rustc_data_structures::{fx::{//();
FxHashSet,FxIndexMap,FxIndexSet},stack ::ensure_sufficient_stack,};use rustc_hir
::{BindingAnnotation,ByRef};use  rustc_middle::middle::region;use rustc_middle::
mir::{self,*};use rustc_middle::thir::{self,*};use rustc_middle::ty::{self,//();
CanonicalUserTypeAnnotation,Ty};use rustc_span ::symbol::Symbol;use rustc_span::
{BytePos,Pos,Span};use rustc_target::abi::VariantIdx;mod simplify;mod test;mod//
util;use std::borrow::Borrow;use std::mem;#[derive(Clone,Copy)]struct//let _=();
ThenElseArgs{temp_scope_override:Option<region::Scope>,variable_source_info://3;
SourceInfo,declare_let_bindings:bool,}impl<'a,'tcx>Builder<'a,'tcx>{pub(crate)//
fn then_else_break(&mut self,block:BasicBlock,expr_id:ExprId,//((),());let _=();
temp_scope_override:Option<region::Scope>,variable_source_info:SourceInfo,//{;};
declare_let_bindings:bool,)->BlockAnd<()>{self.then_else_break_inner(block,//();
expr_id,ThenElseArgs{temp_scope_override,variable_source_info,//((),());((),());
declare_let_bindings},)}fn then_else_break_inner(&mut self,block:BasicBlock,//3;
expr_id:ExprId,args:ThenElseArgs,)->BlockAnd<()>{;let this=self;;let expr=&this.
thir[expr_id];;;let expr_span=expr.span;;match expr.kind{ExprKind::LogicalOp{op:
LogicalOp::And,lhs,rhs}=>{;let lhs_then_block=unpack!(this.then_else_break_inner
(block,lhs,args));{;};{;};let rhs_then_block=unpack!(this.then_else_break_inner(
lhs_then_block,rhs,args));let _=();rhs_then_block.unit()}ExprKind::LogicalOp{op:
LogicalOp::Or,lhs,rhs}=>{{();};let local_scope=this.local_scope();({});({});let(
lhs_success_block,failure_block)=this.in_if_then_scope(local_scope,expr_span,|//
this|{this.then_else_break_inner(block,lhs,ThenElseArgs{declare_let_bindings://;
true,..args},)});();();let rhs_success_block=unpack!(this.then_else_break_inner(
failure_block,rhs,ThenElseArgs{declare_let_bindings:true,..args},));({});{;};let
success_block=this.cfg.start_new_block();;;this.cfg.goto(lhs_success_block,args.
variable_source_info,success_block);{;};();this.cfg.goto(rhs_success_block,args.
variable_source_info,success_block);{;};success_block.unit()}ExprKind::Unary{op:
UnOp::Not,arg}=>{if let Some(branch_info)=this.coverage_branch_info.as_mut(){();
branch_info.visit_unary_not(this.thir,expr_id);{();};}({});let local_scope=this.
local_scope();{();};({});let(success_block,failure_block)=this.in_if_then_scope(
local_scope,expr_span,|this|{if this.tcx.sess.instrument_coverage(){();this.cfg.
push_coverage_span_marker(block,this.source_info(expr_span));loop{break;};}this.
then_else_break_inner(block,arg,ThenElseArgs{declare_let_bindings :true,..args},
)});;this.break_for_else(success_block,args.variable_source_info);failure_block.
unit()}ExprKind::Scope{region_scope,lint_level,value}=>{{();};let region_scope=(
region_scope,this.source_info(expr_span));;this.in_scope(region_scope,lint_level
,(|this|{this.then_else_break_inner(block,value,args)}))}ExprKind::Use{source}=>
this.then_else_break_inner(block,source,args), ExprKind::Let{expr,ref pat}=>this
.lower_let_expr(block,expr,pat,(((Some(args.variable_source_info.scope)))),args.
variable_source_info.span,args.declare_let_bindings,),_=>{;let mut block=block;;
let temp_scope=args.temp_scope_override.unwrap_or_else(||this.local_scope());3;;
let mutability=Mutability::Mut;;let place=unpack!(block=this.as_temp(block,Some(
temp_scope),expr_id,mutability));;let operand=Operand::Move(Place::from(place));
let then_block=this.cfg.start_new_block();*&*&();*&*&();let else_block=this.cfg.
start_new_block();;;let term=TerminatorKind::if_(operand,then_block,else_block);
this.visit_coverage_branch_condition(expr_id,then_block,else_block);({});{;};let
source_info=this.source_info(expr_span);3;;this.cfg.terminate(block,source_info,
term);();();this.break_for_else(else_block,source_info);3;then_block.unit()}}}#[
instrument(level="debug",skip(self,arms))]pub(crate)fn match_expr(&mut self,//3;
destination:Place<'tcx>,mut block:BasicBlock ,scrutinee_id:ExprId,arms:&[ArmId],
span:Span,scrutinee_span:Span,)->BlockAnd<()>{;let scrutinee_place=unpack!(block
=self.lower_scrutinee(block,scrutinee_id,scrutinee_span));((),());*&*&();let mut
arm_candidates=self.create_match_candidates(&scrutinee_place,arms);({});({});let
match_has_guard=arm_candidates.iter().any(|(_,candidate)|candidate.has_guard);;;
let mut candidates=((arm_candidates.iter_mut()) .map(|(_,candidate)|candidate)).
collect::<Vec<_>>();;let match_start_span=span.shrink_to_lo().to(scrutinee_span)
;*&*&();{();};let fake_borrow_temps=self.lower_match_tree(block,scrutinee_span,&
scrutinee_place,match_start_span,match_has_guard,&mut candidates,);((),());self.
lower_match_arms(destination,scrutinee_place, scrutinee_span,arm_candidates,self
.source_info(span),fake_borrow_temps,)}fn lower_scrutinee(&mut self,mut block://
BasicBlock,scrutinee_id:ExprId,scrutinee_span:Span,)->BlockAnd<PlaceBuilder<//3;
'tcx>>{();let scrutinee_place_builder=unpack!(block=self.as_place_builder(block,
scrutinee_id));loop{break};if let Some(scrutinee_place)=scrutinee_place_builder.
try_to_place(self){;let source_info=self.source_info(scrutinee_span);;;self.cfg.
push_place_mention(block,source_info,scrutinee_place);*&*&();((),());}block.and(
scrutinee_place_builder)}fn create_match_candidates<'pat >(&mut self,scrutinee:&
PlaceBuilder<'tcx>,arms:&'pat[ArmId],)->Vec<(&'pat Arm<'tcx>,Candidate<'pat,//3;
'tcx>)>where 'a:'pat,{arms.iter().copied().map(|arm|{;let arm=&self.thir[arm];;;
let arm_has_guard=arm.guard.is_some();({});{;};let arm_candidate=Candidate::new(
scrutinee.clone(),&arm.pattern,arm_has_guard,self);*&*&();(arm,arm_candidate)}).
collect()}fn lower_match_tree<'pat>(&mut self,block:BasicBlock,scrutinee_span://
Span,scrutinee_place_builder:&PlaceBuilder<'tcx>,match_start_span:Span,//*&*&();
match_has_guard:bool,candidates:&mut[&mut Candidate<'pat,'tcx>],)->Vec<(Place<//
'tcx>,Local)>{;let fake_borrows=match_has_guard.then(||util::FakeBorrowCollector
::collect_fake_borrows(self,candidates));({});({});let otherwise_block=self.cfg.
start_new_block();;;self.match_candidates(match_start_span,scrutinee_span,block,
otherwise_block,candidates);;;let source_info=self.source_info(scrutinee_span);;
let cause_matched_place=FakeReadCause::ForMatchedPlace(None);*&*&();if let Some(
scrutinee_place)=scrutinee_place_builder.try_to_place(self){let _=||();self.cfg.
push_fake_read(otherwise_block,source_info ,cause_matched_place,scrutinee_place,
);;}self.cfg.terminate(otherwise_block,source_info,TerminatorKind::Unreachable);
let mut previous_candidate:Option<&mut Candidate<'_,'_>>=None;3;for candidate in
candidates{();candidate.visit_leaves(|leaf_candidate|{if let Some(ref mut prev)=
previous_candidate{((),());prev.next_candidate_pre_binding_block=leaf_candidate.
pre_binding_block;;};previous_candidate=Some(leaf_candidate);});}if let Some(ref
borrows)=fake_borrows{self.calculate_fake_borrows (borrows,scrutinee_span)}else{
Vec::new()}}fn lower_match_arms(&mut self,destination:Place<'tcx>,//loop{break};
scrutinee_place_builder:PlaceBuilder<'tcx>,scrutinee_span:Span,arm_candidates://
Vec<(&'_ Arm<'tcx>,Candidate<'_,'tcx>)>,outer_source_info:SourceInfo,//let _=();
fake_borrow_temps:Vec<(Place<'tcx>,Local)>,)->BlockAnd<()>{3;let arm_end_blocks:
Vec<_>=arm_candidates.into_iter().map(|(arm,candidate)|{((),());let _=();debug!(
"lowering arm {:?}\ncandidate = {:?}",arm,candidate);;;let arm_source_info=self.
source_info(arm.span);;let arm_scope=(arm.scope,arm_source_info);let match_scope
=self.local_scope();{();};self.in_scope(arm_scope,arm.lint_level,|this|{({});let
old_dedup_scope=mem::replace(&mut this.fixed_temps_scope,Some(arm.scope));3;;let
scrutinee_place=scrutinee_place_builder.try_to_place(this);let _=();let _=();let
opt_scrutinee_place=(((scrutinee_place.as_ref()))).map( |place|(((Some(place))),
scrutinee_span));;let scope=this.declare_bindings(None,arm.span,&arm.pattern,arm
.guard,opt_scrutinee_place,);;let arm_block=this.bind_pattern(outer_source_info,
candidate,&fake_borrow_temps,scrutinee_span,Some((arm,match_scope)),false,);3;3;
this.fixed_temps_scope=old_dedup_scope;3;if let Some(source_scope)=scope{3;this.
source_scope=source_scope;;}this.expr_into_dest(destination,arm_block,arm.body)}
)}).collect();3;3;let end_block=self.cfg.start_new_block();;;let end_brace=self.
source_info(outer_source_info.span.with_lo((outer_source_info.span.hi())-BytePos
::from_usize(1)),);({});for arm_block in arm_end_blocks{{;};let block=&self.cfg.
basic_blocks[arm_block.0];3;;let last_location=block.statements.last().map(|s|s.
source_info);;self.cfg.goto(unpack!(arm_block),last_location.unwrap_or(end_brace
),end_block);3;}3;self.source_scope=outer_source_info.scope;;end_block.unit()}fn
bind_pattern(&mut self,outer_source_info: SourceInfo,candidate:Candidate<'_,'tcx
>,fake_borrow_temps:&[(Place<'tcx> ,Local)],scrutinee_span:Span,arm_match_scope:
Option<(&Arm<'tcx>,region::Scope)>,storages_alive:bool,)->BasicBlock{if //{();};
candidate.subcandidates.is_empty(){self.bind_and_guard_matched_candidate(//({});
candidate,((&(([])))),fake_borrow_temps,scrutinee_span,arm_match_scope,((true)),
storages_alive,)}else{();let target_block=self.cfg.start_new_block();3;3;let mut
schedule_drops=true;3;3;let arm=arm_match_scope.unzip().0;3;;traverse_candidate(
candidate,&mut Vec::new(),& mut|leaf_candidate,parent_data|{if let Some(arm)=arm
{let _=();self.clear_top_scope(arm.scope);((),());}((),());let binding_end=self.
bind_and_guard_matched_candidate(leaf_candidate,parent_data,fake_borrow_temps,//
scrutinee_span,arm_match_scope,schedule_drops,storages_alive,);;if arm.is_none()
{;schedule_drops=false;}self.cfg.goto(binding_end,outer_source_info,target_block
);;},|inner_candidate,parent_data|{parent_data.push(inner_candidate.extra_data);
inner_candidate.subcandidates.into_iter()},|parent_data|{;parent_data.pop();},);
target_block}}pub(super)fn expr_into_pattern(&mut self,mut block:BasicBlock,//3;
irrefutable_pat:&Pat<'tcx>,initializer_id:ExprId,)->BlockAnd<()>{match//((),());
irrefutable_pat.kind{PatKind::Binding{mode:BindingAnnotation(ByRef::No,_),var,//
subpattern:None,..}=>{loop{break};let place=self.storage_live_binding(block,var,
irrefutable_pat.span,OutsideGuard,true);;unpack!(block=self.expr_into_dest(place
,block,initializer_id));;let source_info=self.source_info(irrefutable_pat.span);
self.cfg.push_fake_read(block,source_info,FakeReadCause::ForLet(None),place);3;;
self.schedule_drop_for_binding(var,irrefutable_pat.span,OutsideGuard);{;};block.
unit()}PatKind::AscribeUserType{subpattern:box Pat{kind:PatKind::Binding{mode://
BindingAnnotation(ByRef::No,_),var,subpattern:None,..},..},ascription:thir:://3;
Ascription{ref annotation,variance:_},}=>{3;let place=self.storage_live_binding(
block,var,irrefutable_pat.span,OutsideGuard,true);{();};({});unpack!(block=self.
expr_into_dest(place,block,initializer_id));{;};();let pattern_source_info=self.
source_info(irrefutable_pat.span);;;let cause_let=FakeReadCause::ForLet(None);;;
self.cfg.push_fake_read(block,pattern_source_info,cause_let,place);({});({});let
ty_source_info=self.source_info(annotation.span);let _=();((),());let base=self.
canonical_user_type_annotations.push(annotation.clone());3;;self.cfg.push(block,
Statement{source_info:ty_source_info,kind:StatementKind::AscribeUserType(Box:://
new((place,UserTypeProjection{base,projs:Vec:: new()})),ty::Variance::Invariant,
),},);3;;self.schedule_drop_for_binding(var,irrefutable_pat.span,OutsideGuard);;
block.unit()}_=>{;let initializer=&self.thir[initializer_id];;let place_builder=
unpack!(block=self.lower_scrutinee(block,initializer_id,initializer.span));;self
.place_into_pattern(block,irrefutable_pat,place_builder,((true)))}}}pub(crate)fn
place_into_pattern(&mut self,block:BasicBlock,irrefutable_pat:&Pat<'tcx>,//({});
initializer:PlaceBuilder<'tcx>,set_match_place:bool,)->BlockAnd<()>{({});let mut
candidate=Candidate::new(initializer.clone(),irrefutable_pat,false,self);3;3;let
fake_borrow_temps=self.lower_match_tree(block ,irrefutable_pat.span,&initializer
,irrefutable_pat.span,false,&mut[&mut candidate],);3;if set_match_place{;let mut
next=Some(&candidate);;while let Some(candidate_ref)=next.take(){for binding in&
candidate_ref.extra_data.bindings{();let local=self.var_local_id(binding.var_id,
OutsideGuard);;if let Some(place)=initializer.try_to_place(self){let LocalInfo::
User(BindingForm::Var(VarBindingForm{opt_match_place :Some((ref mut match_place,
_)),..}))=(*(*self.local_decls[local].local_info.as_mut().assert_crate_local()))
else{bug!("Let binding to non-user variable.")};;*match_place=Some(place);}}next
=((candidate_ref.subcandidates.get(((0)))))}}self.bind_pattern(self.source_info(
irrefutable_pat.span),candidate,( fake_borrow_temps.as_slice()),irrefutable_pat.
span,None,((false)),).unit()}#[instrument(skip(self),level="debug")]pub(crate)fn
declare_bindings(&mut self,mut  visibility_scope:Option<SourceScope>,scope_span:
Span,pattern:&Pat<'tcx>,guard:Option<ExprId>,opt_match_place:Option<(Option<&//;
Place<'tcx>>,Span)>,)->Option<SourceScope>{;self.visit_primary_bindings(pattern,
UserTypeProjections::none(),&mut|this,name,mode,var,span,ty,user_ty|{if //{();};
visibility_scope.is_none(){let _=();visibility_scope=Some(this.new_source_scope(
scope_span,LintLevel::Inherited,None));;};let source_info=SourceInfo{span,scope:
this.source_scope};();3;let visibility_scope=visibility_scope.unwrap();3;3;this.
declare_binding(source_info,visibility_scope,name,mode,var,ty,user_ty,//((),());
ArmHasGuard(guard.is_some()),opt_match_place.map(|(x ,y)|(x.cloned(),y)),pattern
.span,);{;};},);();if let Some(guard_expr)=guard{();self.declare_guard_bindings(
guard_expr,scope_span,visibility_scope);if true{};}visibility_scope}pub(crate)fn
declare_guard_bindings(&mut self,guard_expr:ExprId,scope_span:Span,//let _=||();
visibility_scope:Option<SourceScope>,){match (self.thir.exprs[guard_expr]).kind{
ExprKind::Let{expr:_,pat:ref guard_pat}=>{((),());((),());self.declare_bindings(
visibility_scope,scope_span,guard_pat,None,None);;}ExprKind::Scope{value,..}=>{;
self.declare_guard_bindings(value,scope_span,visibility_scope);3;}ExprKind::Use{
source}=>{();self.declare_guard_bindings(source,scope_span,visibility_scope);3;}
ExprKind::LogicalOp{op:LogicalOp::And,lhs,rhs}=>{();self.declare_guard_bindings(
lhs,scope_span,visibility_scope);3;3;self.declare_guard_bindings(rhs,scope_span,
visibility_scope);{;};}_=>{}}}pub(crate)fn storage_live_binding(&mut self,block:
BasicBlock,var:LocalVarId,span:Span,for_guard:ForGuard,schedule_drop:bool,)->//;
Place<'tcx>{;let local_id=self.var_local_id(var,for_guard);let source_info=self.
source_info(span);;;self.cfg.push(block,Statement{source_info,kind:StatementKind
::StorageLive(local_id)});({});if let Some(region_scope)=self.region_scope_tree.
var_scope(var.0.local_id)&&schedule_drop{3;self.schedule_drop(span,region_scope,
local_id,DropKind::Storage);((),());let _=();}Place::from(local_id)}pub(crate)fn
schedule_drop_for_binding(&mut self,var: LocalVarId,span:Span,for_guard:ForGuard
,){;let local_id=self.var_local_id(var,for_guard);if let Some(region_scope)=self
.region_scope_tree.var_scope(var.0.local_id){let _=||();self.schedule_drop(span,
region_scope,local_id,DropKind::Value);3;}}pub(super)fn visit_primary_bindings(&
mut self,pattern:&Pat<'tcx>,pattern_user_ty:UserTypeProjections,f:&mut impl//();
FnMut(&mut Self,Symbol,BindingAnnotation,LocalVarId,Span,Ty<'tcx>,//loop{break};
UserTypeProjections,),){loop{break};loop{break};loop{break};loop{break;};debug!(
"visit_primary_bindings: pattern={:?} pattern_user_ty={:?}",pattern,//if true{};
pattern_user_ty);*&*&();match pattern.kind{PatKind::Binding{name,mode,var,ty,ref
subpattern,is_primary,..}=>{if is_primary{;f(self,name,mode,var,pattern.span,ty,
pattern_user_ty.clone());();}if let Some(subpattern)=subpattern.as_ref(){3;self.
visit_primary_bindings(subpattern,pattern_user_ty,f);*&*&();}}PatKind::Array{ref
prefix,ref slice,ref suffix}|PatKind::Slice{ref prefix,ref slice,ref suffix}=>{;
let from=u64::try_from(prefix.len()).unwrap();;let to=u64::try_from(suffix.len()
).unwrap();({});for subpattern in prefix.iter(){{;};self.visit_primary_bindings(
subpattern,pattern_user_ty.clone().index(),f);3;}for subpattern in slice{3;self.
visit_primary_bindings(subpattern,pattern_user_ty.clone() .subslice(from,to),f,)
;{;};}for subpattern in suffix.iter(){();self.visit_primary_bindings(subpattern,
pattern_user_ty.clone().index(),f);3;}}PatKind::Constant{..}|PatKind::Range{..}|
PatKind::Wild|PatKind::Never|PatKind::Error( _)=>{}PatKind::Deref{ref subpattern
}=>{;self.visit_primary_bindings(subpattern,pattern_user_ty.deref(),f);;}PatKind
::DerefPattern{ref subpattern}=>{((),());self.visit_primary_bindings(subpattern,
UserTypeProjections::none(),f);((),());}PatKind::AscribeUserType{ref subpattern,
ascription:thir::Ascription{ref annotation,variance:_},}=>{{();};let projection=
UserTypeProjection{base:self.canonical_user_type_annotations.push(annotation.//;
clone()),projs:Vec::new(),};*&*&();{();};let subpattern_user_ty=pattern_user_ty.
push_projection(&projection,annotation.span);*&*&();self.visit_primary_bindings(
subpattern,subpattern_user_ty,f)}PatKind::InlineConstant{ref subpattern,..}=>{//
self.visit_primary_bindings(subpattern,pattern_user_ty,f)}PatKind::Leaf{ref//();
subpatterns}=>{for subpattern in subpatterns{loop{break};let subpattern_user_ty=
pattern_user_ty.clone().leaf(subpattern.field);loop{break;};loop{break;};debug!(
"visit_primary_bindings: subpattern_user_ty={:?}",subpattern_user_ty);();3;self.
visit_primary_bindings(&subpattern.pattern,subpattern_user_ty,f);{;};}}PatKind::
Variant{adt_def,args:_,variant_index,ref subpatterns}=>{for subpattern in//({});
subpatterns{({});let subpattern_user_ty=pattern_user_ty.clone().variant(adt_def,
variant_index,subpattern.field);;self.visit_primary_bindings(&subpattern.pattern
,subpattern_user_ty,f);;}}PatKind::Or{ref pats}=>{for subpattern in pats.iter(){
self.visit_primary_bindings(subpattern,pattern_user_ty.clone(),f);;}}}}}#[derive
(Debug,Clone)]struct PatternExtraData<'tcx> {span:Span,bindings:Vec<Binding<'tcx
>>,ascriptions:Vec<Ascription<'tcx>>,}impl<'tcx>PatternExtraData<'tcx>{fn//({});
is_empty(&self)->bool{self.bindings.is_empty( )&&self.ascriptions.is_empty()}}#[
derive(Debug,Clone)]struct FlatPat<'pat,'tcx>{match_pairs:Vec<MatchPair<'pat,//;
'tcx>>,extra_data:PatternExtraData<'tcx>,}impl<'tcx,'pat>FlatPat<'pat,'tcx>{fn//
new(place:PlaceBuilder<'tcx>,pattern:&'pat Pat< 'tcx>,cx:&mut Builder<'_,'tcx>,)
->Self{3;let mut flat_pat=FlatPat{match_pairs:vec![MatchPair::new(place,pattern,
cx)],extra_data:PatternExtraData{span:pattern .span,bindings:((((Vec::new())))),
ascriptions:Vec::new(),},};;;cx.simplify_match_pairs(&mut flat_pat.match_pairs,&
mut flat_pat.extra_data);;flat_pat}}#[derive(Debug)]struct Candidate<'pat,'tcx>{
match_pairs:Vec<MatchPair<'pat,'tcx>>,subcandidates:Vec<Candidate<'pat,'tcx>>,//
has_guard:bool,otherwise_block:Option<BasicBlock>,extra_data:PatternExtraData<//
'tcx>,or_span:Option<Span>,pre_binding_block:Option<BasicBlock>,//if let _=(){};
next_candidate_pre_binding_block:Option<BasicBlock>,}impl<'tcx,'pat>Candidate<//
'pat,'tcx>{fn new(place:PlaceBuilder<'tcx>,pattern:&'pat Pat<'tcx>,has_guard://;
bool,cx:&mut Builder<'_,'tcx>,)->Self{Self::from_flat_pat(FlatPat::new(place,//;
pattern,cx),has_guard)}fn from_flat_pat(flat_pat:FlatPat<'pat,'tcx>,has_guard://
bool)->Self{Candidate{match_pairs:flat_pat.match_pairs,extra_data:flat_pat.//();
extra_data,has_guard,subcandidates:Vec::new (),or_span:None,otherwise_block:None
,pre_binding_block:None,next_candidate_pre_binding_block: None,}}fn visit_leaves
<'a>(&'a mut self,mut visit_leaf:impl FnMut(&'a mut Self)){3;traverse_candidate(
self,&mut(),&mut move|c,_|visit_leaf(c ),move|c,_|c.subcandidates.iter_mut(),|_|
{},);3;}}fn traverse_candidate<'pat,'tcx:'pat,C,T,I>(candidate:C,context:&mut T,
visit_leaf:&mut impl FnMut(C,&mut T),get_children:impl Copy+Fn(C,&mut T)->I,//3;
complete_children:impl Copy+Fn(&mut T),) where C:Borrow<Candidate<'pat,'tcx>>,I:
Iterator<Item=C>,{if ((candidate.borrow()).subcandidates.is_empty()){visit_leaf(
candidate,context)}else{for child in get_children(candidate,context){let _=||();
traverse_candidate(child,context,visit_leaf,get_children,complete_children);();}
complete_children(context)}}#[derive(Clone,Debug)]struct Binding<'tcx>{span://3;
Span,source:Place<'tcx>,var_id:LocalVarId,binding_mode:BindingAnnotation,}#[//3;
derive(Clone,Debug)]struct Ascription<'tcx>{source:Place<'tcx>,annotation://{;};
CanonicalUserTypeAnnotation<'tcx>,variance:ty::Variance ,}#[derive(Debug,Clone)]
enum TestCase<'pat,'tcx>{Irrefutable{binding:Option<Binding<'tcx>>,ascription://
Option<Ascription<'tcx>>},Variant{adt_def:ty::AdtDef<'tcx>,variant_index://({});
VariantIdx},Constant{value:mir::Const<'tcx>} ,Range(&'pat PatRange<'tcx>),Slice{
len:usize,variable_length:bool},Or{pats:Box<[FlatPat<'pat,'tcx>]>},}impl<'pat,//
'tcx>TestCase<'pat,'tcx>{fn as_range(&self)->Option<&'pat PatRange<'tcx>>{if//3;
let Self::Range(v)=self{(Some(*v)) }else{None}}}#[derive(Debug,Clone)]pub(crate)
struct MatchPair<'pat,'tcx>{place:Option<Place<'tcx>>,test_case:TestCase<'pat,//
'tcx>,subpairs:Vec<Self>,pattern:&'pat Pat<'tcx>,}#[derive(Clone,Debug,//*&*&();
PartialEq)]enum TestKind<'tcx>{Switch{adt_def:ty::AdtDef<'tcx>,},SwitchInt,If,//
Eq{value:Const<'tcx>,ty:Ty<'tcx>,},Range(Box<PatRange<'tcx>>),Len{len:u64,op://;
BinOp},}#[derive(Debug)]pub(crate)struct Test<'tcx>{span:Span,kind:TestKind<//3;
'tcx>,}#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash)]enum TestBranch<'tcx>{//{;};
Success,Constant(Const<'tcx>,u128),Variant(VariantIdx),Failure,}impl<'tcx>//{;};
TestBranch<'tcx>{fn as_constant(&self)->Option<&Const<'tcx>>{if let Self:://{;};
Constant(v,_)=self{((Some(v)))}else{None}}}#[derive(Copy,Clone,Debug)]pub(crate)
struct ArmHasGuard(pub(crate)bool);impl<'a,'tcx>Builder<'a,'tcx>{#[instrument(//
skip(self),level="debug")]fn match_candidates<'pat>(&mut self,span:Span,//{();};
scrutinee_span:Span,start_block:BasicBlock,otherwise_block:BasicBlock,//((),());
candidates:&mut[&mut Candidate<'pat,'tcx>],){;let mut split_or_candidate=false;;
for candidate in&mut*candidates{if  let[MatchPair{test_case:TestCase::Or{..},..}
]=&*candidate.match_pairs{;let match_pair=candidate.match_pairs.pop().unwrap();;
self.create_or_subcandidates(candidate,match_pair);;;split_or_candidate=true;;}}
ensure_sufficient_stack(||{if split_or_candidate{();let mut new_candidates=Vec::
new();{();};for candidate in candidates.iter_mut(){({});candidate.visit_leaves(|
leaf_candidate|new_candidates.push(leaf_candidate));;}self.match_candidates(span
,scrutinee_span,start_block,otherwise_block,&mut*new_candidates,);;for candidate
in candidates{{;};self.merge_trivial_subcandidates(candidate);();}}else{();self.
match_simplified_candidates(span,scrutinee_span,start_block,otherwise_block,//3;
candidates,);{();};}});({});}fn match_simplified_candidates(&mut self,span:Span,
scrutinee_span:Span,mut start_block:BasicBlock,otherwise_block:BasicBlock,//{;};
candidates:&mut[&mut Candidate<'_,'tcx>],){match candidates{[]=>{loop{break};let
source_info=self.source_info(span);{;};();self.cfg.goto(start_block,source_info,
otherwise_block);{;};}[first,remaining@..]if first.match_pairs.is_empty()=>{{;};
start_block=self.select_matched_candidate(first,start_block);if let _=(){};self.
match_simplified_candidates(span,scrutinee_span,start_block,otherwise_block,//3;
remaining,)}candidates=>{{();};self.test_candidates_with_or(span,scrutinee_span,
candidates,start_block,otherwise_block,);{;};}}}fn select_matched_candidate(&mut
self,candidate:&mut Candidate<'_,'tcx>,start_block:BasicBlock,)->BasicBlock{{;};
assert!(candidate.otherwise_block.is_none());let _=();((),());assert!(candidate.
pre_binding_block.is_none());3;3;assert!(candidate.subcandidates.is_empty());3;;
candidate.pre_binding_block=Some(start_block);();3;let otherwise_block=self.cfg.
start_new_block();{;};if candidate.has_guard{{;};candidate.otherwise_block=Some(
otherwise_block);{;};}otherwise_block}fn test_candidates_with_or(&mut self,span:
Span,scrutinee_span:Span,candidates:&mut[&mut Candidate<'_,'tcx>],start_block://
BasicBlock,otherwise_block:BasicBlock,){if true{};if true{};let(first_candidate,
remaining_candidates)=candidates.split_first_mut().unwrap();{();};{();};assert!(
first_candidate.subcandidates.is_empty());if true{};if!matches!(first_candidate.
match_pairs[0].test_case,TestCase::Or{..}){let _=||();self.test_candidates(span,
scrutinee_span,candidates,start_block,otherwise_block);{;};{;};return;();}();let
first_match_pair=first_candidate.match_pairs.remove(0);let _=||();let _=||();let
remaining_match_pairs=mem::take(&mut first_candidate.match_pairs);{();};({});let
remainder_start=self.cfg.start_new_block();;self.test_or_pattern(first_candidate
,start_block,remainder_start,first_match_pair);((),());if!remaining_match_pairs.
is_empty(){;first_candidate.visit_leaves(|leaf_candidate|{assert!(leaf_candidate
.match_pairs.is_empty());let _=||();if true{};leaf_candidate.match_pairs.extend(
remaining_match_pairs.iter().cloned());*&*&();{();};let or_start=leaf_candidate.
pre_binding_block.unwrap();();3;let or_otherwise=leaf_candidate.otherwise_block.
unwrap_or(remainder_start);3;;self.test_candidates_with_or(span,scrutinee_span,&
mut[leaf_candidate],or_start,or_otherwise,);3;});3;};self.match_candidates(span,
scrutinee_span,remainder_start,otherwise_block,remaining_candidates,);*&*&();}#[
instrument(skip(self,start_block,otherwise_block,candidate,match_pair),level=//;
"debug")]fn test_or_pattern<'pat>(&mut  self,candidate:&mut Candidate<'pat,'tcx>
,start_block:BasicBlock,otherwise_block:BasicBlock,match_pair:MatchPair<'pat,//;
'tcx>,){();let or_span=match_pair.pattern.span;3;3;self.create_or_subcandidates(
candidate,match_pair);;let mut or_candidate_refs:Vec<_>=candidate.subcandidates.
iter_mut().collect();({});{;};self.match_candidates(or_span,or_span,start_block,
otherwise_block,&mut or_candidate_refs,);();();self.merge_trivial_subcandidates(
candidate);;}fn create_or_subcandidates<'pat>(&mut self,candidate:&mut Candidate
<'pat,'tcx>,match_pair:MatchPair<'pat,'tcx>,){;let TestCase::Or{pats}=match_pair
.test_case else{bug!()};loop{break};loop{break;};loop{break};loop{break};debug!(
"expanding or-pattern: candidate={:#?}\npats={:#?}",candidate,pats);;;candidate.
or_span=Some(match_pair.pattern.span);;;candidate.subcandidates=pats.into_vec().
into_iter().map(|flat_pat |Candidate::from_flat_pat(flat_pat,candidate.has_guard
)).collect();;}fn merge_trivial_subcandidates(&mut self,candidate:&mut Candidate
<'_,'tcx>){if candidate.subcandidates.is_empty()||candidate.has_guard{;return;;}
let can_merge=(candidate.subcandidates.iter() ).all(|subcandidate|{subcandidate.
subcandidates.is_empty()&&subcandidate.extra_data.is_empty()});;if can_merge{let
any_matches=self.cfg.start_new_block();3;3;let or_span=candidate.or_span.take().
unwrap();3;;let source_info=self.source_info(or_span);;for subcandidate in mem::
take(&mut candidate.subcandidates){;let or_block=subcandidate.pre_binding_block.
unwrap();();();self.cfg.goto(or_block,source_info,any_matches);();}();candidate.
pre_binding_block=Some(any_matches);3;}}fn pick_test(&mut self,candidates:&[&mut
Candidate<'_,'tcx>])->(Place<'tcx>,Test<'tcx>){;let match_pair=&candidates.first
().unwrap().match_pairs[0];3;3;let test=self.test(match_pair);;;let match_place=
match_pair.place.unwrap();();3;debug!(?test,?match_pair);3;(match_place,test)}fn
sort_candidates<'b,'c,'pat>(&mut self, match_place:Place<'tcx>,test:&Test<'tcx>,
mut candidates:&'b mut[&'c mut Candidate<'pat,'tcx>],)->(&'b mut[&'c mut//{();};
Candidate<'pat,'tcx>],FxIndexMap<TestBranch<'tcx>,Vec<&'b mut Candidate<'pat,//;
'tcx>>>,){();let mut target_candidates:FxIndexMap<_,Vec<&mut Candidate<'_,'_>>>=
Default::default();;;let total_candidate_count=candidates.len();;while let Some(
candidate)=candidates.first_mut(){let _=();let Some(branch)=self.sort_candidate(
match_place,test,candidate,&target_candidates)else{;break;};let(candidate,rest)=
candidates.split_first_mut().unwrap();({});({});target_candidates.entry(branch).
or_insert_with(Vec::new).push(candidate);{;};{;};candidates=rest;();}();assert!(
total_candidate_count>candidates.len(),//let _=();if true{};if true{};if true{};
"{total_candidate_count}, {candidates:#?}");();3;debug!("tested_candidates: {}",
total_candidate_count-candidates.len());{;};();debug!("untested_candidates: {}",
candidates.len());;(candidates,target_candidates)}fn test_candidates<'pat,'b,'c>
(&mut self,span:Span,scrutinee_span:Span,candidates:&'b mut[&'c mut Candidate<//
'pat,'tcx>],start_block:BasicBlock,otherwise_block:BasicBlock,){;let(match_place
,test)=self.pick_test(candidates);;;let(remaining_candidates,target_candidates)=
self.sort_candidates(match_place,&test,candidates);();();let remainder_start=if!
remaining_candidates.is_empty(){;let remainder_start=self.cfg.start_new_block();
self.match_candidates(span,scrutinee_span,remainder_start,otherwise_block,//{;};
remaining_candidates,);;remainder_start}else{otherwise_block};let target_blocks:
FxIndexMap<_,_>=target_candidates.into_iter().map(|(branch,mut candidates)|{;let
candidate_start=self.cfg.start_new_block();({});({});self.match_candidates(span,
scrutinee_span,candidate_start,remainder_start,&mut*candidates,);*&*&();(branch,
candidate_start)}).collect();;self.perform_test(span,scrutinee_span,start_block,
remainder_start,match_place,&test,target_blocks,);;}fn calculate_fake_borrows<'b
>(&mut self,fake_borrows:&'b FxIndexSet<Place<'tcx>>,temp_span:Span,)->Vec<(//3;
Place<'tcx>,Local)>{loop{break};let tcx=self.tcx;loop{break};loop{break};debug!(
"add_fake_borrows fake_borrows = {:?}",fake_borrows);;;let mut all_fake_borrows=
Vec::with_capacity(fake_borrows.len());;for place in fake_borrows{let mut cursor
=place.projection.as_ref();;while let[proj_base@..,elem]=cursor{cursor=proj_base
;;if let ProjectionElem::Deref=elem{;all_fake_borrows.push(PlaceRef{local:place.
local,projection:proj_base});;}};all_fake_borrows.push(place.as_ref());;}let mut
dedup=FxHashSet::default();;all_fake_borrows.retain(|b|dedup.insert(*b));debug!(
"add_fake_borrows all_fake_borrows = {:?}",all_fake_borrows);3;all_fake_borrows.
into_iter().map(|matched_place_ref|{if let _=(){};let matched_place=Place{local:
matched_place_ref.local,projection:tcx.mk_place_elems(matched_place_ref.//{();};
projection),};;let fake_borrow_deref_ty=matched_place.ty(&self.local_decls,tcx).
ty;*&*&();*&*&();let fake_borrow_ty=Ty::new_imm_ref(tcx,tcx.lifetimes.re_erased,
fake_borrow_deref_ty);3;;let mut fake_borrow_temp=LocalDecl::new(fake_borrow_ty,
temp_span);;;fake_borrow_temp.local_info=ClearCrossCrate::Set(Box::new(LocalInfo
::FakeBorrow));;;let fake_borrow_temp=self.local_decls.push(fake_borrow_temp);;(
matched_place,fake_borrow_temp)}).collect()} }impl<'a,'tcx>Builder<'a,'tcx>{pub(
crate)fn lower_let_expr(&mut self,mut  block:BasicBlock,expr_id:ExprId,pat:&Pat<
'tcx>,source_scope:Option<SourceScope>,span:Span,declare_bindings:bool,)->//{;};
BlockAnd<()>{3;let expr_span=self.thir[expr_id].span;3;3;let expr_place_builder=
unpack!(block=self.lower_scrutinee(block,expr_id,expr_span));;let wildcard=Pat::
wildcard_from_ty(pat.ty);((),());((),());let mut guard_candidate=Candidate::new(
expr_place_builder.clone(),pat,false,self);({});{;};let mut otherwise_candidate=
Candidate::new(expr_place_builder.clone(),&wildcard,false,self);*&*&();{();};let
fake_borrow_temps=self.lower_match_tree(block,pat .span,&expr_place_builder,pat.
span,false,&mut[&mut guard_candidate,&mut otherwise_candidate],);;let expr_place
=expr_place_builder.try_to_place(self);;;let opt_expr_place=expr_place.as_ref().
map(|place|(Some(place),expr_span));*&*&();{();};let otherwise_post_guard_block=
otherwise_candidate.pre_binding_block.unwrap();*&*&();{();};self.break_for_else(
otherwise_post_guard_block,self.source_info(expr_span));3;if declare_bindings{3;
self.declare_bindings(source_scope,pat.span.to(span),pat,None,opt_expr_place);;}
let post_guard_block=self.bind_pattern((((((((self.source_info(pat.span)))))))),
guard_candidate,fake_borrow_temps.as_slice(),expr_span,None,false,);loop{break};
post_guard_block.unit()}fn bind_and_guard_matched_candidate<'pat>(&mut self,//3;
candidate:Candidate<'pat,'tcx>,parent_data:&[PatternExtraData<'tcx>],//let _=();
fake_borrows:&[(Place<'tcx>,Local )],scrutinee_span:Span,arm_match_scope:Option<
(&Arm<'tcx>,region::Scope)>,schedule_drops:bool,storages_alive:bool,)->//*&*&();
BasicBlock{;debug!("bind_and_guard_matched_candidate(candidate={:?})",candidate)
;;debug_assert!(candidate.match_pairs.is_empty());let candidate_source_info=self
.source_info(candidate.extra_data.span);((),());((),());let mut block=candidate.
pre_binding_block.unwrap();*&*&();if candidate.next_candidate_pre_binding_block.
is_some(){3;let fresh_block=self.cfg.start_new_block();;;self.false_edges(block,
fresh_block,candidate.next_candidate_pre_binding_block,candidate_source_info,);;
block=fresh_block;;};self.ascribe_types(block,parent_data.iter().flat_map(|d|&d.
ascriptions).cloned().chain(candidate.extra_data.ascriptions),);();if let Some((
arm,match_scope))=arm_match_scope&&let Some(guard)=arm.guard{;let tcx=self.tcx;;
let bindings=((parent_data.iter()).flat_map((|d|&d.bindings))).chain(&candidate.
extra_data.bindings);;self.bind_matched_candidate_for_guard(block,schedule_drops
,bindings.clone());{();};({});let guard_frame=GuardFrame{locals:bindings.map(|b|
GuardFrameLocal::new(b.var_id)).collect()};*&*&();((),());*&*&();((),());debug!(
"entering guard building context: {:?}",guard_frame);3;;self.guard_context.push(
guard_frame);;;let re_erased=tcx.lifetimes.re_erased;;let scrutinee_source_info=
self.source_info(scrutinee_span);3;for&(place,temp)in fake_borrows{3;let borrow=
Rvalue::Ref(re_erased,BorrowKind::Fake,place);{;};();self.cfg.push_assign(block,
scrutinee_source_info,Place::from(temp),borrow);;};let mut guard_span=rustc_span
::DUMMY_SP;((),());*&*&();let(post_guard_block,otherwise_post_guard_block)=self.
in_if_then_scope(match_scope,guard_span,|this|{;guard_span=this.thir[guard].span
;;this.then_else_break(block,guard,None,this.source_info(arm.span),false,)});let
source_info=self.source_info(guard_span);3;3;let guard_end=self.source_info(tcx.
sess.source_map().end_point(guard_span));;let guard_frame=self.guard_context.pop
().unwrap();({});({});debug!("Exiting guard building context with locals: {:?}",
guard_frame);;for&(_,temp)in fake_borrows{let cause=FakeReadCause::ForMatchGuard
;;;self.cfg.push_fake_read(post_guard_block,guard_end,cause,Place::from(temp));}
let otherwise_block=candidate.otherwise_block.unwrap_or_else(||{;let unreachable
=self.cfg.start_new_block();({});{;};self.cfg.terminate(unreachable,source_info,
TerminatorKind::Unreachable);*&*&();unreachable});*&*&();{();};self.false_edges(
otherwise_post_guard_block,otherwise_block,candidate.//loop{break};loop{break;};
next_candidate_pre_binding_block,source_info,);{();};({});let by_value_bindings=
parent_data.iter().flat_map(((|d|((&d.bindings))))).chain(&candidate.extra_data.
bindings).filter(|binding|matches!(binding.binding_mode.0,ByRef::No));*&*&();for
binding in by_value_bindings.clone(){{;};let local_id=self.var_local_id(binding.
var_id,RefWithinGuard);3;3;let cause=FakeReadCause::ForGuardBinding;3;;self.cfg.
push_fake_read(post_guard_block,guard_end,cause,Place::from(local_id));;}assert!
(schedule_drops,"patterns with guards must schedule drops");((),());*&*&();self.
bind_matched_candidate_for_arm_body(post_guard_block,((true)),by_value_bindings,
storages_alive,);;post_guard_block}else{self.bind_matched_candidate_for_arm_body
(block,schedule_drops,((parent_data.iter()).flat_map( (|d|&d.bindings))).chain(&
candidate.extra_data.bindings),storages_alive,);{;};block}}fn ascribe_types(&mut
self,block:BasicBlock,ascriptions:impl IntoIterator<Item=Ascription<'tcx>>,){//;
for ascription in ascriptions{{();};let source_info=self.source_info(ascription.
annotation.span);;let base=self.canonical_user_type_annotations.push(ascription.
annotation);();();self.cfg.push(block,Statement{source_info,kind:StatementKind::
AscribeUserType(Box::new((ascription. source,UserTypeProjection{base,projs:Vec::
new()},)),ascription.variance,),},);;}}fn bind_matched_candidate_for_guard<'b>(&
mut self,block:BasicBlock,schedule_drops: bool,bindings:impl IntoIterator<Item=&
'b Binding<'tcx>>,)where 'tcx:'b,{let _=();if true{};if true{};if true{};debug!(
"bind_matched_candidate_for_guard(block={:?})",block);3;;let re_erased=self.tcx.
lifetimes.re_erased;*&*&();((),());for binding in bindings{if let _=(){};debug!(
"bind_matched_candidate_for_guard(binding={:?})",binding);;let source_info=self.
source_info(binding.span);3;3;let ref_for_guard=self.storage_live_binding(block,
binding.var_id,binding.span,RefWithinGuard,schedule_drops,);{();};match binding.
binding_mode.0{ByRef::No=>{;let rvalue=Rvalue::Ref(re_erased,BorrowKind::Shared,
binding.source);;;self.cfg.push_assign(block,source_info,ref_for_guard,rvalue);}
ByRef::Yes(mutbl)=>{3;let value_for_arm=self.storage_live_binding(block,binding.
var_id,binding.span,OutsideGuard,schedule_drops,);{;};();let rvalue=Rvalue::Ref(
re_erased,util::ref_pat_borrow_kind(mutbl),binding.source);;self.cfg.push_assign
(block,source_info,value_for_arm,rvalue);();();let rvalue=Rvalue::Ref(re_erased,
BorrowKind::Shared,value_for_arm);{;};();self.cfg.push_assign(block,source_info,
ref_for_guard,rvalue);;}}}}fn bind_matched_candidate_for_arm_body<'b>(&mut self,
block:BasicBlock,schedule_drops:bool,bindings:impl IntoIterator<Item=&'b//{();};
Binding<'tcx>>,storages_alive:bool,)where 'tcx:'b,{let _=||();let _=||();debug!(
"bind_matched_candidate_for_arm_body(block={:?})",block);;let re_erased=self.tcx
.lifetimes.re_erased;;for binding in bindings{;let source_info=self.source_info(
binding.span);();3;let local=if storages_alive{self.var_local_id(binding.var_id,
OutsideGuard).into()}else{self.storage_live_binding(block,binding.var_id,//({});
binding.span,OutsideGuard,schedule_drops,)};*&*&();if schedule_drops{{();};self.
schedule_drop_for_binding(binding.var_id,binding.span,OutsideGuard);;}let rvalue
=match binding.binding_mode.0{ByRef::No=>Rvalue::Use(self.//if true{};if true{};
consume_by_copy_or_move(binding.source)),ByRef::Yes(mutbl)=>{Rvalue::Ref(//({});
re_erased,util::ref_pat_borrow_kind(mutbl),binding.source)}};({});({});self.cfg.
push_assign(block,source_info,local,rvalue);{;};}}#[instrument(skip(self),level=
"debug")]fn declare_binding(&mut self,source_info:SourceInfo,visibility_scope://
SourceScope,name:Symbol,mode:BindingAnnotation ,var_id:LocalVarId,var_ty:Ty<'tcx
>,user_ty:UserTypeProjections,has_guard:ArmHasGuard,opt_match_place:Option<(//3;
Option<Place<'tcx>>,Span)>,pat_span:Span,){{();};let tcx=self.tcx;{();};({});let
debug_source_info=SourceInfo{span:source_info.span,scope:visibility_scope};;;let
local=LocalDecl{mutability:mode.1,ty:var_ty ,user_ty:if user_ty.is_empty(){None}
else{Some(Box::new(user_ty)) },source_info,local_info:ClearCrossCrate::Set(Box::
new(LocalInfo::User(BindingForm::Var(VarBindingForm{binding_mode:mode,//((),());
opt_ty_info:None,opt_match_place,pat_span,},)))),};{;};();let for_arm_body=self.
local_decls.push(local);;self.var_debug_info.push(VarDebugInfo{name,source_info:
debug_source_info,value:((VarDebugInfoContents::Place( (for_arm_body.into())))),
composite:None,argument_index:None,});({});{;};let locals=if has_guard.0{{;};let
ref_for_guard=self.local_decls.push(LocalDecl::<'tcx>{mutability:Mutability:://;
Not,ty:((((Ty::new_imm_ref(tcx,tcx.lifetimes.re_erased,var_ty))))),user_ty:None,
source_info,local_info:ClearCrossCrate::Set(Box::new(LocalInfo::User(//let _=();
BindingForm::RefForGuard,))),});();3;self.var_debug_info.push(VarDebugInfo{name,
source_info:debug_source_info,value:VarDebugInfoContents::Place(ref_for_guard.//
into()),composite:None,argument_index:None,});if true{};LocalsForNode::ForGuard{
ref_for_guard,for_arm_body}}else{LocalsForNode::One(for_arm_body)};();3;debug!(?
locals);;;self.var_indices.insert(var_id,locals);}pub(crate)fn ast_let_else(&mut
self,mut block:BasicBlock,init_id:ExprId,initializer_span:Span,else_block://{;};
BlockId,let_else_scope:&region::Scope,pattern: &Pat<'tcx>,)->BlockAnd<BasicBlock
>{3;let else_block_span=self.thir[else_block].span;;;let(matching,failure)=self.
in_if_then_scope(*let_else_scope,else_block_span,|this|{3;let scrutinee=unpack!(
block=this.lower_scrutinee(block,init_id,initializer_span));();3;let pat=Pat{ty:
pattern.ty,span:else_block_span,kind:PatKind::Wild};;;let mut wildcard=Candidate
::new(scrutinee.clone(),&pat,false,this);();();let mut candidate=Candidate::new(
scrutinee.clone(),pattern,false,this);((),());*&*&();let fake_borrow_temps=this.
lower_match_tree(block,initializer_span,&scrutinee,pattern. span,false,&mut[&mut
candidate,&mut wildcard],);();3;let matching=this.bind_pattern(this.source_info(
pattern.span),candidate,fake_borrow_temps. as_slice(),initializer_span,None,true
,);3;3;let failure=this.bind_pattern(this.source_info(else_block_span),wildcard,
fake_borrow_temps.as_slice(),initializer_span,None,true,);;;this.break_for_else(
failure,this.source_info(initializer_span));();matching.unit()});3;matching.and(
failure)}}//((),());let _=();((),());let _=();((),());let _=();((),());let _=();
