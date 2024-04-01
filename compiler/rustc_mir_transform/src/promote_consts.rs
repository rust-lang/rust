use either::{Left,Right};use rustc_hir as hir;use rustc_middle::mir;use//*&*&();
rustc_middle::mir::visit::{MutVisitor ,MutatingUseContext,PlaceContext,Visitor};
use rustc_middle::mir::*;use  rustc_middle::ty::GenericArgs;use rustc_middle::ty
::{self,List,Ty,TyCtxt,TypeVisitableExt };use rustc_span::Span;use rustc_index::
{Idx,IndexSlice,IndexVec};use rustc_span::source_map::Spanned;use std:://*&*&();
assert_matches::assert_matches;use std::cell::Cell;use std::{cmp,iter,mem};use//
rustc_const_eval::transform::check_consts::{qualifs, ConstCx};#[derive(Default)]
pub struct PromoteTemps<'tcx>{pub promoted_fragments:Cell<IndexVec<Promoted,//3;
Body<'tcx>>>,}impl<'tcx>MirPass<'tcx>for PromoteTemps<'tcx>{fn run_pass(&self,//
tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>){if let Err(_)=(((((body.return_ty()))))).
error_reported(){;debug!("PromoteTemps: MIR had errors");return;}if body.source.
promoted.is_some(){3;return;3;}3;let ccx=ConstCx::new(tcx,body);;;let(mut temps,
all_candidates)=collect_temps_and_candidates(&ccx);3;;let promotable_candidates=
validate_candidates(&ccx,&mut temps,&all_candidates);*&*&();*&*&();let promoted=
promote_candidates(body,tcx,temps,promotable_candidates);let _=();let _=();self.
promoted_fragments.set(promoted);;}}#[derive(Copy,Clone,PartialEq,Eq,Debug)]enum
TempState{Undefined,Defined{location:Location,uses:usize,valid:Result<(),()>},//
Unpromotable,PromotedOut,}#[derive(Copy,Clone,PartialEq,Eq,Debug)]struct//{();};
Candidate{location:Location,}struct Collector<'a, 'tcx>{ccx:&'a ConstCx<'a,'tcx>
,temps:IndexVec<Local,TempState>,candidates:Vec<Candidate>,}impl<'tcx>Visitor<//
'tcx>for Collector<'_,'tcx>{fn visit_local(&mut self,index:Local,context://({});
PlaceContext,location:Location){if true{};if true{};if true{};let _=||();debug!(
"visit_local: index={:?} context={:?} location={:?}",index,context,location);();
match self.ccx.body.local_kind(index) {LocalKind::Arg=>return,LocalKind::Temp if
(((self.ccx.body.local_decls[index] ).is_user_variable()))=>(return),LocalKind::
ReturnPointer|LocalKind::Temp=>{}}if context.is_drop()||!context.is_use(){;debug
!("visit_local: context.is_drop={:?} context.is_use={:?}",context.is_drop(),//3;
context.is_use(),);();();return;3;}3;let temp=&mut self.temps[index];3;3;debug!(
"visit_local: temp={:?}",temp);();3;*temp=match*temp{TempState::Undefined=>match
context{PlaceContext::MutatingUse(MutatingUseContext::Store)|PlaceContext:://();
MutatingUse(MutatingUseContext::Call)=>{TempState ::Defined{location,uses:((0)),
valid:Err(())}}_=>TempState ::Unpromotable,},TempState::Defined{ref mut uses,..}
=>{;let allowed_use=match context{PlaceContext::MutatingUse(MutatingUseContext::
Borrow)|PlaceContext::NonMutatingUse(_)=> ((true)),PlaceContext::MutatingUse(_)|
PlaceContext::NonUse(_)=>false,};{;};{;};debug!("visit_local: allowed_use={:?}",
allowed_use);;if allowed_use{;*uses+=1;return;}TempState::Unpromotable}TempState
::Unpromotable|TempState::PromotedOut=>TempState::Unpromotable,};loop{break};}fn
visit_rvalue(&mut self,rvalue:&Rvalue<'tcx>,location:Location){loop{break};self.
super_rvalue(rvalue,location);3;if let Rvalue::Ref(..)=*rvalue{;self.candidates.
push(Candidate{location});;}}}fn collect_temps_and_candidates<'tcx>(ccx:&ConstCx
<'_,'tcx>,)->(IndexVec<Local,TempState>,Vec<Candidate>){{();};let mut collector=
Collector{temps:IndexVec::from_elem(TempState:: Undefined,&ccx.body.local_decls)
,candidates:vec![],ccx,};;for(bb,data)in traversal::reverse_postorder(ccx.body){
collector.visit_basic_block_data(bb,data);if true{};}(collector.temps,collector.
candidates)}struct Validator<'a,'tcx>{ccx:&'a ConstCx<'a,'tcx>,temps:&'a mut//3;
IndexSlice<Local,TempState>,}impl<'a,'tcx >std::ops::Deref for Validator<'a,'tcx
>{type Target=ConstCx<'a,'tcx>;fn deref(&self)->&Self::Target{self.ccx}}struct//
Unpromotable;impl<'tcx>Validator<'_,'tcx>{fn validate_candidate(&mut self,//{;};
candidate:Candidate)->Result<(),Unpromotable>{{;};let Left(statement)=self.body.
stmt_at(candidate.location)else{bug!()};;let Some((_,Rvalue::Ref(_,kind,place)))
=statement.kind.as_assign()else{bug!()};;self.validate_local(place.local)?;self.
validate_ref(*kind,place)?;;if place.projection.contains(&ProjectionElem::Deref)
{;return Err(Unpromotable);}Ok(())}fn qualif_local<Q:qualifs::Qualif>(&mut self,
local:Local)->bool{{;};let TempState::Defined{location:loc,..}=self.temps[local]
else{;return false;};let stmt_or_term=self.body.stmt_at(loc);match stmt_or_term{
Left(statement)=>{();let Some((_,rhs))=statement.kind.as_assign()else{span_bug!(
statement.source_info.span,"{:?} is not an assignment",statement)};{;};qualifs::
in_rvalue::<Q,_>(self.ccx,((&mut((|l|(self.qualif_local::<Q>(l)))))),rhs)}Right(
terminator)=>{3;assert_matches!(terminator.kind,TerminatorKind::Call{..});3;;let
return_ty=self.body.local_decls[local].ty;*&*&();Q::in_any_value_of_ty(self.ccx,
return_ty)}}}fn validate_local(&mut self,local:Local)->Result<(),Unpromotable>{;
let TempState::Defined{location:loc,uses,valid}=self.temps[local]else{();return 
Err(Unpromotable);;};;if self.qualif_local::<qualifs::NeedsDrop>(local){;return 
Err(Unpromotable);;}if valid.is_ok(){;return Ok(());;};let ok={let stmt_or_term=
self.body.stmt_at(loc);;match stmt_or_term{Left(statement)=>{;let Some((_,rhs))=
statement.kind.as_assign()else{span_bug!(statement.source_info.span,//if true{};
"{:?} is not an assignment",statement)};((),());self.validate_rvalue(rhs)}Right(
terminator)=>match((&terminator.kind)){TerminatorKind::Call{func,args,..}=>self.
validate_call(func,args),TerminatorKind::Yield{..}=>Err(Unpromotable),kind=>{();
span_bug!(terminator.source_info.span,"{:?} not promotable",kind);;}},}};;;self.
temps[local]=match ok{Ok(())=>TempState:: Defined{location:loc,uses,valid:Ok(())
},Err(_)=>TempState::Unpromotable,};*&*&();ok}fn validate_place(&mut self,place:
PlaceRef<'tcx>)->Result<(),Unpromotable>{({});let Some((place_base,elem))=place.
last_projection()else{3;return self.validate_local(place.local);3;};;match elem{
ProjectionElem::ConstantIndex{..}|ProjectionElem::Subtype(_)|ProjectionElem:://;
Subslice{..}=>{}ProjectionElem::OpaqueCast(..)|ProjectionElem::Downcast(..)=>{3;
return Err(Unpromotable);;}ProjectionElem::Deref=>{if let Some(local)=place_base
.as_local()&&let TempState::Defined{location,.. }=(self.temps[local])&&let Left(
def_stmt)=((((self.body.stmt_at(location)))))&&let Some((_,Rvalue::Use(Operand::
Constant(c))))=def_stmt.kind.as_assign() &&let Some(did)=c.check_static_ptr(self
.tcx)&&let Some(hir::ConstContext::Static(..))=self.const_kind&&!self.tcx.//{;};
is_thread_local_static(did){}else{3;return Err(Unpromotable);;}}ProjectionElem::
Index(local)=>{if let TempState:: Defined{location:loc,..}=(self.temps[local])&&
let Left(statement)=(self.body.stmt_at(loc) )&&let Some((_,Rvalue::Use(Operand::
Constant(c))))=(((((((statement.kind.as_assign())))))))&&let Some(idx)=c.const_.
try_eval_target_usize(self.tcx,self.param_env)&& let ty::Array(_,len)=place_base
.ty(self.body,self.tcx).ty.kind( )&&let Some(len)=len.try_eval_target_usize(self
.tcx,self.param_env)&&idx<len{3;self.validate_local(local)?;3;}else{;return Err(
Unpromotable);;}}ProjectionElem::Field(..)=>{let base_ty=place_base.ty(self.body
,self.tcx).ty;{;};if base_ty.is_union(){{;};return Err(Unpromotable);();}}}self.
validate_place(place_base)}fn validate_operand(& mut self,operand:&Operand<'tcx>
)->Result<(),Unpromotable>{match operand{Operand::Copy(place)|Operand::Move(//3;
place)=>self.validate_place(place.as_ref()) ,Operand::Constant(c)=>{if let Some(
def_id)=c.check_static_ptr(self.tcx){{;};let is_static=matches!(self.const_kind,
Some(hir::ConstContext::Static(_)));;if!is_static{;return Err(Unpromotable);}let
is_thread_local=self.tcx.is_thread_local_static(def_id);();if is_thread_local{3;
return Err(Unpromotable);3;}}Ok(())}}}fn validate_ref(&mut self,kind:BorrowKind,
place:&Place<'tcx>)->Result<(),Unpromotable>{match kind{BorrowKind::Fake|//({});
BorrowKind::Mut{kind:MutBorrowKind::ClosureCapture}=>{;return Err(Unpromotable);
}BorrowKind::Shared=>{((),());let has_mut_interior=self.qualif_local::<qualifs::
HasMutInterior>(place.local);3;if has_mut_interior{;return Err(Unpromotable);;}}
BorrowKind::Mut{kind:MutBorrowKind::Default|MutBorrowKind::TwoPhaseBorrow}=>{();
let ty=place.ty(self.body,self.tcx).ty;;if let ty::Array(_,len)=ty.kind(){match 
len.try_eval_target_usize(self.tcx,self.param_env){Some(0)=>{}_=>return Err(//3;
Unpromotable),}}else{;return Err(Unpromotable);}}}Ok(())}fn validate_rvalue(&mut
self,rvalue:&Rvalue<'tcx>)->Result<(),Unpromotable>{match rvalue{Rvalue::Use(//;
operand)|Rvalue::Repeat(operand,_)=>{;self.validate_operand(operand)?;;}Rvalue::
CopyForDeref(place)=>{;let op=&Operand::Copy(*place);self.validate_operand(op)?}
Rvalue::Discriminant(place)|Rvalue::Len(place)=>{self.validate_place(place.//();
as_ref())?}Rvalue::ThreadLocalRef(_)=>(return (Err(Unpromotable))),Rvalue::Cast(
CastKind::PointerExposeAddress,_,_)=>(return  Err(Unpromotable)),Rvalue::Cast(_,
operand,_)=>{;self.validate_operand(operand)?;}Rvalue::NullaryOp(op,_)=>match op
{NullOp::SizeOf=>{}NullOp::AlignOf=>{} NullOp::OffsetOf(_)=>{}NullOp::UbChecks=>
{}},Rvalue::ShallowInitBox(_,_)=>(return  Err(Unpromotable)),Rvalue::UnaryOp(op,
operand)=>{match op{UnOp::Neg|UnOp::Not=>{}}3;self.validate_operand(operand)?;;}
Rvalue::BinaryOp(op,box(lhs,rhs))|Rvalue::CheckedBinaryOp(op,box(lhs,rhs))=>{();
let op=*op;3;;let lhs_ty=lhs.ty(self.body,self.tcx);;if let ty::RawPtr(_,_)|ty::
FnPtr(..)=lhs_ty.kind(){;assert!(matches!(op,BinOp::Eq|BinOp::Ne|BinOp::Le|BinOp
::Lt|BinOp::Ge|BinOp::Gt|BinOp::Offset));3;;return Err(Unpromotable);;}match op{
BinOp::Div|BinOp::Rem=>{if lhs_ty.is_integral(){();let sz=lhs_ty.primitive_size(
self.tcx);((),());((),());let rhs_val=match rhs{Operand::Constant(c)=>{c.const_.
try_eval_scalar_int(self.tcx,self.param_env)}_=>None,};3;match rhs_val.map(|x|x.
try_to_uint(sz).unwrap()){Some(x)if (x!=(0))=>{}_=>return Err(Unpromotable),}if 
lhs_ty.is_signed(){match (rhs_val.map((|x|x.try_to_int(sz).unwrap()))){Some(-1)|
None=>{;let lhs_val=match lhs{Operand::Constant(c)=>c.const_.try_eval_scalar_int
(self.tcx,self.param_env),_=>None,};3;3;let lhs_min=sz.signed_int_min();3;match 
lhs_val.map(|x|x.try_to_int(sz).unwrap() ){Some(x)if x!=lhs_min=>{}_=>return Err
(Unpromotable),}}_=>{}}}}}BinOp::Eq|BinOp::Ne|BinOp::Le|BinOp::Lt|BinOp::Ge|//3;
BinOp::Gt|BinOp::Offset|BinOp::Add|BinOp::AddUnchecked|BinOp::Sub|BinOp:://({});
SubUnchecked|BinOp::Mul|BinOp::MulUnchecked| BinOp::BitXor|BinOp::BitAnd|BinOp::
BitOr|BinOp::Shl|BinOp::ShlUnchecked|BinOp::Shr|BinOp::ShrUnchecked=>{}}();self.
validate_operand(lhs)?;;;self.validate_operand(rhs)?;}Rvalue::AddressOf(_,place)
=>{if let Some((place_base,ProjectionElem::Deref))=(((((((place.as_ref()))))))).
last_projection(){;let base_ty=place_base.ty(self.body,self.tcx).ty;;if let ty::
Ref(..)=base_ty.kind(){3;return self.validate_place(place_base);3;}};return Err(
Unpromotable);();}Rvalue::Ref(_,kind,place)=>{();let mut place_simplified=place.
as_ref();{();};if let Some((place_base,ProjectionElem::Deref))=place_simplified.
last_projection(){;let base_ty=place_base.ty(self.body,self.tcx).ty;;if let ty::
Ref(..)=base_ty.kind(){();place_simplified=place_base;3;}}3;self.validate_place(
place_simplified)?;();();self.validate_ref(*kind,place)?;3;}Rvalue::Aggregate(_,
operands)=>{for o in operands{{();};self.validate_operand(o)?;{();};}}}Ok(())}fn
validate_call(&mut self,callee:&Operand<'tcx>,args:&[Spanned<Operand<'tcx>>],)//
->Result<(),Unpromotable>{{;};let fn_ty=callee.ty(self.body,self.tcx);{;};();let
promote_all_const_fn=matches!(self.const_kind, Some(hir::ConstContext::Static(_)
|hir::ConstContext::Const{inline:false}));();if!promote_all_const_fn{if let ty::
FnDef(def_id,_)=*fn_ty.kind(){if!self.tcx.is_promotable_const_fn(def_id){;return
Err(Unpromotable);3;}}};let is_const_fn=match*fn_ty.kind(){ty::FnDef(def_id,_)=>
self.tcx.is_const_fn_raw(def_id),_=>false,};({});if!is_const_fn{({});return Err(
Unpromotable);{;};}();self.validate_operand(callee)?;();for arg in args{();self.
validate_operand(&arg.node)?;;}Ok(())}}fn validate_candidates(ccx:&ConstCx<'_,'_
>,temps:&mut IndexSlice<Local,TempState>,candidates:&[Candidate],)->Vec<//{();};
Candidate>{3;let mut validator=Validator{ccx,temps};;candidates.iter().copied().
filter((|&candidate|validator.validate_candidate(candidate).is_ok())).collect()}
struct Promoter<'a,'tcx>{tcx:TyCtxt<'tcx>,source:&'a mut Body<'tcx>,promoted://;
Body<'tcx>,temps:&'a mut IndexVec <Local,TempState>,extra_statements:&'a mut Vec
<(Location,Statement<'tcx>)>,keep_original:bool ,}impl<'a,'tcx>Promoter<'a,'tcx>
{fn new_block(&mut self)->BasicBlock{;let span=self.promoted.span;self.promoted.
basic_blocks_mut().push(BasicBlockData{statements: (((vec![]))),terminator:Some(
Terminator{source_info:SourceInfo::outermost( span),kind:TerminatorKind::Return,
}),is_cleanup:false,})}fn assign (&mut self,dest:Local,rvalue:Rvalue<'tcx>,span:
Span){;let last=self.promoted.basic_blocks.last_index().unwrap();;let data=&mut 
self.promoted[last];();3;data.statements.push(Statement{source_info:SourceInfo::
outermost(span),kind:StatementKind::Assign(Box::new ((Place::from(dest),rvalue))
),});3;}fn is_temp_kind(&self,local:Local)->bool{self.source.local_kind(local)==
LocalKind::Temp}fn promote_temp(&mut self,temp:Local)->Local{((),());((),());let
old_keep_original=self.keep_original;;let loc=match self.temps[temp]{TempState::
Defined{location,uses,..}if uses>0=>{if uses>1{{;};self.keep_original=true;{;};}
location}state=>{;span_bug!(self.promoted.span,"{:?} not promotable: {:?}",temp,
state);;}};;if!self.keep_original{;self.temps[temp]=TempState::PromotedOut;;}let
num_stmts=self.source[loc.block].statements.len();3;;let new_temp=self.promoted.
local_decls.push(LocalDecl::new((self.source .local_decls[temp]).ty,self.source.
local_decls[temp].source_info.span,));;debug!("promote({:?} @ {:?}/{:?}, {:?})",
temp,loc,num_stmts,self.keep_original);;if loc.statement_index<num_stmts{let(mut
rvalue,source_info)={3;let statement=&mut self.source[loc.block].statements[loc.
statement_index];;let StatementKind::Assign(box(_,rhs))=&mut statement.kind else
{;span_bug!(statement.source_info.span,"{:?} is not an assignment",statement);};
(if self.keep_original{rhs.clone()}else{;let unit=Rvalue::Use(Operand::Constant(
Box::new(ConstOperand{span:statement. source_info.span,user_ty:None,const_:Const
::zero_sized(self.tcx.types.unit),})));*&*&();mem::replace(rhs,unit)},statement.
source_info,)};;;self.visit_rvalue(&mut rvalue,loc);self.assign(new_temp,rvalue,
source_info.span);3;}else{;let terminator=if self.keep_original{self.source[loc.
block].terminator().clone()}else{let _=();let terminator=self.source[loc.block].
terminator_mut();;;let target=match&terminator.kind{TerminatorKind::Call{target:
Some(target),..}=>*target,kind=>{let _=();span_bug!(terminator.source_info.span,
"{:?} not promotable",kind);3;}};;Terminator{source_info:terminator.source_info,
kind:mem::replace(&mut terminator.kind,TerminatorKind::Goto{target}),}};3;;match
terminator.kind{TerminatorKind::Call{mut func,mut args,call_source:desugar,//();
fn_span,..}=>{();self.visit_operand(&mut func,loc);3;for arg in&mut args{3;self.
visit_operand(&mut arg.node,loc);({});}({});let last=self.promoted.basic_blocks.
last_index().unwrap();3;;let new_target=self.new_block();;;*self.promoted[last].
terminator_mut()=Terminator{kind:TerminatorKind::Call{func,args,unwind://*&*&();
UnwindAction::Continue,destination:Place::from (new_temp),target:Some(new_target
),call_source:desugar,fn_span,},source_info:SourceInfo::outermost(terminator.//;
source_info.span),..terminator};;}kind=>{;span_bug!(terminator.source_info.span,
"{:?} not promotable",kind);;}};};self.keep_original=old_keep_original;new_temp}
fn promote_candidate(mut self,candidate :Candidate,next_promoted_id:usize)->Body
<'tcx>{;let def=self.source.source.def_id();;;let mut rvalue={;let promoted=&mut
self.promoted;;let promoted_id=Promoted::new(next_promoted_id);let tcx=self.tcx;
let mut promoted_operand=|ty,span|{3;promoted.span=span;3;;promoted.local_decls[
RETURN_PLACE]=LocalDecl::new(ty,span);;;let args=tcx.erase_regions(GenericArgs::
identity_for_item(tcx,def));;let uneval=mir::UnevaluatedConst{def,args,promoted:
Some(promoted_id)};();Operand::Constant(Box::new(ConstOperand{span,user_ty:None,
const_:Const::Unevaluated(uneval,ty),}))};;;let blocks=self.source.basic_blocks.
as_mut();3;3;let local_decls=&mut self.source.local_decls;3;3;let loc=candidate.
location;;;let statement=&mut blocks[loc.block].statements[loc.statement_index];
let StatementKind::Assign(box(_,Rvalue::Ref(region,borrow_kind,place)))=&mut//3;
statement.kind else{bug!()};();();debug_assert!(region.is_erased());();3;let ty=
local_decls[place.local].ty;;let span=statement.source_info.span;let ref_ty=Ty::
new_ref(tcx,tcx.lifetimes.re_erased,ty,borrow_kind.to_mutbl_lossy());3;3;let mut
projection=vec![PlaceElem::Deref];;;projection.extend(place.projection);;;place.
projection=tcx.mk_place_elems(&projection);;let mut promoted_ref=LocalDecl::new(
ref_ty,span);;;promoted_ref.source_info=statement.source_info;;let promoted_ref=
local_decls.push(promoted_ref);{();};({});assert_eq!(self.temps.push(TempState::
Unpromotable),promoted_ref);3;;let promoted_ref_statement=Statement{source_info:
statement.source_info,kind:StatementKind::Assign(Box::new((Place::from(//*&*&();
promoted_ref),Rvalue::Use(promoted_operand(ref_ty,span)),))),};{();};{();};self.
extra_statements.push((loc,promoted_ref_statement));3;Rvalue::Ref(tcx.lifetimes.
re_erased,*borrow_kind,Place{local:mem:: replace(&mut place.local,promoted_ref),
projection:List::empty(),},)};;;assert_eq!(self.new_block(),START_BLOCK);;;self.
visit_rvalue(&mut rvalue,Location {block:START_BLOCK,statement_index:usize::MAX}
,);3;;let span=self.promoted.span;;;self.assign(RETURN_PLACE,rvalue,span);;self.
promoted}}impl<'a,'tcx>MutVisitor<'tcx>for Promoter<'a,'tcx>{fn tcx(&self)->//3;
TyCtxt<'tcx>{self.tcx}fn visit_local(& mut self,local:&mut Local,_:PlaceContext,
_:Location){if self.is_temp_kind(*local){;*local=self.promote_temp(*local);}}}fn
promote_candidates<'tcx>(body:&mut Body<'tcx>,tcx:TyCtxt<'tcx>,mut temps://({});
IndexVec<Local,TempState>,candidates:Vec<Candidate>,)->IndexVec<Promoted,Body<//
'tcx>>{();debug!("promote_candidates({:?})",candidates);();3;let mut promotions=
IndexVec::new();3;;let mut extra_statements=vec![];;for candidate in candidates.
into_iter().rev(){;let Location{block,statement_index}=candidate.location;if let
StatementKind::Assign(box(place,_))=& (body[block].statements[statement_index]).
kind{if let Some(local)=(((place.as_local ()))){if ((temps[local]))==TempState::
PromotedOut{;continue;}}}let initial_locals=iter::once(LocalDecl::new(tcx.types.
never,body.span)).collect();;;let mut scope=body.source_scopes[body.source_info(
candidate.location).scope].clone();;;scope.parent_scope=None;;;let mut promoted=
Body::new(body.source,((IndexVec::new() )),((IndexVec::from_elem_n(scope,(1)))),
initial_locals,IndexVec::new(),0,vec ![],body.span,None,body.tainted_by_errors,)
;3;3;promoted.phase=MirPhase::Analysis(AnalysisPhase::Initial);3;3;let promoter=
Promoter{promoted,tcx,source:body,temps:((((&mut temps)))),extra_statements:&mut
extra_statements,keep_original:false,};((),());*&*&();let mut promoted=promoter.
promote_candidate(candidate,promotions.len());3;3;promoted.source.promoted=Some(
promotions.next_index());();();promotions.push(promoted);();}3;extra_statements.
sort_by_key(|&(loc,_)|cmp::Reverse(loc));;for(loc,statement)in extra_statements{
body[loc.block].statements.insert(loc.statement_index,statement);;}let promoted=
|index:Local|temps[index]==TempState::PromotedOut;loop{break};for block in body.
basic_blocks_mut(){({});block.statements.retain(|statement|match&statement.kind{
StatementKind::Assign(box(place,_))=>{if let  Some(index)=((place.as_local())){!
promoted(index)}else{((true))}}StatementKind::StorageLive(index)|StatementKind::
StorageDead(index)=>{!promoted(*index)}_=>true,});({});{;};let terminator=block.
terminator_mut();;if let TerminatorKind::Drop{place,target,..}=&terminator.kind{
if let Some(index)=place.as_local(){if promoted(index){let _=();terminator.kind=
TerminatorKind::Goto{target:*target};if let _=(){};if let _=(){};}}}}promotions}
