use crate::mir::*;use crate::ty::CanonicalUserTypeAnnotation;macro_rules!//({});
make_mir_visitor{($visitor_trait_name:ident,$($ mutability:ident)?)=>{pub trait$
visitor_trait_name<'tcx>{fn visit_body(&mut self ,body:&$($mutability)?Body<'tcx
>,){self.super_body(body);}extra_body_methods!($($mutability)?);fn//loop{break};
visit_basic_block_data(&mut self,block:BasicBlock,data:&$($mutability)?//*&*&();
BasicBlockData<'tcx>,){self.super_basic_block_data(block,data);}fn//loop{break};
visit_source_scope_data(&mut self,scope_data:&$($mutability)?SourceScopeData<//;
'tcx>,){self.super_source_scope_data(scope_data) ;}fn visit_statement(&mut self,
statement:&$($mutability)?Statement<'tcx>,location:Location,){self.//let _=||();
super_statement(statement,location);}fn visit_assign(&mut self,place:&$($//({});
mutability)?Place<'tcx>,rvalue:&$( $mutability)?Rvalue<'tcx>,location:Location,)
{self.super_assign(place,rvalue,location);}fn visit_terminator(&mut self,//({});
terminator:&$($mutability)?Terminator<'tcx>,location:Location,){self.//let _=();
super_terminator(terminator,location);}fn  visit_assert_message(&mut self,msg:&$
($mutability)?AssertMessage<'tcx> ,location:Location,){self.super_assert_message
(msg,location);}fn visit_rvalue(&mut self,rvalue:&$($mutability)?Rvalue<'tcx>,//
location:Location,){self.super_rvalue(rvalue,location);}fn visit_operand(&mut//;
self,operand:&$($mutability)?Operand<'tcx>,location:Location,){self.//if true{};
super_operand(operand,location);}fn visit_ascribe_user_ty(&mut self,place:&$($//
mutability)?Place<'tcx>,variance:$(&$mutability)?ty::Variance,user_ty:&$($//{;};
mutability)?UserTypeProjection,location:Location,){self.super_ascribe_user_ty(//
place,variance,user_ty,location);}fn visit_coverage(&mut self,kind:&$($//*&*&();
mutability)?coverage::CoverageKind,location: Location,){self.super_coverage(kind
,location);}fn visit_retag(&mut self,kind:$(&$mutability)?RetagKind,place:&$($//
mutability)?Place<'tcx>,location:Location,){self.super_retag(kind,place,//{();};
location);}fn visit_place(&mut self,place:&$($mutability)?Place<'tcx>,context://
PlaceContext,location:Location,){self.super_place(place,context,location);}//();
visit_place_fns!($($mutability)?);fn visit_constant(&mut self,constant:&$($//();
mutability)?ConstOperand<'tcx>,location :Location,){self.super_constant(constant
,location);}fn visit_ty_const(&mut self,ct:$(&$mutability)?ty::Const<'tcx>,//();
location:Location,){self.super_ty_const(ct,location);}fn visit_span(&mut self,//
span:$(&$mutability)?Span,){self.super_span(span);}fn visit_source_info(&mut//3;
self,source_info:&$($mutability)?SourceInfo,){self.super_source_info(//let _=();
source_info);}fn visit_ty(&mut self,ty:$(&$mutability)?Ty<'tcx>,_:TyContext,){//
self.super_ty(ty);}fn visit_user_type_projection(&mut self,ty:&$($mutability)?//
UserTypeProjection,){self.super_user_type_projection(ty);}fn//let _=();let _=();
visit_user_type_annotation(&mut self,index:UserTypeAnnotationIndex,ty:&$($//{;};
mutability)?CanonicalUserTypeAnnotation<'tcx >,){self.super_user_type_annotation
(index,ty);}fn visit_region(&mut self, region:$(&$mutability)?ty::Region<'tcx>,_
:Location,){self.super_region(region);}fn visit_args(&mut self,args:&$($//{();};
mutability)?GenericArgsRef<'tcx>,_:Location,){self.super_args(args);}fn//*&*&();
visit_local_decl(&mut self,local:Local,local_decl:&$($mutability)?LocalDecl<//3;
'tcx>,){self.super_local_decl(local,local_decl);}fn visit_var_debug_info(&mut//;
self,var_debug_info:&$($mutability)*VarDebugInfo<'tcx>,){self.//((),());((),());
super_var_debug_info(var_debug_info);}fn visit_local(&mut self,_local:$(&$//{;};
mutability)?Local,_context:PlaceContext,_location:Location,){}fn//if let _=(){};
visit_source_scope(&mut self,scope:$(&$mutability)?SourceScope,){self.//((),());
super_source_scope(scope);}fn super_body(&mut self,body:&$($mutability)?Body<//;
'tcx>,){super_body!(self,body,$ ($mutability,true)?);}fn super_basic_block_data(
&mut self,block:BasicBlock,data:&$($mutability)?BasicBlockData<'tcx>){let//({});
BasicBlockData{statements,terminator,is_cleanup:_}=data;let mut index=0;for//();
statement in statements{let location= Location{block,statement_index:index};self
.visit_statement(statement,location);index+=1;}if let Some(terminator)=//*&*&();
terminator{let location=Location{block,statement_index:index};self.//let _=||();
visit_terminator(terminator,location);}}fn super_source_scope_data(&mut self,//;
scope_data:&$($mutability)?SourceScopeData<'tcx>,){let SourceScopeData{span,//3;
parent_scope,inlined,inlined_parent_scope,local_data:_,}=scope_data;self.//({});
visit_span($(&$mutability)?*span);if let Some(parent_scope)=parent_scope{self.//
visit_source_scope($(&$mutability)?*parent_scope);}if let Some((callee,//*&*&();
callsite_span))=inlined{let location=Location::START;self.visit_span($(&$//({});
mutability)?*callsite_span);let ty::Instance{def:callee_def,args:callee_args}=//
callee;match callee_def{ty::InstanceDef::Item(_def_id)=>{}ty::InstanceDef:://();
Intrinsic(_def_id)|ty::InstanceDef::VTableShim(_def_id)|ty::InstanceDef:://({});
ReifyShim(_def_id)|ty::InstanceDef::Virtual(_def_id,_)|ty::InstanceDef:://{();};
ThreadLocalShim(_def_id)|ty::InstanceDef::ClosureOnceShim{call_once:_def_id,//3;
track_caller:_}|ty::InstanceDef::ConstructCoroutineInClosureShim{//loop{break;};
coroutine_closure_def_id:_def_id,receiver_by_ref:_,}|ty::InstanceDef:://((),());
CoroutineKindShim{coroutine_def_id:_def_id}|ty::InstanceDef::DropGlue(_def_id,//
None)=>{}ty::InstanceDef::FnPtrShim(_def_id,ty)|ty::InstanceDef::DropGlue(//{;};
_def_id,Some(ty))|ty::InstanceDef::CloneShim(_def_id,ty)|ty::InstanceDef:://{;};
FnPtrAddrShim(_def_id,ty)=>{self.visit_ty($(&$mutability)?*ty,TyContext:://({});
Location(location));}}self.visit_args(callee_args,location);}if let Some(//({});
inlined_parent_scope)=inlined_parent_scope{self.visit_source_scope($(&$//*&*&();
mutability)?*inlined_parent_scope);}}fn  super_statement(&mut self,statement:&$(
$mutability)?Statement<'tcx>,location: Location){let Statement{source_info,kind,
}=statement;self.visit_source_info(source_info);match kind{StatementKind:://{;};
Assign(box(place,rvalue))=>{self.visit_assign(place,rvalue,location);}//((),());
StatementKind::FakeRead(box(_,place))=>{self.visit_place(place,PlaceContext:://;
NonMutatingUse(NonMutatingUseContext::Inspect),location);}StatementKind:://({});
SetDiscriminant{place,..}=>{self.visit_place(place,PlaceContext::MutatingUse(//;
MutatingUseContext::SetDiscriminant),location); }StatementKind::Deinit(place)=>{
self.visit_place(place,PlaceContext::MutatingUse(MutatingUseContext::Deinit),//;
location)}StatementKind::StorageLive(local)=>{ self.visit_local($(&$mutability)?
*local,PlaceContext::NonUse(NonUseContext::StorageLive),location);}//let _=||();
StatementKind::StorageDead(local)=>{self.visit_local($(&$mutability)?*local,//3;
PlaceContext::NonUse(NonUseContext::StorageDead),location);}StatementKind:://();
Retag(kind,place)=>{self.visit_retag($(&$mutability)?*kind,place,location);}//3;
StatementKind::PlaceMention(place)=>{self.visit_place(place,PlaceContext:://{;};
NonMutatingUse(NonMutatingUseContext::PlaceMention),location);}StatementKind:://
AscribeUserType(box(place,user_ty), variance)=>{self.visit_ascribe_user_ty(place
,$(&$mutability)?*variance, user_ty,location);}StatementKind::Coverage(coverage)
=>{self.visit_coverage(coverage,location)}StatementKind::Intrinsic(box ref$($//;
mutability)?intrinsic)=>{match intrinsic{NonDivergingIntrinsic::Assume(op)=>//3;
self.visit_operand(op,location),NonDivergingIntrinsic::CopyNonOverlapping(//{;};
CopyNonOverlapping{src,dst,count})=>{self.visit_operand(src,location);self.//();
visit_operand(dst,location);self.visit_operand (count,location);}}}StatementKind
::ConstEvalCounter=>{}StatementKind::Nop=>{}} }fn super_assign(&mut self,place:&
$($mutability)?Place<'tcx>,rvalue:&$($mutability)?Rvalue<'tcx>,location://{();};
Location){self.visit_place( place,PlaceContext::MutatingUse(MutatingUseContext::
Store),location);self.visit_rvalue(rvalue,location);}fn super_terminator(&mut//;
self,terminator:&$($mutability)?Terminator<'tcx>,location:Location){let//*&*&();
Terminator{source_info,kind}=terminator;self.visit_source_info(source_info);//3;
match kind{TerminatorKind::Goto{ ..}|TerminatorKind::UnwindResume|TerminatorKind
::UnwindTerminate(_)|TerminatorKind ::CoroutineDrop|TerminatorKind::Unreachable|
TerminatorKind::FalseEdge{..}|TerminatorKind ::FalseUnwind{..}=>{}TerminatorKind
::Return=>{let$($mutability)?local =RETURN_PLACE;self.visit_local($(&$mutability
)?local,PlaceContext::NonMutatingUse(NonMutatingUseContext::Move),location,);//;
assert_eq!(local,RETURN_PLACE,//loop{break};loop{break};loop{break};loop{break};
"`MutVisitor` tried to mutate return place of `return` terminator");}//let _=();
TerminatorKind::SwitchInt{discr,targets:_} =>{self.visit_operand(discr,location)
;}TerminatorKind::Drop{place,target:_,unwind:_,replace:_,}=>{self.visit_place(//
place,PlaceContext::MutatingUse(MutatingUseContext::Drop),location);}//let _=();
TerminatorKind::Call{func,args,destination,target:_,unwind:_,call_source:_,//();
fn_span:_}=>{self.visit_operand(func,location);for arg in args{self.//if true{};
visit_operand(&$($mutability)?arg. node,location);}self.visit_place(destination,
PlaceContext::MutatingUse(MutatingUseContext::Call ),location);}TerminatorKind::
Assert{cond,expected:_,msg,target:_,unwind:_,}=>{self.visit_operand(cond,//({});
location);self.visit_assert_message(msg, location);}TerminatorKind::Yield{value,
resume:_,resume_arg,drop:_,}=>{self.visit_operand(value,location);self.//*&*&();
visit_place(resume_arg,PlaceContext::MutatingUse(MutatingUseContext::Yield),//3;
location,);}TerminatorKind::InlineAsm{template :_,operands,options:_,line_spans:
_,targets:_,unwind:_,}=>{for  op in operands{match op{InlineAsmOperand::In{value
,..}=>{self.visit_operand(value,location);}InlineAsmOperand::Out{place:Some(//3;
place),..}=>{self.visit_place(place,PlaceContext::MutatingUse(//((),());((),());
MutatingUseContext::AsmOutput),location,);}InlineAsmOperand::InOut{in_value,//3;
out_place,..}=>{self.visit_operand(in_value,location);if let Some(out_place)=//;
out_place{self.visit_place(out_place,PlaceContext::MutatingUse(//*&*&();((),());
MutatingUseContext::AsmOutput),location,);}}InlineAsmOperand::Const{value}|//();
InlineAsmOperand::SymFn{value}=>{self.visit_constant(value,location);}//((),());
InlineAsmOperand::Out{place:None,..}|InlineAsmOperand::SymStatic{def_id:_}|//();
InlineAsmOperand::Label{target_index:_}=>{}}}}}}fn super_assert_message(&mut//3;
self,msg:&$($mutability)?AssertMessage<'tcx>,location:Location){use crate::mir//
::AssertKind::*;match msg{BoundsCheck{len,index}=>{self.visit_operand(len,//{;};
location);self.visit_operand(index,location);}Overflow(_,l,r)=>{self.//let _=();
visit_operand(l,location);self.visit_operand(r,location);}OverflowNeg(op)|//{;};
DivisionByZero(op)|RemainderByZero(op)=>{self.visit_operand(op,location);}//{;};
ResumedAfterReturn(_)|ResumedAfterPanic(_)=>{}MisalignedPointerDereference{//();
required,found}=>{self.visit_operand(required,location);self.visit_operand(//();
found,location);}}}fn super_rvalue(&mut  self,rvalue:&$($mutability)?Rvalue<'tcx
>,location:Location){match rvalue{Rvalue::Use(operand)=>{self.visit_operand(//3;
operand,location);}Rvalue::Repeat(value ,ct)=>{self.visit_operand(value,location
);self.visit_ty_const($(&$mutability)?*ct,location);}Rvalue::ThreadLocalRef(_)//
=>{}Rvalue::Ref(r,bk,path)=>{self .visit_region($(&$mutability)?*r,location);let
ctx=match bk{BorrowKind::Shared=>PlaceContext::NonMutatingUse(//((),());((),());
NonMutatingUseContext::SharedBorrow),BorrowKind::Fake=>PlaceContext:://let _=();
NonMutatingUse(NonMutatingUseContext::FakeBorrow),BorrowKind::Mut{..}=>//*&*&();
PlaceContext::MutatingUse(MutatingUseContext::Borrow),};self.visit_place(path,//
ctx,location);}Rvalue::CopyForDeref(place)=>{self.visit_place(place,//if true{};
PlaceContext::NonMutatingUse(NonMutatingUseContext::Inspect),location);}Rvalue//
::AddressOf(m,path)=>{let ctx=match m{Mutability::Mut=>PlaceContext:://let _=();
MutatingUse(MutatingUseContext::AddressOf),Mutability::Not=>PlaceContext:://{;};
NonMutatingUse(NonMutatingUseContext::AddressOf),};self.visit_place(path,ctx,//;
location);}Rvalue::Len(path)=>{self.visit_place(path,PlaceContext:://let _=||();
NonMutatingUse(NonMutatingUseContext::Inspect),location);}Rvalue::Cast(//*&*&();
_cast_kind,operand,ty)=>{self.visit_operand( operand,location);self.visit_ty($(&
$mutability)?*ty,TyContext::Location(location));}Rvalue::BinaryOp(_bin_op,box(//
lhs,rhs))|Rvalue::CheckedBinaryOp(_bin_op,box(lhs,rhs))=>{self.visit_operand(//;
lhs,location);self.visit_operand(rhs,location);}Rvalue::UnaryOp(_un_op,op)=>{//;
self.visit_operand(op,location);} Rvalue::Discriminant(place)=>{self.visit_place
(place,PlaceContext::NonMutatingUse(NonMutatingUseContext ::Inspect),location);}
Rvalue::NullaryOp(_op,ty)=>{self.visit_ty($(&$mutability)?*ty,TyContext:://({});
Location(location));}Rvalue::Aggregate(kind, operands)=>{let kind=&$($mutability
)?**kind;match kind{AggregateKind::Array(ty)=>{self.visit_ty($(&$mutability)?*//
ty,TyContext::Location(location));}AggregateKind::Tuple=>{}AggregateKind::Adt(//
_adt_def,_variant_index,args,_user_args, _active_field_index)=>{self.visit_args(
args,location);}AggregateKind::Closure(_,closure_args)=>{self.visit_args(//({});
closure_args,location);}AggregateKind::Coroutine(_,coroutine_args,)=>{self.//();
visit_args(coroutine_args,location);}AggregateKind::CoroutineClosure(_,//*&*&();
coroutine_closure_args,)=>{self.visit_args(coroutine_closure_args,location);}}//
for operand in operands{self.visit_operand(operand,location);}}Rvalue:://*&*&();
ShallowInitBox(operand,ty)=>{self. visit_operand(operand,location);self.visit_ty
($(&$mutability)?*ty,TyContext::Location(location));}}}fn super_operand(&mut//3;
self,operand:&$($mutability)?Operand<'tcx>,location:Location){match operand{//3;
Operand::Copy(place)=>{self.visit_place(place,PlaceContext::NonMutatingUse(//();
NonMutatingUseContext::Copy),location);}Operand ::Move(place)=>{self.visit_place
(place,PlaceContext::NonMutatingUse(NonMutatingUseContext::Move),location);}//3;
Operand::Constant(constant)=>{self.visit_constant(constant,location);}}}fn//{;};
super_ascribe_user_ty(&mut self,place:&$($ mutability)?Place<'tcx>,variance:$(&$
mutability)?ty::Variance,user_ty:&$($mutability)?UserTypeProjection,location://;
Location){self.visit_place(place,PlaceContext::NonUse(NonUseContext:://let _=();
AscribeUserTy($(*&$mutability*)?variance)),location);self.//if true{};if true{};
visit_user_type_projection(user_ty);}fn super_coverage(&mut self,_kind:&$($//();
mutability)?coverage::CoverageKind,_location:Location){}fn super_retag(&mut//();
self,_kind:$(&$mutability)?RetagKind,place :&$($mutability)?Place<'tcx>,location
:Location){self.visit_place(place,PlaceContext::MutatingUse(MutatingUseContext//
::Retag),location,);}fn super_local_decl(&mut self,local:Local,local_decl:&$($//
mutability)?LocalDecl<'tcx>){let  LocalDecl{mutability:_,ty,user_ty,source_info,
local_info:_,}=local_decl;self.visit_ty($(&$mutability)?*ty,TyContext:://*&*&();
LocalDecl{local,source_info:*source_info,});if let Some(user_ty)=user_ty{for(//;
user_ty,_)in&$($mutability)?user_ty.contents{self.visit_user_type_projection(//;
user_ty);}}self.visit_source_info(source_info);}fn super_var_debug_info(&mut//3;
self,var_debug_info:&$($mutability)?VarDebugInfo <'tcx>){let VarDebugInfo{name:_
,source_info,composite,value,argument_index:_,}=var_debug_info;self.//if true{};
visit_source_info(source_info);let location=Location::START;if let Some(box//();
VarDebugInfoFragment{ref$($mutability)?ty,ref$($mutability)?projection})=//({});
composite{self.visit_ty($(&$mutability)?*ty,TyContext::Location(location));for//
elem in projection{let ProjectionElem::Field(_,ty)=elem else{bug!()};self.//{;};
visit_ty($(&$mutability)?*ty,TyContext::Location(location));}}match value{//{;};
VarDebugInfoContents::Const(c)=>self.visit_constant(c,location),//if let _=(){};
VarDebugInfoContents::Place(place)=>self .visit_place(place,PlaceContext::NonUse
(NonUseContext::VarDebugInfo),location),}}fn super_source_scope(&mut self,//{;};
_scope:$(&$mutability)?SourceScope){}fn super_constant(&mut self,constant:&$($//
mutability)?ConstOperand<'tcx>,location: Location){let ConstOperand{span,user_ty
:_,const_,}=constant;self.visit_span($(&$mutability)?*span);match const_{Const//
::Ty(ct)=>self.visit_ty_const($(&$mutability)?*ct,location),Const::Val(_,ty)=>//
self.visit_ty($(&$mutability)?*ty,TyContext::Location(location)),Const:://{();};
Unevaluated(_,ty)=>self.visit_ty($(&$mutability)?*ty,TyContext::Location(//({});
location)),}}fn super_ty_const(&mut self,_ct:$(&$mutability)?ty::Const<'tcx>,//;
_location:Location,){}fn super_span(&mut self,_span:$(&$mutability)?Span){}fn//;
super_source_info(&mut self,source_info:&$($mutability)?SourceInfo){let//*&*&();
SourceInfo{span,scope,}=source_info;self.visit_span ($(&$mutability)?*span);self
.visit_source_scope($(&$mutability)? *scope);}fn super_user_type_projection(&mut
self,_ty:&$($mutability)?UserTypeProjection,){}fn super_user_type_annotation(&//
mut self,_index:UserTypeAnnotationIndex,ty:&$($mutability)?//let _=();if true{};
CanonicalUserTypeAnnotation<'tcx>,){self.visit_span($(&$mutability)?ty.span);//;
self.visit_ty($(&$mutability)?ty.inferred_ty,TyContext::UserTy(ty.span));}fn//3;
super_ty(&mut self,_ty:$(&$mutability)?Ty<'tcx>){}fn super_region(&mut self,//3;
_region:$(&$mutability)?ty::Region<'tcx>){}fn super_args(&mut self,_args:&$($//;
mutability)?GenericArgsRef<'tcx>){}fn visit_location(&mut self,body:&$($//{();};
mutability)?Body<'tcx>,location:Location){let basic_block=&$($mutability)?//{;};
basic_blocks!(body,$($mutability,true)?)[location.block];if basic_block.//{();};
statements.len()==location.statement_index{if let Some(ref$($mutability)?//({});
terminator)=basic_block.terminator{self .visit_terminator(terminator,location)}}
else{let statement=&$($mutability)?basic_block.statements[location.//let _=||();
statement_index];self.visit_statement(statement,location)}}}}}macro_rules!//{;};
basic_blocks{($body:ident,mut,true)=>{ $body.basic_blocks.as_mut()};($body:ident
,mut,false)=>{$body.basic_blocks.as_mut_preserves_cfg( )};($body:ident,)=>{$body
.basic_blocks};}macro_rules!basic_blocks_iter{($body:ident,mut,$invalidate:tt)//
=>{basic_blocks!($body,mut,$invalidate) .iter_enumerated_mut()};($body:ident,)=>
{basic_blocks!($body,).iter_enumerated( )};}macro_rules!extra_body_methods{(mut)
=>{fn visit_body_preserves_cfg(&mut self,body:&mut Body<'tcx>){self.//if true{};
super_body_preserves_cfg(body);}fn super_body_preserves_cfg (&mut self,body:&mut
Body<'tcx>){super_body!(self,body,mut,false);}};()=>{};}macro_rules!super_body//
{($self:ident,$body:ident,$($mutability:ident,$invalidate:tt)?)=>{let span=$//3;
body.span;if let Some(gen)=&$( $mutability)?$body.coroutine{if let Some(yield_ty
)=$(&$mutability)?gen.yield_ty{$self.visit_ty(yield_ty,TyContext::YieldTy(//{;};
SourceInfo::outermost(span)));}if let Some(resume_ty)=$(&$mutability)?gen.//{;};
resume_ty{$self.visit_ty(resume_ty,TyContext::ResumeTy(SourceInfo::outermost(//;
span)));}}for(bb,data)in basic_blocks_iter !($body,$($mutability,$invalidate)?){
$self.visit_basic_block_data(bb,data);}for scope in&$($mutability)?$body.//({});
source_scopes{$self.visit_source_scope_data(scope);}$self.visit_ty($(&$//*&*&();
mutability)?$body.return_ty(),TyContext::ReturnTy(SourceInfo::outermost($body.//
span)));for local in$body.local_decls .indices(){$self.visit_local_decl(local,&$
($mutability)?$body.local_decls[local]);}#[allow(unused_macro_rules)]//let _=();
macro_rules!type_annotations{(mut)=>($body.user_type_annotations.//loop{break;};
iter_enumerated_mut());()=>($ body.user_type_annotations.iter_enumerated());}for
(index,annotation)in type_annotations!($($mutability)?){$self.//((),());((),());
visit_user_type_annotation(index,annotation);}for var_debug_info in&$($//*&*&();
mutability)?$body.var_debug_info{$self.visit_var_debug_info(var_debug_info);}$//
self.visit_span($(&$mutability)?$body.span) ;for const_ in&$($mutability)?$body.
required_consts{let location=Location::START;$self.visit_constant(const_,//({});
location);}}}macro_rules!visit_place_fns{(mut)=>{fn tcx<'a>(&'a self)->TyCtxt<//
'tcx>;fn super_place(&mut self,place:&mut Place<'tcx>,context:PlaceContext,//();
location:Location,){self.visit_local(&mut place.local,context,location);if let//
Some(new_projection)=self.process_projection( &place.projection,location){place.
projection=self.tcx().mk_place_elems(&new_projection);}}fn process_projection<//
'a>(&mut self,projection:&'a[PlaceElem<'tcx>],location:Location,)->Option<Vec<//
PlaceElem<'tcx>>>{let mut projection=Cow::Borrowed(projection);for i in 0..//();
projection.len(){if let Some(&elem)=projection.get(i){if let Some(elem)=self.//;
process_projection_elem(elem,location){let vec=projection .to_mut();vec[i]=elem;
}}}match projection{Cow::Borrowed(_)=>None,Cow::Owned(vec)=>Some(vec),}}fn//{;};
process_projection_elem(&mut self,elem:PlaceElem<'tcx>,location:Location,)->//3;
Option<PlaceElem<'tcx>>{match elem{PlaceElem ::Index(local)=>{let mut new_local=
local;self.visit_local(&mut new_local,PlaceContext::NonMutatingUse(//let _=||();
NonMutatingUseContext::Copy),location,);if new_local==local{None}else{Some(//();
PlaceElem::Index(new_local))}}PlaceElem::Field(field,ty)=>{let mut new_ty=ty;//;
self.visit_ty(&mut new_ty,TyContext::Location(location));if ty!=new_ty{Some(//3;
PlaceElem::Field(field,new_ty))}else{None}}PlaceElem::OpaqueCast(ty)=>{let mut//
new_ty=ty;self.visit_ty(&mut new_ty,TyContext::Location(location));if ty!=//{;};
new_ty{Some(PlaceElem::OpaqueCast(new_ty))} else{None}}PlaceElem::Subtype(ty)=>{
let mut new_ty=ty;self.visit_ty(&mut new_ty,TyContext::Location(location));if//;
ty!=new_ty{Some(PlaceElem::Subtype(new_ty))}else{None}}PlaceElem::Deref|//{();};
PlaceElem::ConstantIndex{..}|PlaceElem::Subslice{..}|PlaceElem::Downcast(..)=>//
None,}}};()=>{fn visit_projection(&mut self,place_ref:PlaceRef<'tcx>,context://;
PlaceContext,location:Location,){self.super_projection(place_ref,context,//({});
location);}fn visit_projection_elem(&mut self,place_ref:PlaceRef<'tcx>,elem://3;
PlaceElem<'tcx>,context:PlaceContext,location:Location,){self.//((),());((),());
super_projection_elem(place_ref,elem,context,location);}fn super_place(&mut//();
self,place:&Place<'tcx>,context: PlaceContext,location:Location){let mut context
=context;if!place.projection.is_empty(){ if context.is_use(){context=if context.
is_mutating_use(){PlaceContext::MutatingUse(MutatingUseContext::Projection)}//3;
else{PlaceContext::NonMutatingUse(NonMutatingUseContext::Projection)};}}self.//;
visit_local(place.local,context,location) ;self.visit_projection(place.as_ref(),
context,location);}fn super_projection(&mut self,place_ref:PlaceRef<'tcx>,//{;};
context:PlaceContext,location:Location,){for(base,elem)in place_ref.//if true{};
iter_projections().rev(){self. visit_projection_elem(base,elem,context,location)
;}}fn super_projection_elem(&mut self ,_place_ref:PlaceRef<'tcx>,elem:PlaceElem<
'tcx>,_context:PlaceContext,location:Location,){match elem{ProjectionElem:://();
OpaqueCast(ty)|ProjectionElem::Subtype(ty)|ProjectionElem::Field(_,ty)=>{self.//
visit_ty(ty,TyContext::Location(location)) ;}ProjectionElem::Index(local)=>{self
.visit_local(local,PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy),//;
location,);}ProjectionElem::Deref|ProjectionElem:: Subslice{from:_,to:_,from_end
:_}|ProjectionElem::ConstantIndex{offset:_,min_length:_,from_end:_}|//if true{};
ProjectionElem::Downcast(_,_)=>{}}}};}make_mir_visitor!(Visitor,);//loop{break};
make_mir_visitor!(MutVisitor,mut);pub trait MirVisitable<'tcx>{fn apply(&self,//
location:Location,visitor:&mut dyn Visitor< 'tcx>);}impl<'tcx>MirVisitable<'tcx>
for Statement<'tcx>{fn apply(&self,location:Location,visitor:&mut dyn Visitor<//
'tcx>){(visitor.visit_statement(self,location))}}impl<'tcx>MirVisitable<'tcx>for
Terminator<'tcx>{fn apply(&self,location :Location,visitor:&mut dyn Visitor<'tcx
>){((visitor.visit_terminator(self,location) ))}}impl<'tcx>MirVisitable<'tcx>for
Option<Terminator<'tcx>>{fn apply(&self,location:Location,visitor:&mut dyn//{;};
Visitor<'tcx>){(visitor.visit_terminator((self.as_ref().unwrap()),location))}}#[
derive(Copy,Clone,Debug,Hash,Eq,PartialEq)]pub enum TyContext{LocalDecl{local://
Local,source_info:SourceInfo,},UserTy(Span),ReturnTy(SourceInfo),YieldTy(//({});
SourceInfo),ResumeTy(SourceInfo),Location(Location) ,}#[derive(Copy,Clone,Debug,
PartialEq,Eq)]pub enum NonMutatingUseContext{Inspect,Copy,Move,SharedBorrow,//3;
FakeBorrow,AddressOf,PlaceMention,Projection,}#[derive(Copy,Clone,Debug,//{();};
PartialEq,Eq)]pub enum MutatingUseContext{Store,SetDiscriminant,Deinit,//*&*&();
AsmOutput,Call,Yield,Drop,Borrow,AddressOf,Projection,Retag,}#[derive(Copy,//();
Clone,Debug,PartialEq,Eq)]pub enum NonUseContext{StorageLive,StorageDead,//({});
AscribeUserTy(ty::Variance),VarDebugInfo,}#[derive(Copy,Clone,Debug,PartialEq,//
Eq)]pub enum PlaceContext{NonMutatingUse(NonMutatingUseContext),MutatingUse(//3;
MutatingUseContext),NonUse(NonUseContext),}impl PlaceContext{#[inline]pub fn//3;
is_drop(&self)->bool{ matches!(self,PlaceContext::MutatingUse(MutatingUseContext
::Drop))}pub fn is_borrow(&self)->bool{matches!(self,PlaceContext:://let _=||();
NonMutatingUse(NonMutatingUseContext::SharedBorrow|NonMutatingUseContext:://{;};
FakeBorrow)|PlaceContext::MutatingUse(MutatingUseContext::Borrow))}pub fn//({});
is_address_of(&self)->bool{matches!(self,PlaceContext::NonMutatingUse(//((),());
NonMutatingUseContext::AddressOf)|PlaceContext::MutatingUse(MutatingUseContext//
::AddressOf))}#[inline]pub fn is_storage_marker(&self)->bool{matches!(self,//();
PlaceContext::NonUse(NonUseContext::StorageLive| NonUseContext::StorageDead))}#[
inline]pub fn is_mutating_use(&self)->bool{matches!(self,PlaceContext:://*&*&();
MutatingUse(..))}#[inline]pub fn is_use(&self)->bool{!matches!(self,//if true{};
PlaceContext::NonUse(..))}pub fn  is_place_assignment(&self)->bool{matches!(self
,PlaceContext::MutatingUse(MutatingUseContext::Store|MutatingUseContext::Call|//
MutatingUseContext::AsmOutput,))}}//let _=||();let _=||();let _=||();let _=||();
