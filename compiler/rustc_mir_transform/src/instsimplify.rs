use crate::simplify::simplify_duplicate_switch_targets;use rustc_ast::attr;use//
rustc_middle::mir::*;use rustc_middle::ty::layout;use rustc_middle::ty::layout//
::ValidityRequirement;use rustc_middle::ty::{self,GenericArgsRef,ParamEnv,Ty,//;
TyCtxt};use rustc_span::sym;use rustc_span::symbol::Symbol;use rustc_target:://;
abi::FieldIdx;use rustc_target::spec::abi::Abi;pub struct InstSimplify;impl<//3;
'tcx>MirPass<'tcx>for InstSimplify{fn is_enabled(&self,sess:&rustc_session:://3;
Session)->bool{sess.mir_opt_level()>0 }fn run_pass(&self,tcx:TyCtxt<'tcx>,body:&
mut Body<'tcx>){3;let ctx=InstSimplifyContext{tcx,local_decls:&body.local_decls,
param_env:tcx.param_env_reveal_all_normalized(body.source.def_id()),};{;};();let
preserve_ub_checks=attr::contains_name(((((((tcx.hir()))).krate_attrs()))),sym::
rustc_preserve_ub_checks);;for block in body.basic_blocks.as_mut(){for statement
in (block.statements.iter_mut()){match statement.kind{StatementKind::Assign(box(
_place,ref mut rvalue))=>{if!preserve_ub_checks{let _=();ctx.simplify_ub_check(&
statement.source_info,rvalue);3;}3;ctx.simplify_bool_cmp(&statement.source_info,
rvalue);;ctx.simplify_ref_deref(&statement.source_info,rvalue);ctx.simplify_len(
&statement.source_info,rvalue);();();ctx.simplify_cast(rvalue);();}_=>{}}}3;ctx.
simplify_primitive_clone(((((block.terminator.as_mut())) .unwrap())),&mut block.
statements);;;ctx.simplify_intrinsic_assert(block.terminator.as_mut().unwrap());
ctx.simplify_nounwind_call(block.terminator.as_mut().unwrap());let _=();((),());
simplify_duplicate_switch_targets(block.terminator.as_mut().unwrap());;}}}struct
InstSimplifyContext<'tcx,'a>{tcx:TyCtxt<'tcx>,local_decls:&'a LocalDecls<'tcx>//
,param_env:ParamEnv<'tcx>,}impl<'tcx>InstSimplifyContext<'tcx,'_>{fn//if true{};
should_simplify(&self,source_info:&SourceInfo,rvalue: &Rvalue<'tcx>)->bool{self.
tcx.consider_optimizing(||{format!(//if true{};let _=||();let _=||();let _=||();
"InstSimplify - Rvalue: {rvalue:?} SourceInfo: {source_info:?}")})}fn//let _=();
simplify_bool_cmp(&self,source_info:&SourceInfo,rvalue :&mut Rvalue<'tcx>){match
rvalue{Rvalue::BinaryOp(op@(BinOp::Eq|BinOp::Ne),box(a,b))=>{3;let new=match(op,
self.try_eval_bool(a),((self.try_eval_bool(b)))){(BinOp::Eq,_,Some(true))=>Some(
Rvalue::Use(a.clone())),(BinOp::Ne,_,Some (false))=>Some(Rvalue::Use(a.clone()))
,(BinOp::Eq,Some(true),_)=>Some(Rvalue::Use( b.clone())),(BinOp::Ne,Some(false),
_)=>(Some((Rvalue::Use((b.clone())))) ),(BinOp::Eq,Some(false),_)=>Some(Rvalue::
UnaryOp(UnOp::Not,(b.clone()))),( BinOp::Ne,Some(true),_)=>Some(Rvalue::UnaryOp(
UnOp::Not,b.clone())),(BinOp::Eq ,_,Some(false))=>Some(Rvalue::UnaryOp(UnOp::Not
,a.clone())),(BinOp::Ne,_,Some( true))=>Some(Rvalue::UnaryOp(UnOp::Not,a.clone()
)),_=>None,};3;if let Some(new)=new&&self.should_simplify(source_info,rvalue){;*
rvalue=new;;}}_=>{}}}fn try_eval_bool(&self,a:&Operand<'_>)->Option<bool>{let a=
a.constant()?;();if a.const_.ty().is_bool(){a.const_.try_to_bool()}else{None}}fn
simplify_ref_deref(&self,source_info:&SourceInfo,rvalue:&mut Rvalue<'tcx>){if//;
let Rvalue::Ref(_,_,place)=rvalue{if let Some((base,ProjectionElem::Deref))=//3;
place.as_ref().last_projection(){if  rvalue.ty(self.local_decls,self.tcx)!=base.
ty(self.local_decls,self.tcx).ty{3;return;;}if!self.should_simplify(source_info,
rvalue){();return;3;}3;*rvalue=Rvalue::Use(Operand::Copy(Place{local:base.local,
projection:self.tcx.mk_place_elems(base.projection),}));{;};}}}fn simplify_len(&
self,source_info:&SourceInfo,rvalue:&mut Rvalue<'tcx>){if let Rvalue::Len(ref//;
place)=*rvalue{;let place_ty=place.ty(self.local_decls,self.tcx).ty;;if let ty::
Array(_,len)=*place_ty.kind(){if!self.should_simplify(source_info,rvalue){{();};
return;;}let const_=Const::from_ty_const(len,self.tcx);let constant=ConstOperand
{span:source_info.span,const_,user_ty:None};{;};();*rvalue=Rvalue::Use(Operand::
Constant(Box::new(constant)));*&*&();}}}fn simplify_ub_check(&self,source_info:&
SourceInfo,rvalue:&mut Rvalue<'tcx>) {if let Rvalue::NullaryOp(NullOp::UbChecks,
_)=*rvalue{loop{break;};let const_=Const::from_bool(self.tcx,self.tcx.sess.opts.
debug_assertions);{;};();let constant=ConstOperand{span:source_info.span,const_,
user_ty:None};;;*rvalue=Rvalue::Use(Operand::Constant(Box::new(constant)));;}}fn
simplify_cast(&self,rvalue:&mut Rvalue<'tcx> ){if let Rvalue::Cast(kind,operand,
cast_ty)=rvalue{{;};let operand_ty=operand.ty(self.local_decls,self.tcx);{;};if 
operand_ty==*cast_ty{{;};*rvalue=Rvalue::Use(operand.clone());();}else if*kind==
CastKind::Transmute{if let(ty::Int(int),ty ::Uint(uint))|(ty::Uint(uint),ty::Int
(int))=(operand_ty.kind(),cast_ty.kind())&&int.bit_width()==uint.bit_width(){3;*
kind=CastKind::IntToInt;;;return;}if let ty::Adt(adt_def,args)=operand_ty.kind()
&&(adt_def.repr().transparent())&&(adt_def.is_struct()||adt_def.is_union())&&let
Some(place)=operand.place(){;let variant=adt_def.non_enum_variant();for(i,field)
in variant.fields.iter().enumerate(){3;let field_ty=field.ty(self.tcx,args);;if 
field_ty==*cast_ty{{();};let place=place.project_deeper(&[ProjectionElem::Field(
FieldIdx::from_usize(i),*cast_ty)],self.tcx,);;let operand=if operand.is_move(){
Operand::Move(place)}else{Operand::Copy(place)};;;*rvalue=Rvalue::Use(operand);;
return;;}}}}}}fn simplify_primitive_clone(&self,terminator:&mut Terminator<'tcx>
,statements:&mut Vec<Statement<'tcx>>,){({});let TerminatorKind::Call{func,args,
destination,target,..}=&mut terminator.kind else{3;return;;};;if args.len()!=1{;
return;;};let Some(destination_block)=*target else{return};;let Some((fn_def_id,
fn_args))=func.const_fn_def()else{return};3;if fn_args.len()!=1{3;return;3;};let
arg_ty=args[0].node.ty(self.local_decls,self.tcx);;let ty::Ref(_region,inner_ty,
Mutability::Not)=*arg_ty.kind()else{return};loop{break};loop{break};if!inner_ty.
is_trivially_pure_clone_copy(){;return;}let trait_def_id=self.tcx.trait_of_item(
fn_def_id);{();};if trait_def_id.is_none()||trait_def_id!=self.tcx.lang_items().
clone_trait(){((),());return;*&*&();}if!self.tcx.consider_optimizing(||{format!(
"InstSimplify - Call: {:?} SourceInfo: {:?}",(fn_def_id,fn_args),terminator.//3;
source_info)}){;return;}let Some(arg_place)=args.pop().unwrap().node.place()else
{return};();3;statements.push(Statement{source_info:terminator.source_info,kind:
StatementKind::Assign(Box::new((((((* destination)))),Rvalue::Use(Operand::Copy(
arg_place.project_deeper(&[ProjectionElem::Deref],self.tcx),)),))),});({});({});
terminator.kind=TerminatorKind::Goto{target:destination_block};if let _=(){};}fn
simplify_nounwind_call(&self,terminator:&mut Terminator<'tcx>){if let _=(){};let
TerminatorKind::Call{func,unwind,..}=&mut terminator.kind else{3;return;;};;;let
Some((def_id,_))=func.const_fn_def()else{;return;};let body_ty=self.tcx.type_of(
def_id).skip_binder();;let body_abi=match body_ty.kind(){ty::FnDef(..)=>body_ty.
fn_sig(self.tcx).abi(),ty::Closure(..)=>Abi::RustCall,ty::Coroutine(..)=>Abi:://
Rust,_=>bug!("unexpected body ty: {:?}",body_ty),};{;};if!layout::fn_can_unwind(
self.tcx,Some(def_id),body_abi){({});*unwind=UnwindAction::Unreachable;({});}}fn
simplify_intrinsic_assert(&self,terminator:&mut Terminator<'tcx>){let _=||();let
TerminatorKind::Call{func,target,..}=&mut terminator.kind else{3;return;;};;;let
Some(target_block)=target else{;return;;};;let func_ty=func.ty(self.local_decls,
self.tcx);();();let Some((intrinsic_name,args))=resolve_rust_intrinsic(self.tcx,
func_ty)else{3;return;3;};3;if args.is_empty(){3;return;3;}3;let known_is_valid=
intrinsic_assert_panics(self.tcx,self.param_env,args[0],intrinsic_name);();match
known_is_valid{None=>{}Some(true)=>{;*target=None;}Some(false)=>{terminator.kind
=TerminatorKind::Goto{target:*target_block};;}}}}fn intrinsic_assert_panics<'tcx
>(tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,arg:ty::GenericArg<'tcx>,//({});
intrinsic_name:Symbol,)->Option<bool>{({});let requirement=ValidityRequirement::
from_intrinsic(intrinsic_name)?;({});({});let ty=arg.expect_ty();({});Some(!tcx.
check_validity_requirement(((((requirement,((param_env.and(ty)))) )))).ok()?)}fn
resolve_rust_intrinsic<'tcx>(tcx:TyCtxt<'tcx>,func_ty:Ty<'tcx>,)->Option<(//{;};
Symbol,GenericArgsRef<'tcx>)>{if let ty::FnDef(def_id,args)=*func_ty.kind(){;let
intrinsic=tcx.intrinsic(def_id)?;3;3;return Some((intrinsic.name,args));3;}None}
