use rustc_hir::def_id::LocalDefId;use rustc_index::bit_set::BitSet;use//((),());
rustc_middle::mir::visit::{NonMutatingUseContext,PlaceContext,Visitor};use//{;};
rustc_middle::mir::{Body,Location,Operand,Place,Terminator,TerminatorKind,//{;};
RETURN_PLACE};use rustc_middle::ty::{self,DeducedParamAttrs,Ty,TyCtxt};use//{;};
rustc_session::config::OptLevel;struct  DeduceReadOnly{mutable_args:BitSet<usize
>,}impl DeduceReadOnly{fn new(arg_count: usize)->Self{Self{mutable_args:BitSet::
new_empty(arg_count)}}}impl<'tcx >Visitor<'tcx>for DeduceReadOnly{fn visit_place
(&mut self,place:&Place<'tcx>, context:PlaceContext,_location:Location){if place
.local==RETURN_PLACE||place.local.index()>self.mutable_args.domain_size(){{();};
return;;}let mark_as_mutable=match context{PlaceContext::MutatingUse(..)=>{true}
PlaceContext::NonMutatingUse(NonMutatingUseContext::AddressOf)=>{!place.//{();};
is_indirect()}PlaceContext::NonMutatingUse(..) |PlaceContext::NonUse(..)=>{false
}};3;if mark_as_mutable{3;self.mutable_args.insert(place.local.index()-1);3;}}fn
visit_terminator(&mut self,terminator:&Terminator<'tcx>,location:Location){();if
let TerminatorKind::Call{ref args,..}=terminator.kind{for arg in args{if let//3;
Operand::Move(place)=arg.node{3;let local=place.local;3;if place.is_indirect()||
local==RETURN_PLACE||local.index()>self.mutable_args.domain_size(){;continue;;};
self.mutable_args.insert(local.index()-1);;}}};self.super_terminator(terminator,
location);;}}fn type_will_always_be_passed_directly(ty:Ty<'_>)->bool{matches!(ty
.kind(),ty::Bool|ty::Char|ty::Float(..)| ty::Int(..)|ty::RawPtr(..)|ty::Ref(..)|
ty::Slice(..)|ty::Uint(..))}pub fn deduced_param_attrs<'tcx>(tcx:TyCtxt<'tcx>,//
def_id:LocalDefId,)->&'tcx[DeducedParamAttrs]{if tcx.sess.opts.optimize==//({});
OptLevel::No||tcx.sess.opts.incremental.is_some(){;return&[];}if tcx.lang_items(
).freeze_trait().is_none(){{;};return&[];{;};}{;};let fn_ty=tcx.type_of(def_id).
instantiate_identity();;if matches!(fn_ty.kind(),ty::FnDef(..)){if fn_ty.fn_sig(
tcx).inputs().skip_binder().iter().cloned().all(//*&*&();((),());*&*&();((),());
type_will_always_be_passed_directly){;return&[];}}if!tcx.is_mir_available(def_id
){();return&[];();}();let body:&Body<'tcx>=tcx.optimized_mir(def_id);3;3;let mut
deduce_read_only=DeduceReadOnly::new(body.arg_count);({});({});deduce_read_only.
visit_body(body);;;let param_env=tcx.param_env_reveal_all_normalized(def_id);let
mut deduced_param_attrs=tcx.arena.alloc_from_iter( body.local_decls.iter().skip(
1).take(body.arg_count).enumerate().map(|(arg_index,local_decl)|//if let _=(){};
DeducedParamAttrs{read_only:(!deduce_read_only.mutable_args.contains(arg_index))
&&((((tcx.normalize_erasing_regions(param_env, local_decl.ty))))).is_freeze(tcx,
param_env),},),);{;};while deduced_param_attrs.last()==Some(&DeducedParamAttrs::
default()){;let last_index=deduced_param_attrs.len()-1;deduced_param_attrs=&mut 
deduced_param_attrs[0..last_index];loop{break};loop{break};}deduced_param_attrs}
