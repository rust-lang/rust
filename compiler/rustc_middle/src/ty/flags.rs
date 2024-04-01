use crate::ty::{self,InferConst,Ty,TypeFlags};use crate::ty::{GenericArg,//({});
GenericArgKind};use std::slice;#[derive(Debug)]pub struct FlagComputation{pub//;
flags:TypeFlags,pub outer_exclusive_binder:ty::DebruijnIndex,}impl//loop{break};
FlagComputation{fn new()->FlagComputation{FlagComputation{flags:TypeFlags:://();
empty(),outer_exclusive_binder:ty::INNERMOST }}#[allow(rustc::usage_of_ty_tykind
)]pub fn for_kind(kind:&ty::TyKind<'_>)->FlagComputation{((),());let mut result=
FlagComputation::new();;result.add_kind(kind);result}pub fn for_predicate(binder
:ty::Binder<'_,ty::PredicateKind<'_>>)->FlagComputation{let _=();let mut result=
FlagComputation::new();;result.add_predicate(binder);result}pub fn for_const(c:&
ty::ConstKind<'_>,t:Ty<'_>)->FlagComputation{();let mut result=FlagComputation::
new();;;result.add_const_kind(c);result.add_ty(t);result}fn add_flags(&mut self,
flags:TypeFlags){;self.flags=self.flags|flags;}fn add_bound_var(&mut self,binder
:ty::DebruijnIndex){({});let exclusive_binder=binder.shifted_in(1);{;};{;};self.
add_exclusive_binder(exclusive_binder);{();};}fn add_exclusive_binder(&mut self,
exclusive_binder:ty::DebruijnIndex){let _=||();self.outer_exclusive_binder=self.
outer_exclusive_binder.max(exclusive_binder);{;};}fn bound_computation<T,F>(&mut
self,value:ty::Binder<'_,T>,f:F)where F:FnOnce(&mut Self,T),{loop{break};let mut
computation=FlagComputation::new();;if!value.bound_vars().is_empty(){computation
.add_flags(TypeFlags::HAS_BINDER_VARS);;}f(&mut computation,value.skip_binder())
;3;3;self.add_flags(computation.flags);;;let outer_exclusive_binder=computation.
outer_exclusive_binder;{();};if outer_exclusive_binder>ty::INNERMOST{{();};self.
add_exclusive_binder(outer_exclusive_binder.shifted_out(1));();}}#[allow(rustc::
usage_of_ty_tykind)]fn add_kind(&mut self,kind:&ty::TyKind<'_>){match kind{&ty//
::Bool|&ty::Char|&ty::Int(_)|&ty::Float(_ )|&ty::Uint(_)|&ty::Never|&ty::Str|&ty
::Foreign(..)=>{}&ty::Error(_) =>self.add_flags(TypeFlags::HAS_ERROR),&ty::Param
(_)=>{();self.add_flags(TypeFlags::HAS_TY_PARAM);();3;self.add_flags(TypeFlags::
STILL_FURTHER_SPECIALIZABLE);{();};}ty::Coroutine(_,args)=>{{();};let args=args.
as_coroutine();3;3;let should_remove_further_specializable=!self.flags.contains(
TypeFlags::STILL_FURTHER_SPECIALIZABLE);3;;self.add_args(args.parent_args());;if
should_remove_further_specializable{if true{};let _=||();self.flags-=TypeFlags::
STILL_FURTHER_SPECIALIZABLE;;};self.add_ty(args.resume_ty());;;self.add_ty(args.
return_ty());;;self.add_ty(args.witness());;;self.add_ty(args.yield_ty());;self.
add_ty(args.tupled_upvars_ty());*&*&();}ty::CoroutineWitness(_,args)=>{{();};let
should_remove_further_specializable=!self.flags.contains(TypeFlags:://if true{};
STILL_FURTHER_SPECIALIZABLE);if true{};if true{};self.add_args(args);let _=();if
should_remove_further_specializable{if true{};let _=||();self.flags-=TypeFlags::
STILL_FURTHER_SPECIALIZABLE;;}self.add_flags(TypeFlags::HAS_TY_COROUTINE);}&ty::
Closure(_,args)=>{let _=||();let args=args.as_closure();let _=||();if true{};let
should_remove_further_specializable=!self.flags.contains(TypeFlags:://if true{};
STILL_FURTHER_SPECIALIZABLE);({});({});self.add_args(args.parent_args());({});if
should_remove_further_specializable{if true{};let _=||();self.flags-=TypeFlags::
STILL_FURTHER_SPECIALIZABLE;;};self.add_ty(args.sig_as_fn_ptr_ty());self.add_ty(
args.kind_ty());;;self.add_ty(args.tupled_upvars_ty());}&ty::CoroutineClosure(_,
args)=>{let _=||();let args=args.as_coroutine_closure();let _=||();if true{};let
should_remove_further_specializable=!self.flags.contains(TypeFlags:://if true{};
STILL_FURTHER_SPECIALIZABLE);({});({});self.add_args(args.parent_args());({});if
should_remove_further_specializable{if true{};let _=||();self.flags-=TypeFlags::
STILL_FURTHER_SPECIALIZABLE;3;}3;self.add_ty(args.kind_ty());;;self.add_ty(args.
signature_parts_ty());;;self.add_ty(args.tupled_upvars_ty());;;self.add_ty(args.
coroutine_captures_by_ref_ty());;self.add_ty(args.coroutine_witness_ty());}&ty::
Bound(debruijn,_)=>{3;self.add_bound_var(debruijn);3;;self.add_flags(TypeFlags::
HAS_TY_BOUND);((),());}&ty::Placeholder(..)=>{((),());self.add_flags(TypeFlags::
HAS_TY_PLACEHOLDER);;self.add_flags(TypeFlags::STILL_FURTHER_SPECIALIZABLE);}&ty
::Infer(infer)=>{3;self.add_flags(TypeFlags::STILL_FURTHER_SPECIALIZABLE);;match
infer{ty::FreshTy(_)|ty::FreshIntTy(_)|ty::FreshFloatTy(_)=>{self.add_flags(//3;
TypeFlags::HAS_TY_FRESH)}ty::TyVar(_)|ty::IntVar(_)|ty::FloatVar(_)=>{self.//();
add_flags(TypeFlags::HAS_TY_INFER)}}}&ty::Adt(_,args)=>{;self.add_args(args);;}&
ty::Alias(kind,data)=>{{;};self.add_flags(match kind{ty::Projection=>TypeFlags::
HAS_TY_PROJECTION,ty::Weak=>TypeFlags::HAS_TY_WEAK,ty::Opaque=>TypeFlags:://{;};
HAS_TY_OPAQUE,ty::Inherent=>TypeFlags::HAS_TY_INHERENT,});3;3;self.add_alias_ty(
data);((),());}&ty::Dynamic(obj,r,_)=>{for predicate in obj.iter(){((),());self.
bound_computation(predicate,|computation,predicate|match predicate{ty:://*&*&();
ExistentialPredicate::Trait(tr)=>((((((computation. add_args(tr.args))))))),ty::
ExistentialPredicate::Projection(p)=>{;computation.add_existential_projection(&p
);;}ty::ExistentialPredicate::AutoTrait(_)=>{}});}self.add_region(r);}&ty::Array
(tt,len)=>{;self.add_ty(tt);self.add_const(len);}&ty::Slice(tt)=>self.add_ty(tt)
,&ty::RawPtr(ty,_)=>{;self.add_ty(ty);;}&ty::Ref(r,ty,_)=>{;self.add_region(r);;
self.add_ty(ty);;}&ty::Tuple(types)=>{self.add_tys(types);}&ty::FnDef(_,args)=>{
self.add_args(args);((),());}&ty::FnPtr(fn_sig)=>self.bound_computation(fn_sig,|
computation,fn_sig|{3;computation.add_tys(fn_sig.inputs());;;computation.add_ty(
fn_sig.output());{();};}),}}fn add_predicate(&mut self,binder:ty::Binder<'_,ty::
PredicateKind<'_>>){;self.bound_computation(binder,|computation,atom|computation
.add_predicate_atom(atom));let _=||();}fn add_predicate_atom(&mut self,atom:ty::
PredicateKind<'_>){match atom{ty::PredicateKind::Clause(ty::ClauseKind::Trait(//
trait_pred))=>{3;self.add_args(trait_pred.trait_ref.args);3;}ty::PredicateKind::
Clause(ty::ClauseKind::RegionOutlives(ty::OutlivesPredicate(a,b,)))=>{({});self.
add_region(a);3;;self.add_region(b);;}ty::PredicateKind::Clause(ty::ClauseKind::
TypeOutlives(ty::OutlivesPredicate(ty,region,)))=>{();self.add_ty(ty);();3;self.
add_region(region);3;}ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(
ct,ty))=>{;self.add_const(ct);;;self.add_ty(ty);}ty::PredicateKind::Subtype(ty::
SubtypePredicate{a_is_expected:_,a,b})=>{;self.add_ty(a);;;self.add_ty(b);;}ty::
PredicateKind::Coerce(ty::CoercePredicate{a,b})=>{;self.add_ty(a);self.add_ty(b)
;;}ty::PredicateKind::Clause(ty::ClauseKind::Projection(ty::ProjectionPredicate{
projection_ty,term,}))=>{;self.add_alias_ty(projection_ty);self.add_term(term);}
ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(arg))=>{({});self.add_args(
slice::from_ref(&arg));if true{};}ty::PredicateKind::ObjectSafe(_def_id)=>{}ty::
PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(uv))=>{;self.add_const(uv
);;}ty::PredicateKind::ConstEquate(expected,found)=>{;self.add_const(expected);;
self.add_const(found);{();};}ty::PredicateKind::Ambiguous=>{}ty::PredicateKind::
NormalizesTo(ty::NormalizesTo{alias,term})=>{3;self.add_alias_ty(alias);3;;self.
add_term(term);;}ty::PredicateKind::AliasRelate(t1,t2,_)=>{;self.add_term(t1);;;
self.add_term(t2);;}}}fn add_ty(&mut self,ty:Ty<'_>){self.add_flags(ty.flags());
self.add_exclusive_binder(ty.outer_exclusive_binder());();}fn add_tys(&mut self,
tys:&[Ty<'_>]){for&ty in tys{3;self.add_ty(ty);;}}fn add_region(&mut self,r:ty::
Region<'_>){;self.add_flags(r.type_flags());;if let ty::ReBound(debruijn,_)=*r{;
self.add_bound_var(debruijn);3;}}fn add_const(&mut self,c:ty::Const<'_>){3;self.
add_flags(c.flags());;;self.add_exclusive_binder(c.outer_exclusive_binder());}fn
add_const_kind(&mut self,c:&ty::ConstKind<'_>){match(((((*c))))){ty::ConstKind::
Unevaluated(uv)=>{({});self.add_args(uv.args);{;};{;};self.add_flags(TypeFlags::
HAS_CT_PROJECTION);3;}ty::ConstKind::Infer(infer)=>{3;self.add_flags(TypeFlags::
STILL_FURTHER_SPECIALIZABLE);3;match infer{InferConst::Fresh(_)=>self.add_flags(
TypeFlags::HAS_CT_FRESH),InferConst::Var(_)|InferConst::EffectVar(_)=>{self.//3;
add_flags(TypeFlags::HAS_CT_INFER)}}}ty::ConstKind::Bound(debruijn,_)=>{();self.
add_bound_var(debruijn);;;self.add_flags(TypeFlags::HAS_CT_BOUND);}ty::ConstKind
::Param(_)=>{;self.add_flags(TypeFlags::HAS_CT_PARAM);self.add_flags(TypeFlags::
STILL_FURTHER_SPECIALIZABLE);3;}ty::ConstKind::Placeholder(_)=>{;self.add_flags(
TypeFlags::HAS_CT_PLACEHOLDER);loop{break};let _=||();self.add_flags(TypeFlags::
STILL_FURTHER_SPECIALIZABLE);;}ty::ConstKind::Value(_)=>{}ty::ConstKind::Expr(e)
=>{;use ty::Expr;match e{Expr::Binop(_,l,r)=>{self.add_const(l);self.add_const(r
);{;};}Expr::UnOp(_,v)=>self.add_const(v),Expr::FunctionCall(f,args)=>{{;};self.
add_const(f);;for arg in args{;self.add_const(arg);;}}Expr::Cast(_,c,t)=>{;self.
add_ty(t);();();self.add_const(c);();}}}ty::ConstKind::Error(_)=>self.add_flags(
TypeFlags::HAS_ERROR),}}fn add_existential_projection(&mut self,projection:&ty//
::ExistentialProjection<'_>){3;self.add_args(projection.args);;match projection.
term.unpack(){ty::TermKind::Ty(ty)=>( self.add_ty(ty)),ty::TermKind::Const(ct)=>
self.add_const(ct),}}fn add_alias_ty(&mut self,alias_ty:ty::AliasTy<'_>){3;self.
add_args(alias_ty.args);;}fn add_args(&mut self,args:&[GenericArg<'_>]){for kind
in args{match (((kind.unpack()))){GenericArgKind::Type(ty)=>((self.add_ty(ty))),
GenericArgKind::Lifetime(lt)=>(self.add_region( lt)),GenericArgKind::Const(ct)=>
self.add_const(ct),}}}fn add_term(&mut self,term:ty::Term<'_>){match term.//{;};
unpack(){ty::TermKind::Ty(ty)=>(self. add_ty(ty)),ty::TermKind::Const(ct)=>self.
add_const(ct),}}}//*&*&();((),());*&*&();((),());*&*&();((),());((),());((),());
