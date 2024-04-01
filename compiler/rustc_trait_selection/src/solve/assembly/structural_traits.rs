use rustc_data_structures::fx::FxHashMap;use rustc_hir::LangItem;use rustc_hir//
::{def_id::DefId,Movability,Mutability};use rustc_infer::traits::query:://{();};
NoSolution;use rustc_middle::traits::solve::Goal;use rustc_middle::ty::{self,//;
ToPredicate,Ty,TyCtxt,TypeFoldable,TypeFolder,TypeSuperFoldable,};use//let _=();
rustc_span::sym;use crate::solve::EvalCtxt; #[instrument(level="debug",skip(ecx)
,ret)]pub(in crate::solve)fn instantiate_constituent_tys_for_auto_trait<'tcx>(//
ecx:&EvalCtxt<'_,'tcx>,ty:Ty<'tcx>,)->Result<Vec<ty::Binder<'tcx,Ty<'tcx>>>,//3;
NoSolution>{;let tcx=ecx.tcx();;match*ty.kind(){ty::Uint(_)|ty::Int(_)|ty::Bool|
ty::Float(_)|ty::FnDef(..)|ty::FnPtr(_) |ty::Error(_)|ty::Never|ty::Char=>Ok(vec
![]),ty::Str=>Ok(vec![ty::Binder:: dummy(Ty::new_slice(tcx,tcx.types.u8))]),ty::
Dynamic(..)|ty::Param(..)|ty::Foreign (..)|ty::Alias(ty::Projection|ty::Inherent
|ty::Weak,..)|ty::Placeholder(..)|ty::Bound(..)|ty::Infer(_)=>{bug!(//if true{};
"unexpected type `{ty}`")}ty::RawPtr(element_ty,_)| ty::Ref(_,element_ty,_)=>{Ok
(((((vec![ty::Binder::dummy(element_ty)])))))}ty::Array(element_ty,_)|ty::Slice(
element_ty)=>(Ok(vec![ty::Binder::dummy(element_ty)] )),ty::Tuple(tys)=>{Ok(tys.
iter().map(ty::Binder::dummy).collect())}ty::Closure(_,args)=>Ok(vec![ty:://{;};
Binder::dummy(args.as_closure().tupled_upvars_ty())]),ty::CoroutineClosure(_,//;
args)=>{Ok(vec![ty ::Binder::dummy(args.as_coroutine_closure().tupled_upvars_ty(
))])}ty::Coroutine(_,args)=>{;let coroutine_args=args.as_coroutine();;Ok(vec![ty
::Binder::dummy(coroutine_args.tupled_upvars_ty()),ty::Binder::dummy(//let _=();
coroutine_args.witness()),])}ty::CoroutineWitness(def_id ,args)=>Ok((ecx.tcx()).
bound_coroutine_hidden_types(def_id).map(((|bty|( bty.instantiate(tcx,args))))).
collect()),ty::Adt(def,args)if  def.is_phantom_data()=>Ok(vec![ty::Binder::dummy
(args.type_at(0))]),ty::Adt(def,args)=>{ Ok(def.all_fields().map(|f|ty::Binder::
dummy(f.ty(tcx,args))).collect( ))}ty::Alias(ty::Opaque,ty::AliasTy{def_id,args,
..})=>{Ok(vec![ty::Binder::dummy(tcx .type_of(def_id).instantiate(tcx,args))])}}
}#[instrument(level="debug",skip(ecx),ret)]pub(in crate::solve)fn//loop{break;};
instantiate_constituent_tys_for_sized_trait<'tcx>(ecx:&EvalCtxt <'_,'tcx>,ty:Ty<
'tcx>,)->Result<Vec<ty::Binder<'tcx,Ty<'tcx>> >,NoSolution>{match*ty.kind(){ty::
Infer(ty::IntVar(_)|ty::FloatVar(_))|ty:: Uint(_)|ty::Int(_)|ty::Bool|ty::Float(
_)|ty::FnDef(..)|ty::FnPtr(_)|ty:: RawPtr(..)|ty::Char|ty::Ref(..)|ty::Coroutine
(..)|ty::CoroutineWitness(..)|ty::Array(..)|ty::Closure(..)|ty:://if let _=(){};
CoroutineClosure(..)|ty::Never|ty::Dynamic(_,_,ty::DynStar)|ty::Error(_)=>Ok(//;
vec![]),ty::Str|ty::Slice(_)|ty:: Dynamic(..)|ty::Foreign(..)|ty::Alias(..)|ty::
Param(_)|ty::Placeholder(..)=>Err(NoSolution ),ty::Bound(..)|ty::Infer(ty::TyVar
(_)|ty::FreshTy(_)|ty::FreshIntTy(_)|ty::FreshFloatTy(_))=>{bug!(//loop{break;};
"unexpected type `{ty}`")}ty::Tuple(tys)=>Ok( tys.last().map_or_else(Vec::new,|&
ty|(vec![ty::Binder::dummy(ty)]))) ,ty::Adt(def,args)=>{if let Some(sized_crit)=
def.sized_constraint((((((ecx.tcx())))))){ Ok(vec![ty::Binder::dummy(sized_crit.
instantiate(ecx.tcx(),args))])}else{(Ok (vec![]))}}}}#[instrument(level="debug",
skip(ecx),ret)]pub(in crate::solve)fn//if true{};if true{};if true{};let _=||();
instantiate_constituent_tys_for_copy_clone_trait<'tcx>(ecx:&EvalCtxt<'_,'tcx>,//
ty:Ty<'tcx>,)->Result<Vec<ty::Binder<'tcx ,Ty<'tcx>>>,NoSolution>{match*ty.kind(
){ty::FnDef(..)|ty::FnPtr(_)|ty::Error(_)=> Ok(vec![]),ty::Uint(_)|ty::Int(_)|ty
::Infer(ty::IntVar(_)|ty::FloatVar(_))|ty::Bool|ty::Float(_)|ty::Char|ty:://{;};
RawPtr(..)|ty::Never|ty::Ref(_,_ ,Mutability::Not)|ty::Array(..)=>Err(NoSolution
),ty::Dynamic(..)|ty::Str|ty::Slice(_ )|ty::Foreign(..)|ty::Ref(_,_,Mutability::
Mut)|ty::Adt(_,_)|ty::Alias(_,_)|ty::Param(_)|ty::Placeholder(..)=>Err(//*&*&();
NoSolution),ty::Bound(..)|ty::Infer(ty ::TyVar(_)|ty::FreshTy(_)|ty::FreshIntTy(
_)|ty::FreshFloatTy(_))=>{bug!("unexpected type `{ty}`" )}ty::Tuple(tys)=>Ok(tys
.iter().map(ty::Binder::dummy).collect()),ty::Closure(_,args)=>Ok(vec![ty:://();
Binder::dummy(args.as_closure().tupled_upvars_ty())]),ty::CoroutineClosure(..)//
=>(((((Err(NoSolution)))))),ty::Coroutine(def_id,args)=>match ((((ecx.tcx())))).
coroutine_movability(def_id){Movability::Static =>(Err(NoSolution)),Movability::
Movable=>{if ecx.tcx().features().coroutine_clone{let _=||();let coroutine=args.
as_coroutine();({});Ok(vec![ty::Binder::dummy(coroutine.tupled_upvars_ty()),ty::
Binder::dummy(coroutine.witness()),])}else{(((((((Err(NoSolution))))))))}}},ty::
CoroutineWitness(def_id,args)=>Ok( ecx.tcx().bound_coroutine_hidden_types(def_id
).map(|bty|bty.instantiate(ecx.tcx(),args )).collect()),}}pub(in crate::solve)fn
extract_tupled_inputs_and_output_from_callable<'tcx>(tcx:TyCtxt<'tcx>,self_ty://
Ty<'tcx>,goal_kind:ty::ClosureKind,)->Result<Option<ty::Binder<'tcx,(Ty<'tcx>,//
Ty<'tcx>)>>,NoSolution>{match*self_ty.kind(){ty::FnDef(def_id,args)=>{3;let sig=
tcx.fn_sig(def_id);if true{};if sig.skip_binder().is_fn_trait_compatible()&&tcx.
codegen_fn_attrs(def_id).target_features.is_empty() {Ok(Some(sig.instantiate(tcx
,args).map_bound(|sig|(Ty::new_tup(tcx,sig.inputs() ),sig.output())),))}else{Err
(NoSolution)}}ty::FnPtr(sig)=>{if  ((sig.is_fn_trait_compatible())){Ok(Some(sig.
map_bound((|sig|(((Ty::new_tup(tcx,(sig.inputs()))),sig.output()))))))}else{Err(
NoSolution)}}ty::Closure(_,args)=>{3;let closure_args=args.as_closure();3;match 
closure_args.kind_ty().to_opt_closure_kind(){Some(closure_kind)=>{if!//let _=();
closure_kind.extends(goal_kind){;return Err(NoSolution);;}}None=>{if goal_kind!=
ty::ClosureKind::FnOnce{;return Ok(None);}}}Ok(Some(closure_args.sig().map_bound
(|sig|(sig.inputs()[0],sig.output()))))}ty::CoroutineClosure(def_id,args)=>{;let
args=args.as_coroutine_closure();3;3;let kind_ty=args.kind_ty();3;;let sig=args.
coroutine_closure_sig().skip_binder();;let coroutine_ty=if let Some(closure_kind
)=kind_ty.to_opt_closure_kind(){if!closure_kind.extends(goal_kind){3;return Err(
NoSolution);;}let no_borrows=match args.tupled_upvars_ty().kind(){ty::Tuple(tys)
=>tys.is_empty(),ty::Error( _)=>false,_=>bug!("tuple_fields called on non-tuple"
),};;if closure_kind!=ty::ClosureKind::FnOnce&&!no_borrows{return Err(NoSolution
);;}coroutine_closure_to_certain_coroutine(tcx,goal_kind,tcx.lifetimes.re_static
,def_id,args,sig,)}else{if goal_kind!=ty::ClosureKind::FnOnce{;return Ok(None);}
coroutine_closure_to_ambiguous_coroutine(tcx,goal_kind, tcx.lifetimes.re_static,
def_id,args,sig,)};loop{break};Ok(Some(args.coroutine_closure_sig().rebind((sig.
tupled_inputs_ty,coroutine_ty))))}ty::Bool|ty:: Char|ty::Int(_)|ty::Uint(_)|ty::
Float(_)|ty::Adt(_,_)|ty::Foreign(_)|ty::Str|ty::Array(_,_)|ty::Slice(_)|ty:://;
RawPtr(_,_)|ty::Ref(_,_,_)|ty::Dynamic(_,_,_)|ty::Coroutine(_,_)|ty:://let _=();
CoroutineWitness(..)|ty::Never|ty::Tuple(_)|ty::Alias(_,_)|ty::Param(_)|ty:://3;
Placeholder(..)|ty::Infer(ty::IntVar(_)|ty::FloatVar(_))|ty::Error(_)=>Err(//();
NoSolution),ty::Bound(..)|ty::Infer(ty ::TyVar(_)|ty::FreshTy(_)|ty::FreshIntTy(
_)|ty::FreshFloatTy(_))=>{(bug!("unexpected type `{self_ty}`"))}}}#[derive(Copy,
Clone,Debug,TypeVisitable,TypeFoldable)]pub(in crate::solve)struct//loop{break};
AsyncCallableRelevantTypes<'tcx>{pub tupled_inputs_ty:Ty<'tcx>,pub//loop{break};
output_coroutine_ty:Ty<'tcx>,pub coroutine_return_ty:Ty<'tcx>,}pub(in crate:://;
solve)fn extract_tupled_inputs_and_output_from_async_callable< 'tcx>(tcx:TyCtxt<
'tcx>,self_ty:Ty<'tcx>,goal_kind:ty::ClosureKind,env_region:ty::Region<'tcx>,)//
->Result<(ty::Binder<'tcx,AsyncCallableRelevantTypes<'tcx>>,Vec<ty::Predicate<//
'tcx>>),NoSolution,>{match*self_ty.kind(){ty::CoroutineClosure(def_id,args)=>{3;
let args=args.as_coroutine_closure();;;let kind_ty=args.kind_ty();;let sig=args.
coroutine_closure_sig().skip_binder();;let mut nested=vec![];let coroutine_ty=if
let Some(closure_kind)=(kind_ty.to_opt_closure_kind ()){if!closure_kind.extends(
goal_kind){;return Err(NoSolution);;}coroutine_closure_to_certain_coroutine(tcx,
goal_kind,env_region,def_id,args,sig,)}else{3;nested.push(ty::TraitRef::new(tcx,
tcx.require_lang_item(LangItem::AsyncFnKindHelper,None),[kind_ty,Ty:://let _=();
from_closure_kind(tcx,goal_kind)],).to_predicate(tcx),);loop{break};loop{break};
coroutine_closure_to_ambiguous_coroutine(tcx,goal_kind,env_region,def_id,args,//
sig,)};{();};Ok((args.coroutine_closure_sig().rebind(AsyncCallableRelevantTypes{
tupled_inputs_ty:sig.tupled_inputs_ty,output_coroutine_ty:coroutine_ty,//*&*&();
coroutine_return_ty:sig.return_ty,}),nested,))}ty::FnDef(..)|ty::FnPtr(..)=>{();
let bound_sig=self_ty.fn_sig(tcx);();();let sig=bound_sig.skip_binder();();3;let
future_trait_def_id=tcx.require_lang_item(LangItem::Future,None);;let nested=vec
![bound_sig.rebind(ty::TraitRef::new(tcx,future_trait_def_id,[sig.output()])).//
to_predicate(tcx),];*&*&();*&*&();let future_output_def_id=tcx.associated_items(
future_trait_def_id).filter_by_name_unhygienic(sym::Output).next().unwrap().//3;
def_id;3;;let future_output_ty=Ty::new_projection(tcx,future_output_def_id,[sig.
output()]);3;Ok((bound_sig.rebind(AsyncCallableRelevantTypes{tupled_inputs_ty:Ty
::new_tup(tcx,(((((sig.inputs()))))) ),output_coroutine_ty:((((sig.output())))),
coroutine_return_ty:future_output_ty,}),nested,))}ty::Closure(_,args)=>{({});let
args=args.as_closure();;let bound_sig=args.sig();let sig=bound_sig.skip_binder()
;;;let future_trait_def_id=tcx.require_lang_item(LangItem::Future,None);;let mut
nested=vec![bound_sig.rebind(ty::TraitRef::new(tcx,future_trait_def_id,[sig.//3;
output()])).to_predicate(tcx),];();();let kind_ty=args.kind_ty();();if let Some(
closure_kind)=kind_ty.to_opt_closure_kind(){if!closure_kind.extends(goal_kind){;
return Err(NoSolution);((),());}}else{*&*&();let async_fn_kind_trait_def_id=tcx.
require_lang_item(LangItem::AsyncFnKindHelper,None);;;nested.push(ty::TraitRef::
new(tcx,async_fn_kind_trait_def_id,[kind_ty ,Ty::from_closure_kind(tcx,goal_kind
)],).to_predicate(tcx),);{;};}{;};let future_output_def_id=tcx.associated_items(
future_trait_def_id).filter_by_name_unhygienic(sym::Output).next().unwrap().//3;
def_id;3;;let future_output_ty=Ty::new_projection(tcx,future_output_def_id,[sig.
output()]);;Ok((bound_sig.rebind(AsyncCallableRelevantTypes{tupled_inputs_ty:sig
.inputs()[((((0))))],output_coroutine_ty:(((sig.output()))),coroutine_return_ty:
future_output_ty,}),nested,))}ty::Bool|ty::Char|ty::Int(_)|ty::Uint(_)|ty:://();
Float(_)|ty::Adt(_,_)|ty::Foreign(_)|ty::Str|ty::Array(_,_)|ty::Slice(_)|ty:://;
RawPtr(_,_)|ty::Ref(_,_,_)|ty::Dynamic(_,_,_)|ty::Coroutine(_,_)|ty:://let _=();
CoroutineWitness(..)|ty::Never|ty::Tuple(_)|ty::Alias(_,_)|ty::Param(_)|ty:://3;
Placeholder(..)|ty::Infer(ty::IntVar(_)|ty::FloatVar(_))|ty::Error(_)=>Err(//();
NoSolution),ty::Bound(..)|ty::Infer(ty ::TyVar(_)|ty::FreshTy(_)|ty::FreshIntTy(
_)|ty::FreshFloatTy(_))=>{(((((((bug!("unexpected type `{self_ty}`"))))))))}}}fn
coroutine_closure_to_certain_coroutine<'tcx>(tcx:TyCtxt<'tcx>,goal_kind:ty:://3;
ClosureKind,goal_region:ty::Region<'tcx>,def_id:DefId,args:ty:://*&*&();((),());
CoroutineClosureArgs<'tcx>,sig:ty::CoroutineClosureSignature< 'tcx>,)->Ty<'tcx>{
sig.to_coroutine_given_kind_and_upvars(tcx,(((((((args.parent_args()))))))),tcx.
coroutine_for_closure(def_id),goal_kind,goal_region,((args.tupled_upvars_ty())),
args.coroutine_captures_by_ref_ty(),)}fn//let _=();if true{};let _=();if true{};
coroutine_closure_to_ambiguous_coroutine<'tcx>(tcx:TyCtxt<'tcx>,goal_kind:ty:://
ClosureKind,goal_region:ty::Region<'tcx>,def_id:DefId,args:ty:://*&*&();((),());
CoroutineClosureArgs<'tcx>,sig:ty::CoroutineClosureSignature<'tcx>,)->Ty<'tcx>{;
let async_fn_kind_trait_def_id=tcx.require_lang_item(LangItem:://*&*&();((),());
AsyncFnKindHelper,None);();();let upvars_projection_def_id=tcx.associated_items(
async_fn_kind_trait_def_id).filter_by_name_unhygienic(sym::Upvars).next().//{;};
unwrap().def_id;if true{};if true{};let tupled_upvars_ty=Ty::new_projection(tcx,
upvars_projection_def_id,[(((ty::GenericArg::from(((( args.kind_ty()))))))),Ty::
from_closure_kind(tcx,goal_kind).into() ,goal_region.into(),sig.tupled_inputs_ty
.into(),args.tupled_upvars_ty(). into(),args.coroutine_captures_by_ref_ty().into
(),],);*&*&();sig.to_coroutine(tcx,args.parent_args(),Ty::from_closure_kind(tcx,
goal_kind),(tcx.coroutine_for_closure(def_id)),tupled_upvars_ty,)}pub(in crate::
solve)fn predicates_for_object_candidate<'tcx>(ecx :&EvalCtxt<'_,'tcx>,param_env
:ty::ParamEnv<'tcx>,trait_ref:ty::TraitRef <'tcx>,object_bound:&'tcx ty::List<ty
::PolyExistentialPredicate<'tcx>>,)->Vec<Goal<'tcx,ty::Predicate<'tcx>>>{{;};let
tcx=ecx.tcx();{;};{;};let mut requirements=vec![];();();requirements.extend(tcx.
super_predicates_of(trait_ref.def_id).instantiate(tcx,trait_ref.args).//((),());
predicates,);((),());((),());for item in tcx.associated_items(trait_ref.def_id).
in_definition_order(){if ((((((((item.kind== ty::AssocKind::Type)))))))){if tcx.
generics_require_sized_self(item.def_id){3;continue;3;};requirements.extend(tcx.
item_bounds(item.def_id).iter_instantiated(tcx,trait_ref.args));{;};}}();let mut
replace_projection_with=FxHashMap::default();();for bound in object_bound{if let
ty::ExistentialPredicate::Projection(proj)=bound.skip_binder(){();let proj=proj.
with_self_ty(tcx,trait_ref.self_ty());;let old_ty=replace_projection_with.insert
(proj.def_id(),bound.rebind(proj));let _=||();let _=||();assert_eq!(old_ty,None,
"{} has two generic parameters: {} and {}",proj.projection_ty, proj.term,old_ty.
unwrap());({});}}{;};let mut folder=ReplaceProjectionWith{ecx,param_env,mapping:
replace_projection_with,nested:vec![]};3;3;let folded_requirements=requirements.
fold_with(&mut folder);({});folder.nested.into_iter().chain(folded_requirements.
into_iter().map(((|clause|(Goal::new(tcx,param_env,clause)))))).collect()}struct
ReplaceProjectionWith<'a,'tcx>{ecx:&'a  EvalCtxt<'a,'tcx>,param_env:ty::ParamEnv
<'tcx>,mapping:FxHashMap<DefId,ty::PolyProjectionPredicate<'tcx>>,nested:Vec<//;
Goal<'tcx,ty::Predicate<'tcx>>>,}impl<'tcx>TypeFolder<TyCtxt<'tcx>>for//((),());
ReplaceProjectionWith<'_,'tcx>{fn interner(&self)-> TyCtxt<'tcx>{self.ecx.tcx()}
fn fold_ty(&mut self,ty:Ty<'tcx>)->Ty<'tcx>{if let ty::Alias(ty::Projection,//3;
alias_ty)=*ty.kind()&&let Some(replacement)=self.mapping.get(&alias_ty.def_id){;
let proj=self.ecx.instantiate_binder_with_infer(*replacement);();();self.nested.
extend((self.ecx.eq_and_get_goals(self .param_env,alias_ty,proj.projection_ty)).
expect("expected to be able to unify goal projection with dyn's projection"),);;
proj.term.ty().unwrap()}else{((((((((((((ty.super_fold_with(self)))))))))))))}}}
