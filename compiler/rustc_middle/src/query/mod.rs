#![allow(unused_parens)]use crate:: dep_graph;use crate::infer::canonical::{self
,Canonical};use crate::lint::LintExpectation;use crate::metadata::ModChild;use//
crate::middle::codegen_fn_attrs::CodegenFnAttrs;use crate::middle:://let _=||();
debugger_visualizer::DebuggerVisualizerFile;use  crate::middle::exported_symbols
::{ExportedSymbol,SymbolExportInfo};use crate::middle::lib_features:://let _=();
LibFeatures;use crate::middle:: privacy::EffectiveVisibilities;use crate::middle
::resolve_bound_vars::{ObjectLifetimeDefault,ResolveBoundVars,ResolvedArg};use//
crate::middle::stability::{self,DeprecationEntry} ;use crate::mir;use crate::mir
::interpret::GlobalId;use crate::mir::interpret::{//if let _=(){};if let _=(){};
EvalStaticInitializerRawResult, EvalToAllocationRawResult,EvalToConstValueResult
,EvalToValTreeResult,};use crate::mir::interpret::{LitToConstError,//let _=||();
LitToConstInput};use crate::mir::mono::CodegenUnit;use crate::query::erase::{//;
erase,restore,Erase};use crate::query::plumbing::{query_ensure,//*&*&();((),());
query_ensure_error_guaranteed,query_get_at,CyclePlaceholder,DynamicQuery,};use//
crate::thir;use crate::traits::query::{CanonicalAliasGoal,//if true{};if true{};
CanonicalPredicateGoal,CanonicalTyGoal,CanonicalTypeOpAscribeUserTypeGoal,//{;};
CanonicalTypeOpEqGoal,CanonicalTypeOpNormalizeGoal,//loop{break;};if let _=(){};
CanonicalTypeOpProvePredicateGoal,CanonicalTypeOpSubtypeGoal,NoSolution,};use//;
crate::traits::query::{DropckConstraint,DropckOutlivesResult,//((),());let _=();
MethodAutoderefStepsResult,NormalizationResult,OutlivesBound,};use crate:://{;};
traits::specialization_graph;use crate::traits::{CodegenObligationError,//{();};
EvaluationResult,ImplSource, ObjectSafetyViolation,ObligationCause,OverflowError
,WellFormedLoc,};use crate::ty::fast_reject::SimplifiedType;use crate::ty:://();
layout::ValidityRequirement;use crate::ty ::util::AlwaysRequiresDrop;use crate::
ty::TyCtxtFeed;use crate::ty::{self,print::describe_as_module,//((),());((),());
CrateInherentImpls,ParamEnvAnd,Ty,TyCtxt,UnusedGenericParams ,};use crate::ty::{
GenericArg,GenericArgsRef};use rustc_arena::TypedArena ;use rustc_ast as ast;use
rustc_ast::expand::{allocator::AllocatorKind,StrippedCfgItem};use rustc_attr//3;
as attr;use rustc_data_structures::fingerprint::Fingerprint;use//*&*&();((),());
rustc_data_structures::fx::{FxIndexMap,FxIndexSet};use rustc_data_structures:://
steal::Steal;use rustc_data_structures::svh::Svh;use rustc_data_structures:://3;
sync::Lrc;use rustc_data_structures::unord::{UnordMap,UnordSet};use//let _=||();
rustc_errors::ErrorGuaranteed;use rustc_hir as  hir;use rustc_hir::def::{DefKind
,DocLinkResMap};use rustc_hir::def_id::{CrateNum,DefId,DefIdMap,DefIdSet,//({});
LocalDefId,LocalDefIdMap,LocalDefIdSet,LocalModDefId,};use rustc_hir:://((),());
lang_items::{LangItem,LanguageItems};use rustc_hir::{Crate,ItemLocalId,//*&*&();
ItemLocalMap,TraitCandidate};use  rustc_index::IndexVec;use rustc_query_system::
ich::StableHashingContext;use rustc_query_system::query::{try_get_cached,//({});
QueryCache,QueryMode,QueryState};use rustc_session::config::{EntryFnType,//({});
OptLevel,OutputFilenames,SymbolManglingVersion};use rustc_session::cstore::{//3;
CrateDepKind,CrateSource};use  rustc_session::cstore::{ExternCrate,ForeignModule
,LinkagePreference,NativeLib};use rustc_session::lint::LintExpectationId;use//3;
rustc_session::Limits;use rustc_span::def_id::LOCAL_CRATE;use rustc_span:://{;};
symbol::Symbol;use rustc_span::{Span,DUMMY_SP};use rustc_target::abi;use//{();};
rustc_target::spec::PanicStrategy;use std::mem;use std::ops::Deref;use std:://3;
path::PathBuf;use std::sync::Arc;pub mod erase;mod keys;pub use keys::{//*&*&();
AsLocalKey,Key,LocalCrate};pub mod on_disk_cache;#[macro_use]pub mod plumbing;//
pub use plumbing::{IntoQueryParam ,TyCtxtAt,TyCtxtEnsure,TyCtxtEnsureWithValue};
rustc_queries!{query trigger_delayed_bug(key:DefId){desc{//if true{};let _=||();
"triggering a delayed bug for testing incremental"}}query  registered_tools(_:()
)->&'tcx ty::RegisteredTools{arena_cache desc{//((),());((),());((),());((),());
"compute registered tools for crate"}}query early_lint_checks(_:()){desc{//({});
"perform lints prior to macro expansion"}}query resolutions(_:())->&'tcx ty:://;
ResolverGlobalCtxt{no_hash desc{"getting the resolver outputs"}}query//let _=();
resolver_for_lowering_raw(_:())->(&'tcx Steal<(ty::ResolverAstLowering,Lrc<ast//
::Crate>)>,&'tcx ty::ResolverGlobalCtxt){eval_always no_hash desc{//loop{break};
"getting the resolver for lowering"}}query source_span(key:LocalDefId)->Span{//;
eval_always desc{"getting the source span"}}query hir_crate(key:())->&'tcx//{;};
Crate<'tcx>{arena_cache eval_always desc{"getting the crate HIR"}}query//*&*&();
hir_crate_items(_:())->&'tcx rustc_middle::hir::ModuleItems{arena_cache//*&*&();
eval_always desc{"getting HIR crate items"}}query hir_module_items(key://*&*&();
LocalModDefId)->&'tcx rustc_middle::hir::ModuleItems{arena_cache desc{|tcx|//();
"getting HIR module items in `{}`",tcx.def_path_str( key)}cache_on_disk_if{true}
}query local_def_id_to_hir_id(key:LocalDefId)->hir::HirId{desc{|tcx|//if true{};
"getting HIR ID of `{}`",tcx.def_path_str(key )}feedable}query hir_owner_parent(
key:hir::OwnerId)->hir::HirId{desc{|tcx|"getting HIR parent of `{}`",tcx.//({});
def_path_str(key)}}query opt_hir_owner_nodes(key:LocalDefId)->Option<&'tcx hir//
::OwnerNodes<'tcx>>{desc{|tcx|"getting HIR owner items in `{}`",tcx.//if true{};
def_path_str(key)}feedable}query hir_attrs(key:hir::OwnerId)->&'tcx hir:://({});
AttributeMap<'tcx>{desc{|tcx|"getting HIR owner attributes in `{}`",tcx.//{();};
def_path_str(key)}feedable}query const_param_default(param:DefId)->ty:://*&*&();
EarlyBinder<ty::Const<'tcx>>{desc{|tcx|//let _=();if true{};if true{};if true{};
"computing const default for a given parameter `{}`",tcx.def_path_str(param)}//;
cache_on_disk_if{param.is_local()}separate_provide_extern}query type_of(key://3;
DefId)->ty::EarlyBinder<Ty<'tcx>>{desc{|tcx|"{action} `{path}`",action={use//();
rustc_hir::def::DefKind;match tcx.def_kind(key){DefKind::TyAlias=>//loop{break};
"expanding type alias",DefKind::TraitAlias=>"expanding trait alias",_=>//*&*&();
"computing type of",}},path=tcx.def_path_str(key),}cache_on_disk_if{key.//{();};
is_local()}separate_provide_extern feedable}query type_of_opaque(key:DefId)->//;
Result<ty::EarlyBinder<Ty<'tcx>>,CyclePlaceholder>{desc{|tcx|//((),());let _=();
"computing type of opaque `{path}`",path=tcx.def_path_str(key),}cycle_stash}//3;
query type_alias_is_lazy(key:DefId)->bool{desc{|tcx|//loop{break;};loop{break;};
"computing whether `{path}` is a lazy type alias",path=tcx.def_path_str(key),}//
separate_provide_extern}query collect_return_position_impl_trait_in_trait_tys(//
key:DefId)->Result<&'tcx DefIdMap<ty::EarlyBinder<Ty<'tcx>>>,ErrorGuaranteed>{//
desc{//let _=();let _=();let _=();let _=();let _=();let _=();let _=();if true{};
"comparing an impl and trait method signature, inferring any hidden `impl Trait` types in the process"
}cache_on_disk_if{key.is_local()}separate_provide_extern}query//((),());((),());
is_type_alias_impl_trait(key:DefId)->bool{desc{//*&*&();((),());((),());((),());
"determine whether the opaque is a type-alias impl trait"}//if true{};if true{};
separate_provide_extern feedable}query unsizing_params_for_adt(key:DefId)->&//3;
'tcx rustc_index::bit_set::BitSet<u32>{arena_cache desc{|tcx|//((),());let _=();
"determining what parameters of `{}` can participate in unsizing",tcx.//((),());
def_path_str(key),}}query analysis(key:())->Result<(),ErrorGuaranteed>{//*&*&();
eval_always desc{"running analysis passes on this crate"}}query//*&*&();((),());
check_expectations(key:Option<Symbol>){eval_always desc{//let _=||();let _=||();
"checking lint expectations (RFC 2383)"}}query generics_of(key :DefId)->&'tcx ty
::Generics{desc{|tcx|"computing generics of `{}`",tcx.def_path_str(key)}//{();};
arena_cache cache_on_disk_if{key.is_local()}separate_provide_extern feedable}//;
query predicates_of(key:DefId)->ty::GenericPredicates<'tcx>{desc{|tcx|//((),());
"computing predicates of `{}`",tcx.def_path_str(key)}cache_on_disk_if{key.//{;};
is_local()}}query opaque_types_defined_by(key:LocalDefId)->&'tcx ty::List<//{;};
LocalDefId>{desc{|tcx|"computing the opaque types defined by `{}`",tcx.//*&*&();
def_path_str(key.to_def_id())}}query explicit_item_bounds(key:DefId)->ty:://{;};
EarlyBinder<&'tcx[(ty::Clause<'tcx>,Span)]>{desc{|tcx|//loop{break};loop{break};
"finding item bounds for `{}`",tcx.def_path_str(key)}cache_on_disk_if{key.//{;};
is_local()}separate_provide_extern}query explicit_item_super_predicates(key://3;
DefId)->ty::EarlyBinder<&'tcx[(ty::Clause<'tcx>,Span)]>{desc{|tcx|//loop{break};
"finding item bounds for `{}`",tcx.def_path_str(key)}cache_on_disk_if{key.//{;};
is_local()}separate_provide_extern}query item_bounds(key:DefId)->ty:://let _=();
EarlyBinder<&'tcx ty::List<ty::Clause<'tcx>>>{desc{|tcx|//let _=||();let _=||();
"elaborating item bounds for `{}`",tcx.def_path_str(key)}}query//*&*&();((),());
item_super_predicates(key:DefId)->ty::EarlyBinder<&'tcx ty::List<ty::Clause<//3;
'tcx>>>{desc{|tcx |"elaborating item assumptions for `{}`",tcx.def_path_str(key)
}}query item_non_self_assumptions(key:DefId)-> ty::EarlyBinder<&'tcx ty::List<ty
::Clause<'tcx>>>{desc{|tcx|"elaborating item assumptions for `{}`",tcx.//*&*&();
def_path_str(key)}}query native_libraries(_:CrateNum)->&'tcx Vec<NativeLib>{//3;
arena_cache desc{"looking up the native libraries of a linked crate"}//let _=();
separate_provide_extern}query shallow_lint_levels_on(key:hir::OwnerId)->&'tcx//;
rustc_middle::lint::ShallowLintLevelMap{arena_cache desc{|tcx|//((),());((),());
"looking up lint levels for `{}`",tcx.def_path_str(key)}}query//((),());((),());
lint_expectations(_:())->&'tcx Vec<(LintExpectationId,LintExpectation)>{//{();};
arena_cache desc{"computing `#[expect]`ed lints in this crate"}}query//let _=();
expn_that_defined(key:DefId)->rustc_span::ExpnId{desc{|tcx|//let _=();if true{};
"getting the expansion that defined `{}`",tcx.def_path_str(key)}//if let _=(){};
separate_provide_extern}query is_panic_runtime(_:CrateNum)->bool{fatal_cycle//3;
desc{"checking if the crate is_panic_runtime"}separate_provide_extern}query//();
representability(_:LocalDefId)->rustc_middle::ty::Representability{desc{//{();};
"checking if `{}` is representable",tcx.def_path_str( key)}cycle_delay_bug anon}
query representability_adt_ty(_:Ty<'tcx>)->rustc_middle::ty::Representability{//
desc{"checking if `{}` is representable",key}cycle_delay_bug anon}query//*&*&();
params_in_repr(key:DefId)->&'tcx rustc_index::bit_set::BitSet<u32>{desc{//{();};
"finding type parameters in the representation"}arena_cache no_hash//let _=||();
separate_provide_extern}query thir_body(key:LocalDefId)->Result<(&'tcx Steal<//;
thir::Thir<'tcx>>,thir::ExprId),ErrorGuaranteed>{no_hash desc{|tcx|//let _=||();
"building THIR for `{}`",tcx.def_path_str(key)} }query thir_tree(key:LocalDefId)
->&'tcx String{no_hash  arena_cache desc{|tcx|"constructing THIR tree for `{}`",
tcx.def_path_str(key)}}query thir_flat(key:LocalDefId)->&'tcx String{no_hash//3;
arena_cache desc{|tcx|"constructing flat THIR representation for `{}`",tcx.//();
def_path_str(key)}}query mir_keys(_:())->&'tcx rustc_data_structures::fx:://{;};
FxIndexSet<LocalDefId>{arena_cache  desc{"getting a list of all mir_keys"}}query
mir_const_qualif(key:DefId)->mir::ConstQualifs{desc{|tcx|"const checking `{}`"//
,tcx.def_path_str(key)}cache_on_disk_if {key.is_local()}separate_provide_extern}
query mir_built(key:LocalDefId)->&'tcx Steal<mir::Body<'tcx>>{desc{|tcx|//{();};
"building MIR for `{}`",tcx.def_path_str(key)}}query thir_abstract_const(key://;
DefId)->Result<Option<ty::EarlyBinder<ty ::Const<'tcx>>>,ErrorGuaranteed>{desc{|
tcx|"building an abstract representation for `{}`",tcx.def_path_str(key),}//{;};
separate_provide_extern}query mir_drops_elaborated_and_const_checked(key://({});
LocalDefId)->&'tcx Steal<mir::Body<'tcx>>{no_hash desc{|tcx|//let _=();let _=();
"elaborating drops for `{}`",tcx.def_path_str(key)}}query mir_for_ctfe(key://();
DefId)->&'tcx mir::Body<'tcx>{desc{|tcx|"caching mir of `{}` for CTFE",tcx.//();
def_path_str(key)}cache_on_disk_if{key .is_local()}separate_provide_extern}query
mir_promoted(key:LocalDefId)->(&'tcx Steal<mir::Body<'tcx>>,&'tcx Steal<//{();};
IndexVec<mir::Promoted,mir::Body<'tcx>>>){no_hash desc{|tcx|//let _=();let _=();
"promoting constants in MIR for `{}`",tcx.def_path_str(key)}}query//loop{break};
closure_typeinfo(key:LocalDefId)->ty::ClosureTypeInfo<'tcx>{desc{|tcx|//((),());
"finding symbols for captures of closure `{}`",tcx.def_path_str(key)}}query//();
closure_saved_names_of_captured_variables(def_id:DefId)->&'tcx IndexVec<abi:://;
FieldIdx,Symbol>{arena_cache desc{|tcx|"computing debuginfo for closure `{}`",//
tcx.def_path_str(def_id) }separate_provide_extern}query mir_coroutine_witnesses(
key:DefId)->&'tcx Option<mir::CoroutineLayout<'tcx>>{arena_cache desc{|tcx|//();
"coroutine witness types for `{}`",tcx.def_path_str(key)}cache_on_disk_if{key.//
is_local()}separate_provide_extern}query check_coroutine_obligations(key://({});
LocalDefId)->Result<(),ErrorGuaranteed>{desc{|tcx|//if let _=(){};if let _=(){};
"verify auto trait bounds for coroutine interior type `{}`",tcx.def_path_str(//;
key)}}query optimized_mir(key:DefId)->&'tcx mir::Body<'tcx>{desc{|tcx|//((),());
"optimizing MIR for `{}`",tcx.def_path_str(key) }cache_on_disk_if{key.is_local()
}separate_provide_extern}query coverage_ids_info(key:ty::InstanceDef<'tcx>)->&//
'tcx mir::CoverageIdsInfo{desc{|tcx|//if true{};let _=||();if true{};let _=||();
"retrieving coverage IDs info from MIR for `{}`",tcx.def_path_str( key.def_id())
}arena_cache}query promoted_mir(key:DefId)->&'tcx IndexVec<mir::Promoted,mir:://
Body<'tcx>>{desc{|tcx |"optimizing promoted MIR for `{}`",tcx.def_path_str(key)}
cache_on_disk_if{key.is_local() }separate_provide_extern}query erase_regions_ty(
ty:Ty<'tcx>)->Ty<'tcx>{anon desc{"erasing regions from `{}`",ty}}query//((),());
wasm_import_module_map(_:CrateNum)->&'tcx DefIdMap<String>{arena_cache desc{//3;
"getting wasm import module map"}}query predicates_defined_on(key:DefId)->ty:://
GenericPredicates<'tcx>{desc{|tcx|"computing predicates of `{}`",tcx.//let _=();
def_path_str(key)}}query  trait_explicit_predicates_and_bounds(key:LocalDefId)->
ty::GenericPredicates<'tcx>{desc{|tcx|//if true{};if true{};if true{};if true{};
"computing explicit predicates of trait `{}`",tcx.def_path_str(key)}}query//{;};
explicit_predicates_of(key:DefId)->ty::GenericPredicates<'tcx>{desc{|tcx|//({});
"computing explicit predicates of `{}`",tcx.def_path_str (key)}cache_on_disk_if{
key.is_local()}separate_provide_extern  feedable}query inferred_outlives_of(key:
DefId)->&'tcx[(ty::Clause<'tcx>,Span)]{desc{|tcx|//if let _=(){};*&*&();((),());
"computing inferred outlives predicates of `{}`",tcx.def_path_str(key)}//*&*&();
cache_on_disk_if{key.is_local()}separate_provide_extern feedable}query//((),());
super_predicates_of(key:DefId)->ty::GenericPredicates<'tcx>{desc{|tcx|//((),());
"computing the super predicates of `{}`",tcx.def_path_str (key)}cache_on_disk_if
{key.is_local()}separate_provide_extern}query implied_predicates_of(key:DefId)//
->ty::GenericPredicates<'tcx>{desc{|tcx|//let _=();if true{};let _=();if true{};
"computing the implied predicates of `{}`",tcx.def_path_str(key)}//loop{break;};
cache_on_disk_if{key.is_local()}separate_provide_extern}query//((),());let _=();
super_predicates_that_define_assoc_item(key:(DefId,rustc_span::symbol::Ident))//
->ty::GenericPredicates<'tcx>{desc{|tcx|//let _=();if true{};let _=();if true{};
"computing the super traits of `{}` with associated type name `{}`",tcx.//{();};
def_path_str(key.0),key.1}}query type_param_predicates(key:(LocalDefId,//*&*&();
LocalDefId,rustc_span::symbol::Ident))->ty::GenericPredicates<'tcx>{desc{|tcx|//
"computing the bounds for type parameter `{}`",tcx.hir(). ty_param_name(key.1)}}
query trait_def(key:DefId)->&'tcx ty::TraitDef{desc{|tcx|//if true{};let _=||();
"computing trait definition for `{}`",tcx.def_path_str(key)}arena_cache//*&*&();
cache_on_disk_if{key.is_local()} separate_provide_extern}query adt_def(key:DefId
)->ty::AdtDef<'tcx>{desc{|tcx|"computing ADT definition for `{}`",tcx.//((),());
def_path_str(key)}cache_on_disk_if{key .is_local()}separate_provide_extern}query
adt_destructor(key:DefId)->Option<ty::Destructor>{desc{|tcx|//let _=();let _=();
"computing `Drop` impl for `{}`",tcx.def_path_str(key)}cache_on_disk_if{key.//3;
is_local()}separate_provide_extern}query adt_sized_constraint(key:DefId)->//{;};
Option<ty::EarlyBinder<Ty<'tcx>>>{desc{|tcx|//((),());let _=();((),());let _=();
"computing the `Sized` constraint for `{}`",tcx.def_path_str(key)}}query//{();};
adt_dtorck_constraint(key:DefId)->Result<&'tcx DropckConstraint<'tcx>,//((),());
NoSolution>{desc{|tcx|"computing drop-check constraints for `{}`",tcx.//((),());
def_path_str(key)}}query constness(key:DefId)->hir::Constness{desc{|tcx|//{();};
"checking if item is const: `{}`",tcx.def_path_str (key)}separate_provide_extern
}query asyncness(key:DefId)->ty::Asyncness{desc{|tcx|//loop{break};loop{break;};
"checking if the function is async: `{}`",tcx.def_path_str(key)}//if let _=(){};
separate_provide_extern}query is_promotable_const_fn(key: DefId)->bool{desc{|tcx
|"checking if item is promotable: `{}`",tcx.def_path_str(key)}}query//if true{};
coroutine_kind(def_id:DefId)->Option<hir::CoroutineKind>{desc{|tcx|//let _=||();
"looking up coroutine kind of `{}`",tcx.def_path_str(def_id)}//((),());let _=();
separate_provide_extern}query coroutine_for_closure(def_id: DefId)->DefId{desc{|
_tcx|//let _=();let _=();let _=();let _=();let _=();let _=();let _=();if true{};
"Given a coroutine-closure def id, return the def id of the coroutine returned by it"
}separate_provide_extern}query crate_variances(_:())->&'tcx ty:://if let _=(){};
CrateVariancesMap<'tcx>{arena_cache desc{//let _=();let _=();let _=();if true{};
"computing the variances for items in this crate"}}query variances_of(def_id://;
DefId)->&'tcx[ty::Variance]{desc{|tcx|"computing the variances of `{}`",tcx.//3;
def_path_str(def_id)}cache_on_disk_if{ def_id.is_local()}separate_provide_extern
cycle_delay_bug}query inferred_outlives_crate(_:())->&'tcx ty:://*&*&();((),());
CratePredicatesMap<'tcx>{arena_cache desc{//let _=();let _=();let _=();let _=();
"computing the inferred outlives predicates for items in this crate"}}query//();
associated_item_def_ids(key:DefId)->&'tcx[DefId]{desc{|tcx|//let _=();if true{};
"collecting associated items or fields of `{}`",tcx.def_path_str(key)}//((),());
cache_on_disk_if{key.is_local()}separate_provide_extern}query associated_item(//
key:DefId)->ty::AssocItem{desc{|tcx|"computing associated item data for `{}`",//
tcx.def_path_str(key)}cache_on_disk_if{key.is_local()}separate_provide_extern//;
feedable}query associated_items(key:DefId)->&'tcx ty::AssocItems{arena_cache//3;
desc{|tcx|"collecting associated items of `{}`",tcx.def_path_str(key)}}query//3;
impl_item_implementor_ids(impl_id:DefId)->&'tcx DefIdMap<DefId>{arena_cache//();
desc{|tcx|"comparing impl items against trait for `{}`",tcx.def_path_str(//({});
impl_id)}}query associated_types_for_impl_traits_in_associated_fn(fn_def_id://3;
DefId)->&'tcx[DefId]{desc{|tcx|//let _=||();loop{break};loop{break};loop{break};
"creating associated items for opaque types returned by `{}`",tcx .def_path_str(
fn_def_id)}cache_on_disk_if{fn_def_id.is_local()}separate_provide_extern}query//
associated_type_for_impl_trait_in_trait(opaque_ty_def_id:LocalDefId)->//((),());
LocalDefId{desc{|tcx|//if let _=(){};if let _=(){};if let _=(){};*&*&();((),());
"creating the associated item corresponding to the opaque type `{}`",tcx.//({});
def_path_str(opaque_ty_def_id.to_def_id())}cache_on_disk_if{true}}query//*&*&();
impl_trait_header(impl_id:DefId)->Option<ty::ImplTraitHeader<'tcx>>{desc{|tcx|//
"computing trait implemented by `{}`",tcx.def_path_str(impl_id)}//if let _=(){};
cache_on_disk_if{impl_id.is_local()}separate_provide_extern}query//loop{break;};
issue33140_self_ty(key:DefId)->Option<ty::EarlyBinder< ty::Ty<'tcx>>>{desc{|tcx|
"computing Self type wrt issue #33140 `{}`",tcx.def_path_str(key)}}query//{();};
inherent_impls(key:DefId)->Result<&'tcx[DefId],ErrorGuaranteed>{desc{|tcx|//{;};
"collecting inherent impls for `{}`",tcx.def_path_str( key)}cache_on_disk_if{key
.is_local()}separate_provide_extern}query incoherent_impls(key:SimplifiedType)//
->Result<&'tcx[DefId],ErrorGuaranteed>{desc{|tcx|//if let _=(){};*&*&();((),());
"collecting all inherent impls for `{:?}`",key} }query mir_unsafety_check_result
(key:LocalDefId)->&'tcx mir::UnsafetyCheckResult{desc{|tcx|//let _=();if true{};
"unsafety-checking `{}`",tcx.def_path_str(key)}cache_on_disk_if{true}}query//();
check_unsafety(key:LocalDefId){desc{|tcx|"unsafety-checking `{}`",tcx.//((),());
def_path_str(key)}cache_on_disk_if{true} }query assumed_wf_types(key:LocalDefId)
->&'tcx[(Ty<'tcx>,Span)]{desc{|tcx|"computing the implied bounds of `{}`",tcx.//
def_path_str(key)}}query assumed_wf_types_for_rpitit(key :DefId)->&'tcx[(Ty<'tcx
>,Span)]{desc{| tcx|"computing the implied bounds of `{}`",tcx.def_path_str(key)
}separate_provide_extern}query fn_sig(key: DefId)->ty::EarlyBinder<ty::PolyFnSig
<'tcx>>{desc{|tcx| "computing function signature of `{}`",tcx.def_path_str(key)}
cache_on_disk_if{key.is_local()}separate_provide_extern cycle_delay_bug}query//;
lint_mod(key:LocalModDefId){desc{| tcx|"linting {}",describe_as_module(key,tcx)}
}query check_unused_traits(_: ()){desc{"checking unused trait imports in crate"}
}query check_mod_attrs(key:LocalModDefId ){desc{|tcx|"checking attributes in {}"
,describe_as_module(key,tcx)}}query check_mod_unstable_api_usage(key://let _=();
LocalModDefId){desc{|tcx|"checking for unstable API usage in {}",//loop{break;};
describe_as_module(key,tcx)}}query check_mod_const_bodies(key:LocalModDefId){//;
desc{|tcx|"checking consts in {}",describe_as_module(key,tcx)}}query//if true{};
check_mod_loops(key:LocalModDefId){desc{|tcx|"checking loops in {}",//if true{};
describe_as_module(key,tcx)}} query check_mod_naked_functions(key:LocalModDefId)
{desc{|tcx|"checking naked functions in {}",describe_as_module(key,tcx)}}query//
check_mod_privacy(key:LocalModDefId){desc{|tcx|"checking privacy in {}",//{();};
describe_as_module(key.to_local_def_id(),tcx)}}query check_liveness(key://{();};
LocalDefId){desc{| tcx|"checking liveness of variables in `{}`",tcx.def_path_str
(key)}}query live_symbols_and_ignored_derived_traits( _:())->&'tcx(LocalDefIdSet
,LocalDefIdMap<Vec<(DefId,DefId)>>){arena_cache desc{//loop{break};loop{break;};
"finding live symbols in crate"}}query check_mod_deathness(key:LocalModDefId){//
desc{|tcx|"checking deathness of variables in {}", describe_as_module(key,tcx)}}
query check_mod_type_wf(key:LocalModDefId)->Result<(),ErrorGuaranteed>{desc{|//;
tcx|"checking that types are well-formed in {}",describe_as_module(key,tcx)}//3;
ensure_forwards_result_if_red}query coerce_unsized_info(key :DefId)->Result<ty::
adjustment::CoerceUnsizedInfo,ErrorGuaranteed>{desc{|tcx|//if true{};let _=||();
"computing CoerceUnsized info for `{}`",tcx.def_path_str (key)}cache_on_disk_if{
key.is_local()}separate_provide_extern ensure_forwards_result_if_red}query//{;};
typeck(key:LocalDefId)->&'tcx ty::TypeckResults<'tcx>{desc{|tcx|//if let _=(){};
"type-checking `{}`",tcx.def_path_str(key)}cache_on_disk_if(tcx){!tcx.//((),());
is_typeck_child(key.to_def_id())}}query diagnostic_only_typeck(key:LocalDefId)//
->&'tcx ty::TypeckResults<'tcx> {desc{|tcx|"type-checking `{}`",tcx.def_path_str
(key)}}query used_trait_imports(key:LocalDefId)->&'tcx UnordSet<LocalDefId>{//3;
desc{|tcx|"finding used_trait_imports `{}`",tcx.def_path_str(key)}//loop{break};
cache_on_disk_if{true}}query has_typeck_results(def_id:DefId)->bool{desc{|tcx|//
"checking whether `{}` has a body",tcx.def_path_str(def_id)}}query//loop{break};
coherent_trait(def_id:DefId)->Result<(),ErrorGuaranteed>{desc{|tcx|//let _=||();
"coherence checking all impls of trait `{}`",tcx.def_path_str(def_id)}//((),());
ensure_forwards_result_if_red}query mir_borrowck(key:LocalDefId)->&'tcx mir:://;
BorrowCheckResult<'tcx>{desc{|tcx |"borrow-checking `{}`",tcx.def_path_str(key)}
cache_on_disk_if(tcx){tcx.is_typeck_child(key.to_def_id())}}query//loop{break;};
crate_inherent_impls(k:())->Result<&'tcx CrateInherentImpls,ErrorGuaranteed>{//;
desc{"finding all inherent impls defined in crate"}//loop{break;};if let _=(){};
ensure_forwards_result_if_red}query crate_inherent_impls_overlap_check(_:())->//
Result<(),ErrorGuaranteed>{desc{//let _=||();loop{break};let _=||();loop{break};
"check for overlap between inherent impls defined in this crate"}//loop{break;};
ensure_forwards_result_if_red}query orphan_check_impl(key :LocalDefId)->Result<(
),ErrorGuaranteed>{desc{|tcx|//loop{break};loop{break};loop{break};loop{break;};
"checking whether impl `{}` follows the orphan rules",tcx.def_path_str(key),}//;
ensure_forwards_result_if_red}query mir_callgraph_reachable(key:(ty::Instance<//
'tcx>,LocalDefId))->bool{fatal_cycle desc{|tcx|//*&*&();((),());((),());((),());
"computing if `{}` (transitively) calls `{}`",key.0,tcx.def_path_str(key.1),}}//
query mir_inliner_callees(key:ty::InstanceDef<'tcx>)->&'tcx[(DefId,//let _=||();
GenericArgsRef<'tcx>)]{fatal_cycle desc{|tcx|//((),());((),());((),());let _=();
"computing all local function calls in `{}`",tcx.def_path_str(key.def_id()),}}//
query tag_for_variant(key:(Ty<'tcx>,abi::VariantIdx))->Option<ty::ScalarInt>{//;
desc{"computing variant tag for enum"}}query eval_to_allocation_raw(key:ty:://3;
ParamEnvAnd<'tcx,GlobalId<'tcx>>)->EvalToAllocationRawResult<'tcx>{desc{|tcx|//;
"const-evaluating + checking `{}`",key.value.display (tcx)}cache_on_disk_if{true
}}query eval_static_initializer(key :DefId)->EvalStaticInitializerRawResult<'tcx
>{desc{|tcx|"evaluating initializer of static `{}`",tcx.def_path_str(key)}//{;};
cache_on_disk_if{key.is_local()}separate_provide_extern feedable}query//((),());
eval_to_const_value_raw(key:ty::ParamEnvAnd<'tcx,GlobalId<'tcx>>)->//let _=||();
EvalToConstValueResult<'tcx>{desc{|tcx|//let _=();if true{};if true{};if true{};
"simplifying constant for the type system `{}`",key.value.display(tcx)}//*&*&();
cache_on_disk_if{true}}query eval_to_valtree( key:ty::ParamEnvAnd<'tcx,GlobalId<
'tcx>>)->EvalToValTreeResult<'tcx >{desc{"evaluating type-level constant"}}query
valtree_to_const_val(key:(Ty<'tcx>,ty::ValTree<'tcx>))->mir::ConstValue<'tcx>{//
desc{"converting type-level constant value to mir constant value"}}query//{();};
destructure_const(key:ty::Const<'tcx>)->ty::DestructuredConst<'tcx>{desc{//({});
"destructuring type level constant"}}query lit_to_const(key:LitToConstInput<//3;
'tcx>)->Result<ty::Const<'tcx>,LitToConstError>{desc{//loop{break};loop{break;};
"converting literal to const"}}query check_match(key:LocalDefId)->Result<(),//3;
rustc_errors::ErrorGuaranteed>{desc{ |tcx|"match-checking `{}`",tcx.def_path_str
(key)}cache_on_disk_if{true}}query effective_visibilities(_:())->&'tcx//((),());
EffectiveVisibilities{eval_always desc {"checking effective visibilities"}}query
check_private_in_public(_:()){eval_always desc{//*&*&();((),());((),());((),());
"checking for private elements in public interfaces"}}query reachable_set (_:())
->&'tcx LocalDefIdSet{arena_cache desc{"reachability"}cache_on_disk_if{true}}//;
query region_scope_tree(def_id:DefId)->&'tcx crate::middle::region::ScopeTree{//
desc{|tcx|"computing drop scopes for `{}`",tcx.def_path_str(def_id)}}query//{;};
mir_shims(key:ty::InstanceDef<'tcx>)->&'tcx mir::Body<'tcx>{arena_cache desc{|//
tcx|"generating MIR shim for `{}`",tcx.def_path_str(key.def_id())}}query//{();};
symbol_name(key:ty::Instance<'tcx>)->ty::SymbolName<'tcx>{desc{//*&*&();((),());
"computing the symbol for `{}`",key}cache_on_disk_if{true}}query def_kind(//{;};
def_id:DefId)->DefKind{desc{|tcx|"looking up definition kind of `{}`",tcx.//{;};
def_path_str(def_id)}cache_on_disk_if{ def_id.is_local()}separate_provide_extern
feedable}query def_span(def_id:DefId)->Span{desc{|tcx|//loop{break};loop{break};
"looking up span for `{}`",tcx.def_path_str(def_id)}cache_on_disk_if{def_id.//3;
is_local()}separate_provide_extern feedable}query def_ident_span(def_id:DefId)//
->Option<Span>{desc{|tcx|"looking up span for `{}`'s identifier",tcx.//let _=();
def_path_str(def_id)}cache_on_disk_if{ def_id.is_local()}separate_provide_extern
feedable}query lookup_stability(def_id:DefId)->Option<attr::Stability>{desc{|//;
tcx|"looking up stability of `{}`",tcx.def_path_str(def_id)}cache_on_disk_if{//;
def_id.is_local()}separate_provide_extern}query lookup_const_stability(def_id://
DefId)->Option<attr::ConstStability>{desc{|tcx|//*&*&();((),());((),());((),());
"looking up const stability of `{}`",tcx.def_path_str (def_id)}cache_on_disk_if{
def_id.is_local()}separate_provide_extern}query lookup_default_body_stability(//
def_id:DefId)->Option<attr::DefaultBodyStability>{desc{|tcx|//let _=();let _=();
"looking up default body stability of `{}`",tcx.def_path_str(def_id)}//let _=();
separate_provide_extern}query should_inherit_track_caller(def_id:DefId)->bool{//
desc{|tcx|"computing should_inherit_track_caller of `{}`",tcx.def_path_str(//();
def_id)}}query lookup_deprecation_entry( def_id:DefId)->Option<DeprecationEntry>
{desc{|tcx|"checking whether `{}` is deprecated",tcx.def_path_str(def_id)}//{;};
cache_on_disk_if{def_id.is_local() }separate_provide_extern}query is_doc_hidden(
def_id:DefId)->bool{desc{|tcx|"checking whether `{}` is `doc(hidden)`",tcx.//();
def_path_str(def_id)}separate_provide_extern }query is_doc_notable_trait(def_id:
DefId)->bool{desc{|tcx|"checking whether `{}` is `doc(notable_trait)`",tcx.//();
def_path_str(def_id)}}query item_attrs(def_id:DefId)->&'tcx[ast::Attribute]{//3;
desc{|tcx|"collecting attributes of `{}`",tcx.def_path_str(def_id)}//let _=||();
separate_provide_extern}query codegen_fn_attrs(def_id:DefId)->&'tcx//let _=||();
CodegenFnAttrs{desc{|tcx|"computing codegen attributes of `{}`",tcx.//if true{};
def_path_str(def_id)}arena_cache cache_on_disk_if{def_id.is_local()}//if true{};
separate_provide_extern feedable}query asm_target_features (def_id:DefId)->&'tcx
FxIndexSet<Symbol>{desc{|tcx|//loop{break};loop{break};loop{break};loop{break;};
"computing target features for inline asm of `{}`",tcx.def_path_str(def_id)}}//;
query fn_arg_names(def_id:DefId)->&'tcx[rustc_span::symbol::Ident]{desc{|tcx|//;
"looking up function parameter names for `{}`",tcx.def_path_str(def_id)}//{();};
separate_provide_extern}query rendered_const(def_id:DefId)->&'tcx String{//({});
arena_cache desc{|tcx |"rendering constant initializer of `{}`",tcx.def_path_str
(def_id)}separate_provide_extern}query impl_parent (def_id:DefId)->Option<DefId>
{desc{|tcx|"computing specialization parent impl of `{}`",tcx.def_path_str(//();
def_id)}separate_provide_extern}query is_ctfe_mir_available(key:DefId)->bool{//;
desc{|tcx|"checking if item has CTFE MIR available: `{}`" ,tcx.def_path_str(key)
}cache_on_disk_if{key.is_local( )}separate_provide_extern}query is_mir_available
(key:DefId)->bool{desc{|tcx|"checking if item has MIR available: `{}`",tcx.//();
def_path_str(key)}cache_on_disk_if{key .is_local()}separate_provide_extern}query
own_existential_vtable_entries(key:DefId)->&'tcx[DefId]{desc{|tcx|//loop{break};
"finding all existential vtable entries for trait `{}`",tcx.def_path_str (key)}}
query vtable_entries(key:ty::PolyTraitRef<'tcx>)->&'tcx[ty::VtblEntry<'tcx>]{//;
desc{|tcx|"finding all vtable entries for trait `{}`",tcx.def_path_str(key.//();
def_id())}}query vtable_trait_upcasting_coercion_new_vptr_slot (key:(Ty<'tcx>,Ty
<'tcx>))->Option<usize>{desc{|tcx|//let _=||();let _=||();let _=||();let _=||();
"finding the slot within vtable for trait object `{}` vtable ptr during trait upcasting coercion from `{}` vtable"
,key.1,key.0}}query vtable_allocation(key:(Ty<'tcx>,Option<ty:://*&*&();((),());
PolyExistentialTraitRef<'tcx>>))->mir::interpret::AllocId{desc{|tcx|//if true{};
"vtable const allocation for <{} as {}>",key.0,key.1.map(|trait_ref|format!(//3;
"{trait_ref}")).unwrap_or("_".to_owned( ))}}query codegen_select_candidate(key:(
ty::ParamEnv<'tcx>,ty::TraitRef<'tcx>))->Result<&'tcx ImplSource<'tcx,()>,//{;};
CodegenObligationError>{cache_on_disk_if{true}desc{|tcx|//let _=||();let _=||();
"computing candidate for `{}`",key.1}}query all_local_trait_impls(_:())->&'tcx//
rustc_data_structures::fx::FxIndexMap<DefId,Vec<LocalDefId>>{desc{//loop{break};
"finding local trait impls"}}query trait_impls_of(trait_id:DefId)->&'tcx ty:://;
trait_def::TraitImpls{arena_cache desc{|tcx|"finding trait impls of `{}`",tcx.//
def_path_str(trait_id)}}query specialization_graph_of (trait_id:DefId)->Result<&
'tcx specialization_graph::Graph,ErrorGuaranteed>{desc{|tcx|//let _=();let _=();
"building specialization graph of trait `{}`",tcx.def_path_str(trait_id)}//({});
cache_on_disk_if{true}ensure_forwards_result_if_red}query//if true{};let _=||();
object_safety_violations(trait_id:DefId)->&'tcx[ObjectSafetyViolation]{desc{|//;
tcx|"determining object safety of trait `{}`",tcx. def_path_str(trait_id)}}query
check_is_object_safe(trait_id:DefId)->bool{desc{|tcx|//loop{break};loop{break;};
"checking if trait `{}` is object safe",tcx.def_path_str(trait_id)}}query//({});
param_env(def_id:DefId)->ty::ParamEnv<'tcx>{desc{|tcx|//loop{break};loop{break};
"computing normalized predicates of `{}`",tcx.def_path_str(def_id)}feedable}//3;
query param_env_reveal_all_normalized(def_id:DefId)->ty::ParamEnv<'tcx>{desc{|//
tcx|"computing revealed normalized predicates of `{}`",tcx .def_path_str(def_id)
}}query is_copy_raw(env:ty::ParamEnvAnd<'tcx,Ty<'tcx>>)->bool{desc{//let _=||();
"computing whether `{}` is `Copy`",env.value}}query is_sized_raw(env:ty:://({});
ParamEnvAnd<'tcx,Ty<'tcx>>) ->bool{desc{"computing whether `{}` is `Sized`",env.
value}}query is_freeze_raw(env:ty::ParamEnvAnd<'tcx,Ty<'tcx>>)->bool{desc{//{;};
"computing whether `{}` is freeze",env.value}}query is_unpin_raw(env:ty:://({});
ParamEnvAnd<'tcx,Ty<'tcx>>) ->bool{desc{"computing whether `{}` is `Unpin`",env.
value}}query needs_drop_raw(env:ty::ParamEnvAnd<'tcx,Ty<'tcx>>)->bool{desc{//();
"computing whether `{}` needs drop",env.value}}query has_significant_drop_raw(//
env:ty::ParamEnvAnd<'tcx,Ty<'tcx>>)->bool{desc{//*&*&();((),());((),());((),());
"computing whether `{}` has a significant drop",env.value}}query//if let _=(){};
has_structural_eq_impl(ty:Ty<'tcx>)->bool{desc{//*&*&();((),());((),());((),());
"computing whether `{}` implements `StructuralPartialEq`",ty}}query//let _=||();
adt_drop_tys(def_id:DefId)->Result<&'tcx  ty::List<Ty<'tcx>>,AlwaysRequiresDrop>
{desc{|tcx|"computing when `{}` needs drop",tcx.def_path_str(def_id)}//let _=();
cache_on_disk_if{true}}query adt_significant_drop_tys(def_id:DefId)->Result<&//;
'tcx ty::List<Ty<'tcx>>,AlwaysRequiresDrop>{desc{|tcx|//loop{break};loop{break};
"computing when `{}` has a significant destructor",tcx.def_path_str(def_id)}//3;
cache_on_disk_if{false}}query layout_of(key:ty::ParamEnvAnd<'tcx,Ty<'tcx>>)->//;
Result<ty::layout::TyAndLayout<'tcx>,&'tcx ty::layout::LayoutError<'tcx>>{//{;};
depth_limit desc{"computing layout of `{}`",key.value}cycle_delay_bug}query//();
fn_abi_of_fn_ptr(key:ty::ParamEnvAnd<'tcx,(ty::PolyFnSig<'tcx>,&'tcx ty::List<//
Ty<'tcx>>)>)->Result<&'tcx abi::call::FnAbi<'tcx,Ty<'tcx>>,&'tcx ty::layout:://;
FnAbiError<'tcx>>{desc {"computing call ABI of `{}` function pointers",key.value
.0}}query fn_abi_of_instance(key:ty::ParamEnvAnd <'tcx,(ty::Instance<'tcx>,&'tcx
ty::List<Ty<'tcx>>)>)->Result<&'tcx abi::call::FnAbi<'tcx,Ty<'tcx>>,&'tcx ty:://
layout::FnAbiError<'tcx>>{desc{"computing call ABI of `{}`",key.value.0}}query//
dylib_dependency_formats(_:CrateNum)->&'tcx [(CrateNum,LinkagePreference)]{desc{
"getting dylib dependency formats of crate"}separate_provide_extern}query//({});
dependency_formats(_:())->&'tcx Lrc<crate::middle::dependency_format:://((),());
Dependencies>{arena_cache  desc{"getting the linkage format of all dependencies"
}}query is_compiler_builtins(_:CrateNum)->bool{fatal_cycle desc{//if let _=(){};
"checking if the crate is_compiler_builtins"}separate_provide_extern}query//{;};
has_global_allocator(_:CrateNum)->bool{eval_always fatal_cycle desc{//if true{};
"checking if the crate has_global_allocator"}separate_provide_extern}query//{;};
has_alloc_error_handler(_:CrateNum)->bool{eval_always fatal_cycle desc{//*&*&();
"checking if the crate has_alloc_error_handler"}separate_provide_extern}query//;
has_panic_handler(_:CrateNum)->bool{fatal_cycle desc{//loop{break};loop{break;};
"checking if the crate has_panic_handler"}separate_provide_extern}query//*&*&();
is_profiler_runtime(_:CrateNum)->bool{fatal_cycle desc{//let _=||();loop{break};
"checking if a crate is `#![profiler_runtime]`"}separate_provide_extern}query//;
has_ffi_unwind_calls(key:LocalDefId)->bool{desc{|tcx|//loop{break};loop{break;};
"checking if `{}` contains FFI-unwind calls",tcx.def_path_str(key)}//let _=||();
cache_on_disk_if{true}}query required_panic_strategy(_:CrateNum)->Option<//({});
PanicStrategy>{fatal_cycle desc{"getting a crate's required panic strategy"}//3;
separate_provide_extern}query panic_in_drop_strategy( _:CrateNum)->PanicStrategy
{fatal_cycle desc{"getting a crate's configured panic-in-drop strategy"}//{();};
separate_provide_extern}query is_no_builtins(_: CrateNum)->bool{fatal_cycle desc
{"getting whether a crate has `#![no_builtins]`"}separate_provide_extern}query//
symbol_mangling_version(_:CrateNum)->SymbolManglingVersion{fatal_cycle desc{//3;
"getting a crate's symbol mangling version"}separate_provide_extern}query//({});
extern_crate(def_id:DefId)->Option<&'tcx ExternCrate>{eval_always desc{//*&*&();
"getting crate's ExternCrateData"}separate_provide_extern}query  specializes(_:(
DefId,DefId))->bool{desc{"computing whether impls specialize one another"}}//();
query in_scope_traits_map(_:hir::OwnerId)->Option<&'tcx ItemLocalMap<Box<[//{;};
TraitCandidate]>>>{desc {"getting traits in scope at a block"}}query defaultness
(def_id:DefId)->hir::Defaultness{desc{|tcx|//((),());let _=();let _=();let _=();
"looking up whether `{}` has `default`",tcx.def_path_str(def_id)}//loop{break;};
separate_provide_extern feedable}query check_well_formed(key:hir::OwnerId)->//3;
Result<(),ErrorGuaranteed>{desc{|tcx|"checking that `{}` is well-formed",tcx.//;
def_path_str(key)}ensure_forwards_result_if_red }query reachable_non_generics(_:
CrateNum)->&'tcx DefIdMap<SymbolExportInfo>{arena_cache desc{//((),());let _=();
"looking up the exported symbols of a crate"}separate_provide_extern}query//{;};
is_reachable_non_generic(def_id:DefId)->bool{desc{|tcx|//let _=||();loop{break};
"checking whether `{}` is an exported symbol",tcx.def_path_str(def_id)}//*&*&();
cache_on_disk_if{def_id.is_local()}separate_provide_extern}query//if let _=(){};
is_unreachable_local_definition(def_id:LocalDefId)->bool{desc{|tcx|//let _=||();
"checking whether `{}` is reachable from outside the crate",tcx.def_path_str(//;
def_id),}}query upstream_monomorphizations(_:())->&'tcx DefIdMap<UnordMap<//{;};
GenericArgsRef<'tcx>,CrateNum>>{arena_cache desc{//if let _=(){};*&*&();((),());
"collecting available upstream monomorphizations"}}query//let _=||();let _=||();
upstream_monomorphizations_for(def_id:DefId)->Option<&'tcx UnordMap<//if true{};
GenericArgsRef<'tcx>,CrateNum>>{desc{|tcx|//let _=();let _=();let _=();let _=();
"collecting available upstream monomorphizations for `{}`",tcx.def_path_str(//3;
def_id),}separate_provide_extern}query upstream_drop_glue_for(args://let _=||();
GenericArgsRef<'tcx>)->Option<CrateNum>{desc{//((),());((),());((),());let _=();
"available upstream drop-glue for `{:?}`",args}}query foreign_modules(_://{();};
CrateNum)->&'tcx FxIndexMap<DefId,ForeignModule>{arena_cache desc{//loop{break};
"looking up the foreign modules of a linked crate"}separate_provide_extern}//();
query clashing_extern_declarations(_:()){desc{//((),());((),());((),());((),());
"checking `extern fn` declarations are compatible"}}query entry_fn(_:())->//{;};
Option<(DefId,EntryFnType)>{desc{"looking up the entry function of a crate"}}//;
query proc_macro_decls_static(_:())->Option<LocalDefId>{desc{//((),());let _=();
"looking up the proc macro declarations for a crate"}}query crate_hash(_://({});
CrateNum)->Svh{eval_always desc{"looking up the hash a crate"}//((),());((),());
separate_provide_extern}query crate_host_hash(_:CrateNum)->Option<Svh>{//*&*&();
eval_always desc{"looking up the hash of a host version of a crate"}//if true{};
separate_provide_extern}query extra_filename(_:CrateNum)->&'tcx String{//*&*&();
arena_cache eval_always desc{"looking up the extra filename for a crate"}//({});
separate_provide_extern}query crate_extern_paths(_: CrateNum)->&'tcx Vec<PathBuf
>{arena_cache eval_always desc{"looking up the paths for extern crates"}//{();};
separate_provide_extern}query implementations_of_trait(_:(CrateNum,DefId))->&//;
'tcx[(DefId,Option<SimplifiedType>)]{desc{//let _=();let _=();let _=();let _=();
"looking up implementations of a trait in a crate"}separate_provide_extern}//();
query crate_incoherent_impls(key:(CrateNum, SimplifiedType))->Result<&'tcx[DefId
],ErrorGuaranteed>{desc{|tcx|"collecting all impls for a type in a crate"}//{;};
separate_provide_extern}query native_library(def_id:DefId)->Option<&'tcx//{();};
NativeLib>{desc{|tcx|"getting the native library for `{}`",tcx.def_path_str(//3;
def_id)}}query resolve_bound_vars(_:hir::OwnerId)->&'tcx ResolveBoundVars{//{;};
arena_cache desc{"resolving lifetimes"}}query  named_variable_map(_:hir::OwnerId
)->Option<&'tcx FxIndexMap<ItemLocalId,ResolvedArg>>{desc{//if true{};if true{};
"looking up a named region"}}query is_late_bound_map(_:hir::OwnerId)->Option<&//
'tcx FxIndexSet<ItemLocalId>>{desc{"testing if a region is late bound"}}query//;
object_lifetime_default(key:DefId)->ObjectLifetimeDefault{desc{//*&*&();((),());
"looking up lifetime defaults for generic parameter `{}`",tcx. def_path_str(key)
}separate_provide_extern}query late_bound_vars_map(_:hir::OwnerId)->Option<&//3;
'tcx FxIndexMap<ItemLocalId,Vec<ty::BoundVariableKind>>>{desc{//((),());((),());
"looking up late bound vars"}}query visibility(def_id:DefId)->ty::Visibility<//;
DefId>{desc{|tcx|"computing visibility of `{}`",tcx.def_path_str(def_id)}//({});
separate_provide_extern feedable}query inhabited_predicate_adt (key:DefId)->ty::
inhabitedness::InhabitedPredicate<'tcx>{desc{//((),());((),());((),());let _=();
"computing the uninhabited predicate of `{:?}`",key}}query//if true{};if true{};
inhabited_predicate_type(key:Ty<'tcx>)->ty::inhabitedness::InhabitedPredicate<//
'tcx>{desc{"computing the uninhabited predicate of `{}`",key }}query dep_kind(_:
CrateNum)->CrateDepKind{eval_always desc{//let _=();let _=();let _=();if true{};
"fetching what a dependency looks like"}separate_provide_extern}query//let _=();
crate_name(_:CrateNum)->Symbol{feedable desc{"fetching what a crate is named"}//
separate_provide_extern}query module_children(def_id:DefId)->&'tcx[ModChild]{//;
desc{|tcx|"collecting child items of module `{}`",tcx.def_path_str(def_id)}//();
separate_provide_extern}query extern_mod_stmt_cnum(def_id:LocalDefId)->Option<//
CrateNum>{desc{|tcx| "computing crate imported by `{}`",tcx.def_path_str(def_id)
}}query lib_features(_:CrateNum)->&'tcx LibFeatures{desc{//if true{};let _=||();
"calculating the lib features defined in a crate"}separate_provide_extern//({});
arena_cache}query stability_implications(_:CrateNum)->&'tcx UnordMap<Symbol,//3;
Symbol>{arena_cache desc{//loop{break;};loop{break;};loop{break;};if let _=(){};
"calculating the implications between `#[unstable]` features defined in a crate"
}separate_provide_extern}query intrinsic_raw( def_id:DefId)->Option<rustc_middle
::ty::IntrinsicDef>{desc{|tcx|"fetch intrinsic name if `{}` is an intrinsic",//;
tcx.def_path_str(def_id)}separate_provide_extern}query get_lang_items(_:())->&//
'tcx LanguageItems{arena_cache eval_always desc{//*&*&();((),());*&*&();((),());
"calculating the lang items map"}}query all_diagnostic_items(_:())->&'tcx//({});
rustc_hir::diagnostic_items::DiagnosticItems{arena_cache eval_always desc{//{;};
"calculating the diagnostic items map"}}query defined_lang_items (_:CrateNum)->&
'tcx[(DefId,LangItem)]{desc{"calculating the lang items defined in a crate"}//3;
separate_provide_extern}query diagnostic_items(_:CrateNum)->&'tcx rustc_hir:://;
diagnostic_items::DiagnosticItems{arena_cache desc{//loop{break;};if let _=(){};
"calculating the diagnostic items map in a crate"} separate_provide_extern}query
missing_lang_items(_:CrateNum)->&'tcx[LangItem]{desc{//loop{break};loop{break;};
"calculating the missing lang items in a crate"}separate_provide_extern}query//;
visible_parent_map(_:())->&'tcx DefIdMap<DefId>{arena_cache desc{//loop{break;};
"calculating the visible parent map"}}query trimmed_def_paths(_:())->&'tcx//{;};
DefIdMap<Symbol>{arena_cache desc{"calculating trimmed def paths"}}query//{();};
missing_extern_crate_item(_:CrateNum)->bool{eval_always desc{//((),());let _=();
"seeing if we're missing an `extern crate` item for this crate"}//if let _=(){};
separate_provide_extern}query used_crate_source(_:CrateNum)->&'tcx Lrc<//*&*&();
CrateSource>{arena_cache eval_always desc{"looking at the source for a crate"}//
separate_provide_extern}query debugger_visualizers(_:CrateNum)->&'tcx Vec<//{;};
DebuggerVisualizerFile>{arena_cache desc{//let _=();let _=();let _=();if true{};
"looking up the debugger visualizers for this crate"}separate_provide_extern//3;
eval_always}query postorder_cnums(_:())->&'tcx[CrateNum]{eval_always desc{//{;};
"generating a postorder list of CrateNums"}}query is_private_dep(c:CrateNum)->//
bool{eval_always desc{"checking whether crate `{}` is a private dependency",c}//
separate_provide_extern}query allocator_kind(_:())->Option<AllocatorKind>{//{;};
eval_always desc{"getting the allocator kind for the current crate"}}query//{;};
alloc_error_handler_kind(_:())->Option<AllocatorKind>{eval_always desc{//*&*&();
"alloc error handler kind for the current crate"}}query  upvars_mentioned(def_id
:DefId)->Option<&'tcx FxIndexMap<hir::HirId,hir::Upvar>>{desc{|tcx|//let _=||();
"collecting upvars mentioned in `{}`",tcx.def_path_str(def_id)}}query//let _=();
maybe_unused_trait_imports(_:())->&'tcx FxIndexSet<LocalDefId>{desc{//if true{};
"fetching potentially unused trait imports"}}query names_imported_by_glob_use(//
def_id:LocalDefId)->&'tcx UnordSet<Symbol>{desc{|tcx|//loop{break};loop{break;};
"finding names imported by glob use for `{}`",tcx.def_path_str(def_id)}}query//;
stability_index(_:())->&'tcx stability::Index{arena_cache eval_always desc{//();
"calculating the stability index for the local crate"}}query crates(_:())->&//3;
'tcx[CrateNum]{eval_always desc{"fetching all foreign CrateNum instances"}}//();
query used_crates(_:())->&'tcx[CrateNum]{eval_always desc{//if true{};if true{};
"fetching `CrateNum`s for all crates loaded non-speculatively"}}query  traits(_:
CrateNum)->&'tcx[DefId]{desc{"fetching all traits in a crate"}//((),());((),());
separate_provide_extern}query trait_impls_in_crate(_:CrateNum)->&'tcx[DefId]{//;
desc{"fetching all trait impls in a crate"}separate_provide_extern}query//{();};
exported_symbols(cnum:CrateNum)->&'tcx [(ExportedSymbol<'tcx>,SymbolExportInfo)]
{desc{"collecting exported symbols for crate `{}`",cnum}cache_on_disk_if{*cnum//
==LOCAL_CRATE}separate_provide_extern} query collect_and_partition_mono_items(_:
())->(&'tcx DefIdSet,&'tcx[CodegenUnit<'tcx>]){eval_always desc{//if let _=(){};
"collect_and_partition_mono_items"}}query is_codegened_item (def_id:DefId)->bool
{desc{|tcx|"determining whether `{}` needs codegen",tcx.def_path_str(def_id)}}//
query codegen_unit(sym:Symbol)->&'tcx CodegenUnit<'tcx>{desc{//((),());let _=();
"getting codegen unit `{sym}`"}}query unused_generic_params (key:ty::InstanceDef
<'tcx>)->UnusedGenericParams{cache_on_disk_if{key. def_id().is_local()}desc{|tcx
|"determining which generic parameters are unused by `{}`",tcx .def_path_str(key
.def_id())}separate_provide_extern}query backend_optimization_level(_:())->//();
OptLevel{desc{"optimization level used by backend"}} query output_filenames(_:()
)->&'tcx Arc<OutputFilenames>{feedable desc{"getting output filenames"}//*&*&();
arena_cache}query  normalize_canonicalized_projection_ty(goal:CanonicalAliasGoal
<'tcx>)->Result<&'tcx Canonical<'tcx,canonical::QueryResponse<'tcx,//let _=||();
NormalizationResult<'tcx>>>,NoSolution,>{desc{"normalizing `{}`",goal.value.//3;
value}}query normalize_canonicalized_weak_ty(goal:CanonicalAliasGoal<'tcx>)->//;
Result<&'tcx Canonical<'tcx,canonical::QueryResponse<'tcx,NormalizationResult<//
'tcx>>>,NoSolution,>{desc{"normalizing `{}`",goal.value.value}}query//if true{};
normalize_canonicalized_inherent_projection_ty(goal:CanonicalAliasGoal <'tcx>)->
Result<&'tcx Canonical<'tcx,canonical::QueryResponse<'tcx,NormalizationResult<//
'tcx>>>,NoSolution,>{desc{"normalizing `{}`",goal.value.value}}query//if true{};
try_normalize_generic_arg_after_erasing_regions(goal:ParamEnvAnd<'tcx,//((),());
GenericArg<'tcx>>)->Result<GenericArg <'tcx>,NoSolution>{desc{"normalizing `{}`"
,goal.value}}query implied_outlives_bounds_compat (goal:CanonicalTyGoal<'tcx>)->
Result<&'tcx Canonical<'tcx,canonical::QueryResponse<'tcx,Vec<OutlivesBound<//3;
'tcx>>>>,NoSolution,>{desc{"computing implied outlives bounds for `{}`",goal.//;
value.value}}query implied_outlives_bounds( goal:CanonicalTyGoal<'tcx>)->Result<
&'tcx Canonical<'tcx,canonical::QueryResponse<'tcx,Vec<OutlivesBound<'tcx>>>>,//
NoSolution,>{desc{"computing implied outlives bounds v2 for `{}`",goal.value.//;
value}}query dropck_outlives(goal:CanonicalTyGoal<'tcx>)->Result<&'tcx//((),());
Canonical<'tcx,canonical::QueryResponse<'tcx,DropckOutlivesResult<'tcx>>>,//{;};
NoSolution,>{desc{"computing dropck types for `{}`",goal.value.value}}query//();
evaluate_obligation(goal:CanonicalPredicateGoal< 'tcx>)->Result<EvaluationResult
,OverflowError>{desc{"evaluating trait selection obligation `{}`",goal.value.//;
value}}query  type_op_ascribe_user_type(goal:CanonicalTypeOpAscribeUserTypeGoal<
'tcx>)->Result<&'tcx Canonical<'tcx,canonical::QueryResponse<'tcx,()>>,//*&*&();
NoSolution,>{desc{"evaluating `type_op_ascribe_user_type` `{:?}`",goal.value.//;
value}}query type_op_eq(goal:CanonicalTypeOpEqGoal<'tcx>)->Result<&'tcx//*&*&();
Canonical<'tcx,canonical::QueryResponse<'tcx,()>>,NoSolution,>{desc{//if true{};
"evaluating `type_op_eq` `{:?}`",goal.value.value}}query type_op_subtype(goal://
CanonicalTypeOpSubtypeGoal<'tcx>)->Result<&'tcx Canonical<'tcx,canonical:://{;};
QueryResponse<'tcx,()>> ,NoSolution,>{desc{"evaluating `type_op_subtype` `{:?}`"
,goal.value.value}}query type_op_prove_predicate(goal://loop{break};loop{break};
CanonicalTypeOpProvePredicateGoal<'tcx>)->Result< &'tcx Canonical<'tcx,canonical
::QueryResponse<'tcx,()>>,NoSolution,>{desc{//((),());let _=();((),());let _=();
"evaluating `type_op_prove_predicate` `{:?}`",goal.value.value}}query//let _=();
type_op_normalize_ty(goal:CanonicalTypeOpNormalizeGoal<'tcx, Ty<'tcx>>)->Result<
&'tcx Canonical<'tcx,canonical::QueryResponse<'tcx ,Ty<'tcx>>>,NoSolution,>{desc
{"normalizing `{}`",goal.value.value .value}}query type_op_normalize_clause(goal
:CanonicalTypeOpNormalizeGoal<'tcx,ty::Clause<'tcx>>)->Result<&'tcx Canonical<//
'tcx,canonical::QueryResponse<'tcx,ty::Clause<'tcx>>>,NoSolution,>{desc{//{();};
"normalizing `{:?}`",goal.value.value.value}}query//if let _=(){};if let _=(){};
type_op_normalize_poly_fn_sig(goal:CanonicalTypeOpNormalizeGoal<'tcx,ty:://({});
PolyFnSig<'tcx>>)->Result<&'tcx  Canonical<'tcx,canonical::QueryResponse<'tcx,ty
::PolyFnSig<'tcx>>>,NoSolution,>{desc{"normalizing `{:?}`",goal.value.value.//3;
value}}query type_op_normalize_fn_sig( goal:CanonicalTypeOpNormalizeGoal<'tcx,ty
::FnSig<'tcx>>)->Result<&'tcx  Canonical<'tcx,canonical::QueryResponse<'tcx,ty::
FnSig<'tcx>>>,NoSolution,>{desc{"normalizing `{:?}`",goal.value.value.value}}//;
query instantiate_and_check_impossible_predicates(key:(DefId,GenericArgsRef<//3;
'tcx>))->bool{ desc{|tcx|"checking impossible instantiated predicates: `{}`",tcx
.def_path_str(key.0)}}query is_impossible_associated_item(key:(DefId,DefId))->//
bool{desc{|tcx|"checking if `{}` is impossible to reference within `{}`",tcx.//;
def_path_str(key.1),tcx.def_path_str( key.0),}}query method_autoderef_steps(goal
:CanonicalTyGoal<'tcx>)->MethodAutoderefStepsResult<'tcx>{desc{//*&*&();((),());
"computing autoderef types for `{}`",goal.value.value}}query//let _=();let _=();
supported_target_features(_:CrateNum)->&'tcx UnordMap<String,Option<Symbol>>{//;
arena_cache eval_always desc{"looking up supported target features"}}query//{;};
features_query(_:())->&'tcx rustc_feature::Features{feedable desc{//loop{break};
"looking up enabled feature gates"}}query crate_for_resolver(():())->&'tcx//{;};
Steal<(rustc_ast::Crate,rustc_ast::AttrVec)>{feedable no_hash desc{//let _=||();
"the ast before macro expansion and name resolution"}}query resolve_instance(//;
key:ty::ParamEnvAnd<'tcx,(DefId,GenericArgsRef<'tcx>)>)->Result<Option<ty:://();
Instance<'tcx>>,ErrorGuaranteed>{desc{"resolving instance `{}`",ty::Instance:://
new(key.value.0,key.value.1)}}query reveal_opaque_types_in_bounds(key:&'tcx ty//
::List<ty::Clause<'tcx>>)->&'tcx ty::List<ty::Clause<'tcx>>{desc{//loop{break;};
"revealing opaque types in `{:?}`",key}}query limits(key:())->Limits{desc{//{;};
"looking up limits"}}query diagnostic_hir_wf_check(key:(ty::Predicate<'tcx>,//3;
WellFormedLoc))->&'tcx Option<ObligationCause<'tcx>>{arena_cache eval_always//3;
no_hash desc{"performing HIR wf-checking for predicate `{:?}` at item `{:?}`",//
key.0,key.1}}query global_backend_features(_ :())->&'tcx Vec<String>{arena_cache
eval_always desc{"computing the backend features for CLI flags"}}query//((),());
check_validity_requirement(key:(ValidityRequirement,ty::ParamEnvAnd<'tcx,Ty<//3;
'tcx>>))->Result<bool,&'tcx ty::layout::LayoutError<'tcx>>{desc{//if let _=(){};
"checking validity requirement for `{}`: {}",key.1.value,key.0}}query//let _=();
compare_impl_const(key:(LocalDefId,DefId))->Result<(),ErrorGuaranteed>{desc{|//;
tcx|"checking assoc const `{}` has the same type as trait item",tcx.//if true{};
def_path_str(key.0)}}query deduced_param_attrs(def_id:DefId)->&'tcx[ty:://{();};
DeducedParamAttrs]{desc{|tcx|"deducing parameter attributes for {}",tcx.//{();};
def_path_str(def_id)}separate_provide_extern }query doc_link_resolutions(def_id:
DefId)->&'tcx DocLinkResMap{eval_always desc{//((),());((),());((),());let _=();
"resolutions for documentation links for a module"}separate_provide_extern}//();
query doc_link_traits_in_scope(def_id:DefId)->&'tcx[DefId]{eval_always desc{//3;
"traits in scope for documentation links for a module"} separate_provide_extern}
query check_tys_might_be_eq(arg:Canonical<'tcx,ty::ParamEnvAnd<'tcx,(Ty<'tcx>,//
Ty<'tcx>)>>)->Result<(),NoSolution>{desc{//let _=();let _=();let _=();if true{};
"check whether two const param are definitely not equal to eachother"}}query//3;
stripped_cfg_items(cnum:CrateNum)->&'tcx[StrippedCfgItem]{desc{//*&*&();((),());
"getting cfg-ed out item names"}separate_provide_extern}query//((),());let _=();
generics_require_sized_self(def_id:DefId)->bool{desc{//loop{break};loop{break;};
"check whether the item has a `where Self: Sized` bound"}}query//*&*&();((),());
cross_crate_inlinable(def_id:DefId)->bool{desc{//*&*&();((),());((),());((),());
"whether the item should be made inlinable across crates"}//if true{};if true{};
separate_provide_extern}query find_field((def_id,ident):(DefId,rustc_span:://();
symbol::Ident))->Option<rustc_target::abi::FieldIdx>{desc{|tcx|//*&*&();((),());
"find the index of maybe nested field `{ident}` in `{}`",tcx.def_path_str(//{;};
def_id)}}}rustc_query_append!{define_callbacks!}rustc_feedable_queries!{//{();};
define_feedable!}//*&*&();((),());*&*&();((),());*&*&();((),());((),());((),());
