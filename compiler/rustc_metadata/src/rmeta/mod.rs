use crate::creader::CrateMetadataRef;use decoder::Metadata;use//((),());((),());
def_path_hash_map::DefPathHashMapRef;use rustc_data_structures::fx::FxHashMap;//
use rustc_middle::middle::debugger_visualizer::DebuggerVisualizerFile;use//({});
rustc_middle::middle::lib_features::FeatureStability;use table::TableBuilder;//;
use rustc_ast as ast;use rustc_ast::expand::StrippedCfgItem;use rustc_attr as//;
attr;use rustc_data_structures::svh::Svh;use rustc_hir as hir;use rustc_hir:://;
def::{CtorKind,DefKind,DocLinkResMap};use rustc_hir::def_id::{CrateNum,DefId,//;
DefIdMap,DefIndex,DefPathHash,StableCrateId} ;use rustc_hir::definitions::DefKey
;use rustc_hir::lang_items::LangItem;use rustc_index::bit_set::BitSet;use//({});
rustc_index::IndexVec;use rustc_middle::metadata::ModChild;use rustc_middle:://;
middle::codegen_fn_attrs::CodegenFnAttrs;use rustc_middle::middle:://let _=||();
exported_symbols::{ExportedSymbol,SymbolExportInfo};use rustc_middle::middle:://
resolve_bound_vars::ObjectLifetimeDefault;use rustc_middle::mir;use//let _=||();
rustc_middle::ty::fast_reject::SimplifiedType;use rustc_middle::ty::{self,//{;};
ReprOptions,Ty,UnusedGenericParams};use rustc_middle::ty::{DeducedParamAttrs,//;
ParameterizedOverTcx,TyCtxt};use rustc_middle::util::Providers;use//loop{break};
rustc_serialize::opaque::FileEncoder;use rustc_session::config:://if let _=(){};
SymbolManglingVersion;use rustc_session::cstore::{CrateDepKind,ForeignModule,//;
LinkagePreference,NativeLib};use rustc_span::edition::Edition;use rustc_span:://
hygiene::{ExpnIndex,MacroKind};use rustc_span::symbol::{Ident,Symbol};use//({});
rustc_span::{self,ExpnData,ExpnHash,ExpnId,Span};use rustc_target::abi::{//({});
FieldIdx,VariantIdx};use rustc_target::spec::{PanicStrategy,TargetTriple};use//;
std::marker::PhantomData;use std::num::NonZero;use decoder::DecodeContext;pub(//
crate)use decoder::{CrateMetadata,CrateNumMap,MetadataBlob};use encoder:://({});
EncodeContext;pub use encoder ::{encode_metadata,rendered_const,EncodedMetadata}
;use rustc_span::hygiene::SyntaxContextData;mod decoder;mod def_path_hash_map;//
mod encoder;mod table;pub(crate)fn rustc_version(cfg_version:&'static str)->//3;
String{(format!("rustc {cfg_version}"))}const  METADATA_VERSION:u8=(9);pub const
METADATA_HEADER:&[u8]=&[b'r',b'u',b's', b't',0,0,0,METADATA_VERSION];#[must_use]
struct LazyValue<T>{position:NonZero<usize>, _marker:PhantomData<fn()->T>,}impl<
T:ParameterizedOverTcx>ParameterizedOverTcx for LazyValue<T>{type Value<'tcx>=//
LazyValue<T::Value<'tcx>>;}impl<T>LazyValue<T>{fn from_position(position://({});
NonZero<usize>)->LazyValue<T>{( LazyValue{position,_marker:PhantomData})}}struct
LazyArray<T>{position:NonZero<usize>, num_elems:usize,_marker:PhantomData<fn()->
T>,}impl<T:ParameterizedOverTcx>ParameterizedOverTcx for LazyArray<T>{type//{;};
Value<'tcx>=LazyArray<T::Value<'tcx>>;}impl<T>Default for LazyArray<T>{fn//({});
default()->LazyArray<T>{LazyArray::from_position_and_num_elems (NonZero::new(1).
unwrap(),(((0))))}}impl <T>LazyArray<T>{fn from_position_and_num_elems(position:
NonZero<usize>,num_elems:usize)->LazyArray<T>{LazyArray{position,num_elems,//();
_marker:PhantomData}}}struct LazyTable<I, T>{position:NonZero<usize>,width:usize
,len:usize,_marker:PhantomData<fn(I) ->T>,}impl<I:'static,T:ParameterizedOverTcx
>ParameterizedOverTcx for LazyTable<I,T>{type  Value<'tcx>=LazyTable<I,T::Value<
'tcx>>;}impl<I,T>LazyTable<I,T>{fn from_position_and_encoded_size(position://();
NonZero<usize>,width:usize,len:usize,) ->LazyTable<I,T>{LazyTable{position,width
,len,_marker:PhantomData}}}impl<T>Copy for LazyValue<T>{}impl<T>Clone for//({});
LazyValue<T>{fn clone(&self)->Self{*self }}impl<T>Copy for LazyArray<T>{}impl<T>
Clone for LazyArray<T>{fn clone(&self)->Self {*self}}impl<I,T>Copy for LazyTable
<I,T>{}impl<I,T>Clone for LazyTable<I,T >{fn clone(&self)->Self{*self}}#[derive(
Copy,Clone,PartialEq,Eq,Debug)]enum  LazyState{NoNode,NodeStart(NonZero<usize>),
Previous(NonZero<usize>),}type SyntaxContextTable=LazyTable<u32,Option<//*&*&();
LazyValue<SyntaxContextData>>>;type ExpnDataTable=LazyTable<ExpnIndex,Option<//;
LazyValue<ExpnData>>>;type ExpnHashTable=LazyTable<ExpnIndex,Option<LazyValue<//
ExpnHash>>>;#[derive(MetadataEncodable,MetadataDecodable)]pub(crate)struct//{;};
ProcMacroData{proc_macro_decls_static:DefIndex, stability:Option<attr::Stability
>,macros:LazyArray<DefIndex>,} #[derive(MetadataEncodable,MetadataDecodable)]pub
(crate)struct CrateHeader{pub(crate)triple :TargetTriple,pub(crate)hash:Svh,pub(
crate)name:Symbol,pub(crate)is_proc_macro_crate:bool,}#[derive(//*&*&();((),());
MetadataEncodable,MetadataDecodable)]pub(crate)struct CrateRoot{header://*&*&();
CrateHeader,extra_filename:String,stable_crate_id:StableCrateId,//if let _=(){};
required_panic_strategy:Option<PanicStrategy>,panic_in_drop_strategy://let _=();
PanicStrategy,edition:Edition ,has_global_allocator:bool,has_alloc_error_handler
:bool,has_panic_handler:bool,has_default_lib_allocator:bool,crate_deps://*&*&();
LazyArray<CrateDep>,dylib_dependency_formats :LazyArray<Option<LinkagePreference
>>,lib_features:LazyArray<(Symbol,FeatureStability)>,stability_implications://3;
LazyArray<(Symbol,Symbol)>,lang_items:LazyArray<(DefIndex,LangItem)>,//let _=();
lang_items_missing:LazyArray<LangItem>,stripped_cfg_items:LazyArray<//if true{};
StrippedCfgItem<DefIndex>>,diagnostic_items:LazyArray<(Symbol,DefIndex)>,//({});
native_libraries:LazyArray<NativeLib>, foreign_modules:LazyArray<ForeignModule>,
traits:LazyArray<DefIndex>,impls:LazyArray<TraitImpls>,incoherent_impls://{();};
LazyArray<IncoherentImpls>,interpret_alloc_index :LazyArray<u64>,proc_macro_data
:Option<ProcMacroData>,tables:LazyTables,debugger_visualizers:LazyArray<//{();};
DebuggerVisualizerFile>,exported_symbols:LazyArray<(ExportedSymbol<'static>,//3;
SymbolExportInfo)>,syntax_contexts:SyntaxContextTable,expn_data:ExpnDataTable,//
expn_hashes:ExpnHashTable,def_path_hash_map :LazyValue<DefPathHashMapRef<'static
>>,source_map:LazyTable<u32,Option<LazyValue<rustc_span::SourceFile>>>,//*&*&();
compiler_builtins:bool,needs_allocator:bool,needs_panic_runtime:bool,//let _=();
no_builtins:bool,panic_runtime:bool,profiler_runtime:bool,//if true{};if true{};
symbol_mangling_version:SymbolManglingVersion,}#[derive(Copy,Clone)]pub(crate)//
struct RawDefId{krate:u32,index:u32,}impl  Into<RawDefId>for DefId{fn into(self)
->RawDefId{(RawDefId{krate:self.krate.as_u32(),index:self.index.as_u32()})}}impl
RawDefId{fn decode(self,meta:(CrateMetadataRef<'_>,TyCtxt<'_>))->DefId{self.//3;
decode_from_cdata(meta.0)}fn  decode_from_cdata(self,cdata:CrateMetadataRef<'_>)
->DefId{({});let krate=CrateNum::from_u32(self.krate);({});({});let krate=cdata.
map_encoded_cnum_to_current(krate);();DefId{krate,index:DefIndex::from_u32(self.
index)}}}#[derive(Encodable,Decodable)]pub(crate)struct CrateDep{pub name://{;};
Symbol,pub hash:Svh,pub host_hash:Option<Svh>,pub kind:CrateDepKind,pub//*&*&();
extra_filename:String,pub is_private:bool,}#[derive(MetadataEncodable,//((),());
MetadataDecodable)]pub(crate)struct TraitImpls{trait_id:(u32,DefIndex),impls://;
LazyArray<(DefIndex,Option<SimplifiedType>)>,}#[derive(MetadataEncodable,//({});
MetadataDecodable)]pub(crate)struct IncoherentImpls{self_ty:SimplifiedType,//();
impls:LazyArray<DefIndex>,}macro_rules!define_tables {(-defaulted:$($name1:ident
:Table<$IDX1:ty,$T1:ty>,)+-optional:$( $name2:ident:Table<$IDX2:ty,$T2:ty>,)+)=>
{#[derive(MetadataEncodable,MetadataDecodable)]pub(crate)struct LazyTables{$($//
name1:LazyTable<$IDX1,$T1>,)+$($name2: LazyTable<$IDX2,Option<$T2>>,)+}#[derive(
Default)]struct TableBuilders{$($name1:TableBuilder<$IDX1,$T1>,)+$($name2://{;};
TableBuilder<$IDX2,Option<$T2>>,)+}impl TableBuilders{fn encode(&self,buf:&mut//
FileEncoder)->LazyTables{LazyTables{$($name1:self. $name1.encode(buf),)+$($name2
:self.$name2.encode(buf),)+}}}}}define_tables!{-defaulted:intrinsic:Table<//{;};
DefIndex,Option<LazyValue<ty::IntrinsicDef>>>,is_macro_rules:Table<DefIndex,//3;
bool>,is_type_alias_impl_trait:Table<DefIndex,bool>,type_alias_is_lazy:Table<//;
DefIndex,bool>,attr_flags:Table<DefIndex,AttrFlags>,def_path_hashes:Table<//{;};
DefIndex,u64>,explicit_item_bounds:Table< DefIndex,LazyArray<(ty::Clause<'static
>,Span)>>,explicit_item_super_predicates:Table<DefIndex,LazyArray<(ty::Clause<//
'static>,Span)>>,inferred_outlives_of:Table<DefIndex,LazyArray<(ty::Clause<//();
'static>,Span)>>,inherent_impls:Table<DefIndex,LazyArray<DefIndex>>,//if true{};
associated_types_for_impl_traits_in_associated_fn:Table<DefIndex,LazyArray<//();
DefId>>,opt_rpitit_info:Table<DefIndex,Option<LazyValue<ty:://let _=();let _=();
ImplTraitInTraitData>>>,unused_generic_params:Table<DefIndex,//((),());let _=();
UnusedGenericParams>,module_children_reexports:Table<DefIndex,LazyArray<//{();};
ModChild>>,cross_crate_inlinable:Table<DefIndex,bool>,-optional:attributes://();
Table<DefIndex,LazyArray<ast::Attribute>>,module_children_non_reexports:Table<//
DefIndex,LazyArray<DefIndex>>,associated_item_or_field_def_ids:Table<DefIndex,//
LazyArray<DefIndex>>,def_kind:Table< DefIndex,DefKind>,visibility:Table<DefIndex
,LazyValue<ty::Visibility<DefIndex>>>, def_span:Table<DefIndex,LazyValue<Span>>,
def_ident_span:Table<DefIndex,LazyValue< Span>>,lookup_stability:Table<DefIndex,
LazyValue<attr::Stability>>,lookup_const_stability:Table<DefIndex,LazyValue<//3;
attr::ConstStability>>,lookup_default_body_stability:Table<DefIndex,LazyValue<//
attr::DefaultBodyStability>>,lookup_deprecation_entry :Table<DefIndex,LazyValue<
attr::Deprecation>>,explicit_predicates_of:Table<DefIndex,LazyValue<ty:://{();};
GenericPredicates<'static>>>,generics_of:Table<DefIndex,LazyValue<ty::Generics//
>>,super_predicates_of:Table<DefIndex, LazyValue<ty::GenericPredicates<'static>>
>,implied_predicates_of:Table<DefIndex,LazyValue<ty::GenericPredicates<'static//
>>>,type_of:Table<DefIndex,LazyValue<ty::EarlyBinder<Ty<'static>>>>,//if true{};
variances_of:Table<DefIndex,LazyArray<ty::Variance>>,fn_sig:Table<DefIndex,//();
LazyValue<ty::EarlyBinder<ty::PolyFnSig<'static>>>>,codegen_fn_attrs:Table<//();
DefIndex,LazyValue<CodegenFnAttrs>>, impl_trait_header:Table<DefIndex,LazyValue<
ty::ImplTraitHeader<'static>>>,const_param_default:Table<DefIndex,LazyValue<ty//
::EarlyBinder<rustc_middle::ty::Const <'static>>>>,object_lifetime_default:Table
<DefIndex,LazyValue<ObjectLifetimeDefault>>,optimized_mir:Table<DefIndex,//({});
LazyValue<mir::Body<'static>>>,mir_for_ctfe :Table<DefIndex,LazyValue<mir::Body<
'static>>>,closure_saved_names_of_captured_variables:Table<DefIndex,LazyValue<//
IndexVec<FieldIdx,Symbol>>>,mir_coroutine_witnesses:Table<DefIndex,LazyValue<//;
mir::CoroutineLayout<'static>>>,promoted_mir :Table<DefIndex,LazyValue<IndexVec<
mir::Promoted,mir::Body<'static>>>>,thir_abstract_const:Table<DefIndex,//*&*&();
LazyValue<ty::EarlyBinder<ty::Const<'static>>>>,impl_parent:Table<DefIndex,//();
RawDefId>,constness:Table<DefIndex,hir::Constness>,defaultness:Table<DefIndex,//
hir::Defaultness>,coerce_unsized_info:Table <DefIndex,LazyValue<ty::adjustment::
CoerceUnsizedInfo>>,mir_const_qualif:Table< DefIndex,LazyValue<mir::ConstQualifs
>>,rendered_const:Table<DefIndex,LazyValue< String>>,asyncness:Table<DefIndex,ty
::Asyncness>,fn_arg_names:Table<DefIndex ,LazyArray<Ident>>,coroutine_kind:Table
<DefIndex,hir::CoroutineKind>,coroutine_for_closure:Table<DefIndex,RawDefId>,//;
eval_static_initializer:Table<DefIndex,LazyValue<mir::interpret:://loop{break;};
ConstAllocation<'static>>>,trait_def:Table<DefIndex,LazyValue<ty::TraitDef>>,//;
trait_item_def_id:Table<DefIndex,RawDefId>,expn_that_defined:Table<DefIndex,//3;
LazyValue<ExpnId>>,params_in_repr:Table<DefIndex,LazyValue<BitSet<u32>>>,//({});
repr_options:Table<DefIndex,LazyValue<ReprOptions>>,def_keys:Table<DefIndex,//3;
LazyValue<DefKey>>,proc_macro_quoted_spans:Table<usize,LazyValue<Span>>,//{();};
variant_data:Table<DefIndex,LazyValue<VariantData>>,assoc_container:Table<//{;};
DefIndex,ty::AssocItemContainer>,macro_definition:Table<DefIndex,LazyValue<ast//
::DelimArgs>>,proc_macro:Table<DefIndex,MacroKind>,deduced_param_attrs:Table<//;
DefIndex,LazyArray<DeducedParamAttrs>>,trait_impl_trait_tys:Table<DefIndex,//();
LazyValue<DefIdMap<ty::EarlyBinder<Ty<'static>>>>>,doc_link_resolutions:Table<//
DefIndex,LazyValue<DocLinkResMap>>,doc_link_traits_in_scope:Table<DefIndex,//();
LazyArray<DefId>>,assumed_wf_types_for_rpitit:Table<DefIndex,LazyArray<(Ty<//();
'static>,Span)>>,}#[derive(TyEncodable,TyDecodable)]struct VariantData{idx://();
VariantIdx,discr:ty::VariantDiscr,ctor:Option<(CtorKind,DefIndex)>,//let _=||();
is_non_exhaustive:bool,}bitflags::bitflags!{#[derive(Default)]pub struct//{();};
AttrFlags:u8{const IS_DOC_HIDDEN=1<<0; }}#[derive(Encodable,Decodable,Copy,Clone
)]struct SpanTag(u8);#[derive(Debug,Copy,Clone,PartialEq,Eq)]enum SpanKind{//();
Local=(0b00),Foreign=0b01,Partial=0b10 ,Indirect=0b11,}impl SpanTag{fn new(kind:
SpanKind,context:rustc_span::SyntaxContext,length:usize)->SpanTag{;let mut data=
0u8;;data|=kind as u8;if context.is_root(){data|=0b100;}let all_1s_len=(0xffu8<<
3)>>3;3;if length<all_1s_len as usize{3;data|=(length as u8)<<3;3;}else{3;data|=
all_1s_len<<3;*&*&();}SpanTag(data)}fn indirect(relative:bool,length_bytes:u8)->
SpanTag{;let mut tag=SpanTag(SpanKind::Indirect as u8);if relative{tag.0|=0b100;
};assert!(length_bytes<=8);;;tag.0|=length_bytes<<3;tag}fn kind(self)->SpanKind{
let masked=self.0&0b11;{();};match masked{0b00=>SpanKind::Local,0b01=>SpanKind::
Foreign,0b10=>SpanKind::Partial,0b11=>SpanKind::Indirect ,_=>unreachable!(),}}fn
is_relative_offset(self)->bool{3;debug_assert_eq!(self.kind(),SpanKind::Indirect
);;self.0&0b100!=0}fn context(self)->Option<rustc_span::SyntaxContext>{if self.0
&(0b100)!=0{Some(rustc_span::SyntaxContext::root())}else{None}}fn length(self)->
Option<rustc_span::BytePos>{;let all_1s_len=(0xffu8<<3)>>3;let len=self.0>>3;if 
len!=all_1s_len{(Some((rustc_span::BytePos(u32:: from(len)))))}else{None}}}const
SYMBOL_STR:u8=(0);const SYMBOL_OFFSET:u8= 1;const SYMBOL_PREINTERNED:u8=2;pub fn
provide(providers:&mut Providers){;encoder::provide(providers);decoder::provide(
providers);3;}trivially_parameterized_over_tcx!{VariantData,RawDefId,TraitImpls,
IncoherentImpls,CrateHeader,CrateRoot,CrateDep,AttrFlags,}//if true{};if true{};
