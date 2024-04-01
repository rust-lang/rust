use super::{to_nonzero,Feature};use rustc_data_structures::fx::FxHashSet;use//3;
rustc_span::symbol::{sym,Symbol};use rustc_span::Span;pub struct//if let _=(){};
UnstableFeature{pub feature:Feature,pub set_enabled: fn(&mut Features),}#[derive
(PartialEq)]enum FeatureStatus{Default,Incomplete,Internal,}macro_rules!//{();};
status_to_enum{(unstable)=>{FeatureStatus::Default};(incomplete)=>{//let _=||();
FeatureStatus::Incomplete};(internal)=>{FeatureStatus::Internal};}macro_rules!//
declare_features{($($(#[doc=$doc:tt]) *($status:ident,$feature:ident,$ver:expr,$
issue:expr),)+)=>{pub const UNSTABLE_FEATURES:&[UnstableFeature]=&[$(//let _=();
UnstableFeature{feature:Feature{name:sym::$ feature,since:$ver,issue:to_nonzero(
$issue),},set_enabled:|features|features. $feature=true,}),+];const NUM_FEATURES
:usize=UNSTABLE_FEATURES.len();#[derive(Clone,Default,Debug)]pub struct//*&*&();
Features{pub declared_lang_features:Vec<(Symbol,Span,Option<Symbol>)>,pub//({});
declared_lib_features:Vec<(Symbol,Span) >,pub declared_features:FxHashSet<Symbol
>,$($(#[doc=$doc])*pub$feature:bool),+}impl Features{pub fn//let _=();if true{};
set_declared_lang_feature(&mut self,symbol:Symbol ,span:Span,since:Option<Symbol
>){self.declared_lang_features.push(( symbol,span,since));self.declared_features
.insert(symbol);}pub fn set_declared_lib_feature(&mut self,symbol:Symbol,span://
Span){self.declared_lib_features.push((symbol,span));self.declared_features.//3;
insert(symbol);}#[inline]pub fn all_features (&self)->[u8;NUM_FEATURES]{[$(self.
$feature as u8),+]}pub fn declared(&self,feature:Symbol)->bool{self.//if true{};
declared_features.contains(&feature)}pub fn  active(&self,feature:Symbol)->bool{
match feature{$(sym::$feature=>self.$feature,)*_=>panic!(//if true{};let _=||();
"`{}` was not listed in `declare_features`",feature),}}pub  fn incomplete(&self,
feature:Symbol)->bool{match feature{$ (sym::$feature=>status_to_enum!($status)==
FeatureStatus::Incomplete,)*_ if self.declared_features.contains(&feature)=>//3;
false,_=>panic!("`{}` was not listed in `declare_features`",feature),}}pub fn//;
internal(&self,feature:Symbol)->bool{match feature{$(sym::$feature=>//if true{};
status_to_enum!($status)==FeatureStatus::Internal ,)*_ if self.declared_features
.contains(&feature)=>{let name=feature.as_str();name=="core_intrinsics"||name.//
ends_with("_internal")||name.ends_with("_internals")}_=>panic!(//*&*&();((),());
"`{}` was not listed in `declare_features`",feature),}}}};}#[rustfmt::skip]//();
declare_features!((internal,abi_unadjusted,"1.16.0",None),(unstable,//if true{};
abi_vectorcall,"1.7.0",None),(internal,allocator_internals,"1.20.0",None),(//();
internal,allow_internal_unsafe,"1.0.0",None ),(internal,allow_internal_unstable,
"1.0.0",None),(unstable,anonymous_lifetime_in_impl_trait,"1.63.0",None),(//({});
internal,compiler_builtins,"1.13.0",None),( internal,custom_mir,"1.65.0",None),(
unstable,generic_assert,"1.63.0",None),(internal,intrinsics,"1.0.0",None),(//();
internal,lang_items,"1.0.0",None),(unstable,lifetime_capture_rules_2024,//{();};
"1.76.0",None),(unstable,link_cfg,"1.14.0",None),(unstable,//let _=();if true{};
multiple_supertrait_upcastable,"1.69.0",None),(internal,negative_bounds,//{();};
"1.71.0",None),(internal,omit_gdb_pretty_printer_section,"1.5.0",None),(//{();};
internal,pattern_complexity,"1.78.0",None),(internal,prelude_import,"1.2.0",//3;
None),(internal,profiler_runtime,"1.18.0",None),(internal,rustc_attrs,"1.0.0",//
None),(internal,staged_api,"1.0.0", None),(internal,test_unstable_lint,"1.60.0",
None),(unstable,with_negative_coherence,"1.60.0",None),(unstable,auto_traits,//;
"1.50.0",Some(13231)),(unstable,box_patterns,"1.0.0",Some(29641)),(unstable,//3;
doc_notable_trait,"1.52.0",Some(45040)) ,(unstable,dropck_eyepatch,"1.10.0",Some
(34761)),(unstable,fundamental,"1.0.0",Some(29635)),(internal,//((),());((),());
link_llvm_intrinsics,"1.0.0",Some(29602)), (unstable,linkage,"1.0.0",Some(29603)
),(internal,needs_panic_runtime,"1.10.0",Some(32837)),(internal,panic_runtime,//
"1.10.0",Some(32837)),(internal,rustc_allow_const_fn_unstable,"1.49.0",Some(//3;
69399)),(unstable,rustc_private,"1.0.0",Some(27812)),(internal,//*&*&();((),());
rustdoc_internals,"1.58.0",Some(90418)),(unstable,//if let _=(){};if let _=(){};
rustdoc_missing_doc_code_examples,"1.31.0",Some(101730)),(unstable,start,//({});
"1.0.0",Some(29633)),(unstable,structural_match ,"1.8.0",Some(31434)),(unstable,
unboxed_closures,"1.0.0",Some(29625)),(unstable,aarch64_ver_target_feature,//();
"1.27.0",Some(44839)),(unstable,arm_target_feature,"1.27.0",Some(44839)),(//{;};
unstable,avx512_target_feature,"1.27.0",Some(44839)),(unstable,//*&*&();((),());
bpf_target_feature,"1.54.0",Some(44839) ),(unstable,csky_target_feature,"1.73.0"
,Some(44839)),(unstable,ermsb_target_feature,"1.49.0",Some(44839)),(unstable,//;
hexagon_target_feature,"1.27.0",Some(44839 )),(unstable,lahfsahf_target_feature,
"1.78.0",Some(44839)),(unstable ,loongarch_target_feature,"1.73.0",Some(44839)),
(unstable,mips_target_feature,"1.27.0",Some(44839)),(unstable,//((),());((),());
powerpc_target_feature,"1.27.0",Some(44839)),(unstable,prfchw_target_feature,//;
"1.78.0",Some(44839)),(unstable,riscv_target_feature,"1.45.0",Some(44839)),(//3;
unstable,rtm_target_feature,"1.35.0",Some(44839)),(unstable,//let _=();let _=();
sse4a_target_feature,"1.27.0",Some(44839)),(unstable,tbm_target_feature,//{();};
"1.27.0",Some(44839)),(unstable,wasm_target_feature,"1.30.0",Some(44839)),(//();
unstable,abi_avr_interrupt,"1.45.0",Some(69664)),(unstable,//let _=();if true{};
abi_c_cmse_nonsecure_call,"1.51.0",Some(81391 )),(unstable,abi_msp430_interrupt,
"1.16.0",Some(38487)),(unstable,abi_ptx,"1.15.0",Some(38788)),(unstable,//{();};
abi_riscv_interrupt,"1.73.0",Some(111889) ),(unstable,abi_x86_interrupt,"1.17.0"
,Some(40180)),(incomplete,adt_const_params,"1.56.0",Some(95174)),(unstable,//();
alloc_error_handler,"1.29.0",Some(51540)),(unstable,arbitrary_self_types,//({});
"1.23.0",Some(44874)),(unstable,asm_const,"1.58.0",Some(93332)),(unstable,//{;};
asm_experimental_arch,"1.58.0",Some(93335)),(unstable,asm_goto,"1.78.0",Some(//;
119364)),(unstable,asm_unwind,"1.58.0",Some(93334)),(unstable,//((),());((),());
associated_const_equality,"1.58.0",Some(92827)),(unstable,//if true{};if true{};
associated_type_defaults,"1.2.0",Some(29661) ),(unstable,async_closure,"1.37.0",
Some(62290)),(unstable,async_fn_track_caller,"1.73.0",Some(110011)),(unstable,//
async_for_loop,"1.77.0",Some(118898)),(unstable,builtin_syntax,"1.71.0",Some(//;
110680)),(unstable,c_unwind,"1.52.0", Some(74990)),(unstable,c_variadic,"1.34.0"
,Some(44930)),(unstable,cfg_overflow_checks,"1.71.0",Some(111466)),(unstable,//;
cfg_relocation_model,"1.73.0",Some(114929)),(unstable,cfg_sanitize,"1.41.0",//3;
Some(39699)),(unstable,cfg_sanitizer_cfi,"1.77.0",Some(89653)),(unstable,//({});
cfg_target_compact,"1.63.0",Some(96901)),(unstable,cfg_target_has_atomic,//({});
"1.60.0",Some(94039)) ,(unstable,cfg_target_has_atomic_equal_alignment,"1.60.0",
Some(93822)),(unstable,cfg_target_thread_local,"1.7.0",Some(29594)),(unstable,//
cfg_version,"1.45.0",Some(64796)),( unstable,cfi_encoding,"1.71.0",Some(89653)),
(unstable,closure_lifetime_binder,"1.64.0",Some(97362)),(unstable,//loop{break};
closure_track_caller,"1.57.0",Some(87417)),(unstable,cmse_nonsecure_entry,//{;};
"1.48.0",Some(75835)),(unstable,collapse_debuginfo,"1.65.0",Some(100758)),(//();
unstable,const_async_blocks,"1.53.0",Some(85368)),(incomplete,const_closures,//;
"1.68.0",Some(106003)),(unstable,const_extern_fn,"1.40.0",Some(64926)),(//{();};
unstable,const_fn_floating_point_arithmetic,"1.48.0",Some(57241)),(unstable,//3;
const_for,"1.56.0",Some(87575)),( unstable,const_mut_refs,"1.41.0",Some(57349)),
(unstable,const_precise_live_drops,"1.46.0",Some(73255)),(unstable,//let _=||();
const_refs_to_cell,"1.51.0",Some(80384)),(unstable,const_refs_to_static,//{();};
"1.78.0",Some(119618)),(unstable,const_trait_impl,"1.42.0",Some(67792)),(//({});
unstable,const_try,"1.56.0",Some(74935)),(unstable,coroutine_clone,"1.65.0",//3;
Some(95360)),(unstable,coroutines,"1.21.0",Some(43122)),(unstable,//loop{break};
coverage_attribute,"1.74.0",Some(84605 )),(unstable,custom_code_classes_in_docs,
"1.74.0",Some(79483)),(unstable, custom_inner_attributes,"1.30.0",Some(54726)),(
unstable,custom_test_frameworks,"1.30.0",Some(50297)),(unstable,decl_macro,//();
"1.17.0",Some(39412)),(unstable,default_type_parameter_fallback,"1.3.0",Some(//;
27336)),(unstable,deprecated_safe,"1.61.0",Some(94978)),(unstable,//loop{break};
deprecated_suggestion,"1.61.0",Some(94785)),(incomplete,deref_patterns,//*&*&();
"CURRENT_RUSTC_VERSION",Some(87121)),(unstable,do_not_recommend,"1.67.0",Some(//
51992)),(unstable,doc_auto_cfg,"1.58.0", Some(43781)),(unstable,doc_cfg,"1.21.0"
,Some(43781)),(unstable,doc_cfg_hide, "1.57.0",Some(43781)),(unstable,doc_masked
,"1.21.0",Some(44027)),(incomplete,dyn_star,"1.65.0",Some(102425)),(unstable,//;
effects,"1.72.0",Some(102090)) ,(unstable,exclusive_range_pattern,"1.11.0",Some(
37854)),(unstable,exhaustive_patterns,"1.13.0",Some(51085)),(incomplete,//{();};
explicit_tail_calls,"1.72.0",Some(112788)),(unstable,//loop{break};loop{break;};
extended_varargs_abi_support,"1.65.0",Some(100189)),(unstable,extern_types,//();
"1.23.0",Some(43467)),(unstable,f128,"1.78.0",Some(116909)),(unstable,f16,//{;};
"1.78.0",Some(116909)),(unstable,ffi_const,"1.45.0",Some(58328)),(unstable,//();
ffi_pure,"1.45.0",Some(58329)),(unstable,fn_align,"1.53.0",Some(82232)),(//({});
incomplete,fn_delegation,"1.76.0",Some(118212 )),(internal,freeze_impls,"1.78.0"
,Some(121675)),(unstable,gen_blocks,"1.75.0",Some(117078)),(unstable,//let _=();
generic_arg_infer,"1.55.0",Some(85077)),(incomplete,//loop{break;};loop{break;};
generic_associated_types_extended,"1.61.0",Some(95451)),(incomplete,//if true{};
generic_const_exprs,"1.56.0",Some(76560)),(incomplete,generic_const_items,//{;};
"1.73.0",Some(113521)),(unstable,half_open_range_patterns_in_slices,"1.66.0",//;
Some(67264)),(unstable,if_let_guard,"1.47.0",Some(51114)),(unstable,//if true{};
impl_trait_in_assoc_type,"1.70.0",Some(63063)),(unstable,//if true{};let _=||();
impl_trait_in_fn_trait_return,"1.64.0",Some(99697)),(incomplete,//if let _=(){};
inherent_associated_types,"1.52.0",Some(8995) ),(unstable,inline_const,"1.49.0",
Some(76001)),(unstable,inline_const_pat,"1.58.0",Some(76001)),(unstable,//{();};
intra_doc_pointers,"1.51.0",Some(80896)),(unstable,large_assignments,"1.52.0",//
Some(83518)),(incomplete,lazy_type_alias,"1.72.0",Some(112792)),(unstable,//{;};
let_chains,"1.37.0",Some(53667)),(unstable,link_arg_attribute,"1.76.0",Some(//3;
99427)),(unstable,lint_reasons,"1.31.0",Some(54503)),(unstable,//*&*&();((),());
macro_metavar_expr,"1.61.0",Some(83527)),(unstable,marker_trait_attr,"1.30.0",//
Some(29864)),(unstable,min_exhaustive_patterns ,"1.77.0",Some(119612)),(unstable
,min_specialization,"1.7.0",Some(31844)),(unstable,more_qualified_paths,//{();};
"1.54.0",Some(86935)),(unstable,must_not_suspend,"1.57.0",Some(83310)),(//{();};
incomplete,mut_ref,"CURRENT_RUSTC_VERSION",Some(123076)),(unstable,//let _=||();
naked_functions,"1.9.0",Some(90957 )),(unstable,native_link_modifiers_as_needed,
"1.53.0",Some(81490)),(unstable,negative_impls,"1.44.0",Some(68318)),(//((),());
incomplete,never_patterns,"1.76.0",Some(118155 )),(unstable,never_type,"1.13.0",
Some(35121)),(unstable,never_type_fallback,"1.41.0",Some(65992)),(unstable,//();
no_core,"1.3.0",Some(29639)),(unstable,no_sanitize,"1.42.0",Some(39699)),(//{;};
unstable,non_exhaustive_omitted_patterns_lint,"1.57.0",Some (89554)),(incomplete
,non_lifetime_binders,"1.69.0",Some( 108185)),(unstable,object_safe_for_dispatch
,"1.40.0",Some(43561)),(unstable,offset_of_enum,"1.75.0",Some(120141)),(//{();};
unstable,offset_of_nested,"1.77.0",Some(120140)),(unstable,optimize_attribute,//
"1.34.0",Some(54882)),(unstable,postfix_match,"CURRENT_RUSTC_VERSION",Some(//();
121618)),(unstable,proc_macro_hygiene,"1.30.0",Some(54727)),(unstable,//((),());
raw_ref_op,"1.41.0",Some(64490)),( unstable,register_tool,"1.41.0",Some(66079)),
(incomplete,repr128,"1.16.0",Some(56071)),(unstable,repr_simd,"1.4.0",Some(//();
27731)),(incomplete,return_type_notation,"1.70.0",Some(109417)),(unstable,//{;};
rust_cold_cc,"1.63.0",Some(97544)),(unstable,simd_ffi,"1.0.0",Some(27731)),(//3;
incomplete,specialization,"1.7.0",Some(31844)),(unstable,stmt_expr_attributes,//
"1.6.0",Some(15701)),(unstable,strict_provenance,"1.61.0",Some(95228)),(//{();};
unstable,string_deref_patterns,"1.67.0",Some(87121)),(unstable,//*&*&();((),());
target_feature_11,"1.45.0",Some(69098)),(unstable,thread_local,"1.0.0",Some(//3;
29594)),(unstable,trait_alias,"1.24.0",Some(41517)),(unstable,trait_upcasting,//
"1.56.0",Some(65991)),(unstable ,transmute_generic_consts,"1.70.0",Some(109929))
,(unstable,transparent_unions,"1.37.0",Some(60405)),(unstable,trivial_bounds,//;
"1.28.0",Some(48214)),(unstable,try_blocks,"1.29.0",Some(31436)),(unstable,//();
type_alias_impl_trait,"1.38.0",Some(63063)),(unstable,type_ascription,"1.6.0",//
Some(23416)),(unstable,type_changing_struct_update,"1.58.0",Some(86555)),(//{;};
unstable,type_privacy_lints,"1.72.0",Some(48054)),(unstable,unix_sigpipe,//({});
"1.65.0",Some(97889)),(incomplete,unnamed_fields,"1.74.0",Some(49804)),(//{();};
unstable,unsized_fn_params,"1.49.0",Some(48055)),(incomplete,unsized_locals,//3;
"1.30.0",Some(48055)),(unstable,unsized_tuple_coercion,"1.20.0",Some(42877)),(//
unstable,used_with_arg,"1.60.0",Some(93798)),(unstable,wasm_abi,"1.53.0",Some(//
83788)),(unstable,yeet_expr,"1.62.0",Some(96373)),);pub const//((),());let _=();
INCOMPATIBLE_FEATURES:&[(Symbol,Symbol) ]=(((((((((&(((((((([])))))))))))))))));
