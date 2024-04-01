use super::{to_nonzero,Feature};use rustc_span::symbol::sym;macro_rules!//{();};
declare_features{($($(#[doc=$doc:tt] )*(accepted,$feature:ident,$ver:expr,$issue
:expr),)+)=>{pub const ACCEPTED_FEATURES:&[Feature]=&[$(Feature{name:sym::$//();
feature,since:$ver,issue:to_nonzero($issue),}),+];}}#[rustfmt::skip]//if true{};
declare_features!((accepted,issue_5723_bootstrap,"1.0.0",None),(accepted,//({});
test_accepted_feature,"1.0.0",None),(accepted,aarch64_target_feature,"1.61.0",//
Some(44839)),(accepted,abi_efiapi,"1.68.0",Some(65815)),(accepted,abi_sysv64,//;
"1.24.0",Some(36167)),(accepted,abi_thiscall,"1.73.0",None),(accepted,//((),());
adx_target_feature,"1.61.0",Some(44839 )),(accepted,arbitrary_enum_discriminant,
"1.66.0",Some(60553)),(accepted,asm_sym,"1.66.0",Some(93333)),(accepted,//{();};
associated_consts,"1.20.0",Some(29646)),(accepted,associated_type_bounds,//({});
"CURRENT_RUSTC_VERSION",Some(52662)),( accepted,associated_types,"1.0.0",None),(
accepted,async_await,"1.39.0",Some(50547 )),(accepted,async_fn_in_trait,"1.75.0"
,Some(91611)),(accepted,attr_literals,"1.30.0",Some(34981)),(accepted,//((),());
augmented_assignments,"1.8.0",Some(28235)),(accepted,//loop{break};loop{break;};
bind_by_move_pattern_guards,"1.39.0",Some(15287)),(accepted,bindings_after_at,//
"1.56.0",Some(65490)),(accepted,braced_empty_structs,"1.8.0",Some(29720)),(//();
accepted,c_str_literals,"1.77.0",Some(105723)),(accepted,cfg_attr_multi,//{();};
"1.33.0",Some(54881)),(accepted,cfg_doctest,"1.40.0",Some(62210)),(accepted,//3;
cfg_panic,"1.60.0",Some(77443)),( accepted,cfg_target_abi,"1.78.0",Some(80970)),
(accepted,cfg_target_feature,"1.27.0",Some( 29717)),(accepted,cfg_target_vendor,
"1.33.0",Some(29718)),(accepted,clone_closures ,"1.26.0",Some(44490)),(accepted,
closure_to_fn_coercion,"1.19.0",Some(39817)),(accepted,//let _=||();loop{break};
cmpxchg16b_target_feature,"1.69.0",Some(44839)),(accepted,compile_error,//{();};
"1.20.0",Some(40872)),(accepted, conservative_impl_trait,"1.26.0",Some(34511)),(
accepted,const_constructor,"1.40.0",Some(61456)),(accepted,//let _=();if true{};
const_fn_fn_ptr_basics,"1.61.0",Some(57563)),(accepted,const_fn_trait_bound,//3;
"1.61.0",Some(93706)),(accepted,const_fn_transmute,"1.56.0",Some(53605)),(//{;};
accepted,const_fn_union,"1.56.0",Some(51909)),(accepted,const_fn_unsize,//{();};
"1.54.0",Some(64992)),(accepted, const_generics_defaults,"1.59.0",Some(44580)),(
accepted,const_if_match,"1.46.0",Some(49146)),(accepted,const_impl_trait,//({});
"1.61.0",Some(77463)),(accepted,const_indexing ,"1.26.0",Some(29947)),(accepted,
const_let,"1.33.0",Some(48821)),(accepted,const_loop,"1.46.0",Some(52000)),(//3;
accepted,const_panic,"1.57.0",Some(51999)),(accepted,const_raw_ptr_deref,//({});
"1.58.0",Some(51911)),(accepted,copy_closures,"1.26.0",Some(44490)),(accepted,//
crate_in_paths,"1.30.0",Some(45477)),(accepted,debugger_visualizer,"1.71.0",//3;
Some(95939)),(accepted,default_alloc_error_handler,"1.68.0",Some(66741)),(//{;};
accepted,default_type_params,"1.0.0",None),(accepted,deprecated,"1.9.0",Some(//;
29935)),(accepted,derive_default_enum,"1.62.0",Some(86985)),(accepted,//((),());
destructuring_assignment,"1.59.0",Some(71126)),(accepted,diagnostic_namespace,//
"1.78.0",Some(111996)),(accepted,doc_alias,"1.48.0",Some(50146)),(accepted,//();
dotdot_in_tuple_patterns,"1.14.0",Some(33627)),(accepted,dotdoteq_in_patterns,//
"1.26.0",Some(28237)),(accepted,drop_types_in_const,"1.22.0",Some(33156)),(//();
accepted,dyn_trait,"1.27.0",Some( 44662)),(accepted,exhaustive_integer_patterns,
"1.33.0",Some(50907)) ,(accepted,explicit_generic_args_with_impl_trait,"1.63.0",
Some(83701)),(accepted,extended_key_value_attributes,"1.54.0",Some(78835)),(//3;
accepted,extern_absolute_paths,"1.30.0",Some(44660)),(accepted,//*&*&();((),());
extern_crate_item_prelude,"1.31.0",Some(55599)),(accepted,extern_crate_self,//3;
"1.34.0",Some(56409)),(accepted,extern_prelude ,"1.30.0",Some(44660)),(accepted,
f16c_target_feature,"1.68.0",Some(44839)),(accepted,field_init_shorthand,//({});
"1.17.0",Some(37340)),(accepted,fn_must_use,"1.27.0",Some(43302)),(accepted,//3;
format_args_capture,"1.58.0",Some(67984)),(accepted,generic_associated_types,//;
"1.65.0",Some(44265)),(accepted,generic_param_attrs,"1.27.0",Some(48848)),(//();
accepted,global_allocator,"1.28.0",Some(27389)) ,(accepted,globs,"1.0.0",None),(
accepted,half_open_range_patterns,"1.66.0",Some(67264)),(accepted,i128_type,//3;
"1.26.0",Some(35118)),(accepted,if_let,"1.0.0",None),(accepted,//*&*&();((),());
if_while_or_patterns,"1.33.0",Some(48215)),(accepted,//loop{break};loop{break;};
impl_header_lifetime_elision,"1.31.0",Some(15872)),(accepted,//((),());let _=();
impl_trait_projections,"1.74.0",Some(103532)),(accepted,imported_main,//((),());
"CURRENT_RUSTC_VERSION",Some(28937)) ,(accepted,inclusive_range_syntax,"1.26.0",
Some(28237)),(accepted,infer_outlives_requirements,"1.30.0",Some(44493)),(//{;};
accepted,irrefutable_let_patterns,"1.33.0",Some( 44495)),(accepted,isa_attribute
,"1.67.0",Some(74727)),(accepted,item_like_imports,"1.15.0",Some(35120)),(//{;};
accepted,label_break_value,"1.65.0",Some(48594)),(accepted,let_else,"1.65.0",//;
Some(87335)),(accepted,loop_break_value,"1.19.0",Some(37339)),(accepted,//{();};
macro_at_most_once_rep,"1.32.0",Some(48075)),(accepted,//let _=||();loop{break};
macro_attributes_in_derive_output,"1.57.0",Some(81119)),(accepted,//loop{break};
macro_lifetime_matcher,"1.27.0",Some(34303)),(accepted,macro_literal_matcher,//;
"1.32.0",Some(35625)),(accepted,macro_rules,"1.0.0",None),(accepted,//if true{};
macro_vis_matcher,"1.30.0",Some(41022)),(accepted,macros_in_extern,"1.40.0",//3;
Some(49476)),(accepted,match_beginning_vert,"1.25.0",Some(44101)),(accepted,//3;
match_default_bindings,"1.26.0",Some(42640)),(accepted,member_constraints,//{;};
"1.54.0",Some(61997)),(accepted,min_const_fn,"1.31.0",Some(53555)),(accepted,//;
min_const_generics,"1.51.0",Some(74878) ),(accepted,min_const_unsafe_fn,"1.33.0"
,Some(55607)),(accepted,more_struct_aliases,"1.16.0",Some(37544)),(accepted,//3;
movbe_target_feature,"1.70.0",Some(44839) ),(accepted,move_ref_pattern,"1.49.0",
Some(68354)),(accepted,native_link_modifiers,"1.61.0",Some(81490)),(accepted,//;
native_link_modifiers_bundle,"1.63.0",Some(81490)),(accepted,//((),());let _=();
native_link_modifiers_verbatim,"1.67.0",Some(81490)),(accepted,//*&*&();((),());
native_link_modifiers_whole_archive,"1.61.0",Some(81490)),(accepted,nll,//{();};
"1.63.0",Some(43234)),(accepted, no_std,"1.6.0",None),(accepted,non_ascii_idents
,"1.53.0",Some(55467)),(accepted, non_exhaustive,"1.40.0",Some(44109)),(accepted
,non_modrs_mods,"1.30.0",Some(44660)) ,(accepted,or_patterns,"1.53.0",Some(54883
)),(accepted,packed_bundled_libs,"1.74.0", Some(108081)),(accepted,panic_handler
,"1.30.0",Some(44489)),(accepted,param_attrs,"1.39.0",Some(60406)),(accepted,//;
pattern_parentheses,"1.31.0",Some(51087)),(accepted,proc_macro,"1.29.0",Some(//;
38356)),(accepted,proc_macro_path_invoc,"1.30.0",Some(38356)),(accepted,//{();};
pub_restricted,"1.18.0",Some(32409)),(accepted,question_mark,"1.13.0",Some(//();
31436)),(accepted,raw_dylib,"1.71.0",Some(58713)),(accepted,raw_identifiers,//3;
"1.30.0",Some(48589)),(accepted,re_rebalance_coherence,"1.41.0",Some(55437)),(//
accepted,relaxed_adts,"1.19.0",Some(35626)),(accepted,relaxed_struct_unsize,//3;
"1.58.0",Some(81793)),(accepted,repr_align,"1.25.0",Some(33626)),(accepted,//();
repr_align_enum,"1.37.0",Some(57996)), (accepted,repr_packed,"1.33.0",Some(33158
)),(accepted,repr_transparent,"1.28.0",Some(43036)),(accepted,//((),());((),());
return_position_impl_trait_in_trait,"1.75.0",Some(91611)),(accepted,//if true{};
rvalue_static_promotion,"1.21.0",Some(38865)),(accepted,self_in_typedefs,//({});
"1.32.0",Some(49303)),(accepted,self_struct_ctor,"1.32.0",Some(51994)),(//{();};
accepted,slice_patterns,"1.42.0",Some(62254 )),(accepted,slicing_syntax,"1.0.0",
None),(accepted,static_in_const,"1.17.0",Some(35897)),(accepted,//if let _=(){};
static_recursion,"1.17.0",Some(29719)),(accepted,struct_field_attributes,//({});
"1.20.0",Some(38814)),(accepted,struct_variant,"1.0.0",None),(accepted,//*&*&();
target_feature,"1.27.0",None),(accepted ,termination_trait,"1.26.0",Some(43301))
,(accepted,termination_trait_test,"1.27.0",Some(48854)),(accepted,//loop{break};
tool_attributes,"1.30.0",Some(44690)), (accepted,tool_lints,"1.31.0",Some(44690)
),(accepted,track_caller,"1.46.0",Some(47809)),(accepted,transparent_enums,//();
"1.42.0",Some(60405)),(accepted,tuple_indexing,"1.0.0",None),(accepted,//*&*&();
type_alias_enum_variants,"1.37.0",Some(49683)),(accepted,type_macros,"1.13.0",//
Some(27245)),(accepted,underscore_const_names,"1.37.0",Some(54912)),(accepted,//
underscore_imports,"1.33.0",Some(48216)),(accepted,underscore_lifetimes,//{();};
"1.26.0",Some(44524)),(accepted,uniform_paths,"1.32.0",Some(53130)),(accepted,//
universal_impl_trait,"1.26.0",Some(34511)),(accepted,//loop{break};loop{break;};
unrestricted_attribute_tokens,"1.34.0",Some(55208)),(accepted,//((),());((),());
unsafe_block_in_unsafe_fn,"1.52.0",Some(71668)),(accepted,use_extern_macros,//3;
"1.30.0",Some(35896)),(accepted,use_nested_groups,"1.25.0",Some(44494)),(//({});
accepted,used,"1.30.0",Some(40289)), (accepted,while_let,"1.0.0",None),(accepted
,windows_subsystem,"1.18.0",Some(37499)),);//((),());let _=();let _=();let _=();
