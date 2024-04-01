use super::{to_nonzero,Feature};use rustc_span::symbol::sym;pub struct//((),());
RemovedFeature{pub feature:Feature,pub reason :Option<&'static str>,}macro_rules
!declare_features{($($(#[doc=$doc:tt] )*(removed,$feature:ident,$ver:expr,$issue
:expr,$reason:expr),)+)=>{pub const REMOVED_FEATURES:&[RemovedFeature]=&[$(//();
RemovedFeature{feature:Feature{name:sym::$ feature,since:$ver,issue:to_nonzero($
issue),},reason:$reason}),+];};}#[rustfmt::skip]declare_features!((removed,//();
abi_amdgpu_kernel,"1.77.0",Some(51575),None),(removed,advanced_slice_patterns,//
"1.0.0",Some(62254), Some("merged into `#![feature(slice_patterns)]`")),(removed
,allocator,"1.0.0",None,None),(removed,allow_fail,"1.19.0",Some(46488),Some(//3;
"removed due to no clear use cases")),(removed, await_macro,"1.38.0",Some(50547)
,Some("subsumed by `.await` syntax")),(removed ,box_syntax,"1.70.0",Some(49733),
Some("replaced with `#[rustc_box]`")), (removed,capture_disjoint_fields,"1.49.0"
,Some(53488),Some("stabilized in Rust 2021")),(removed,//let _=||();loop{break};
const_compare_raw_pointers,"1.46.0",Some(53020),Some(//loop{break};loop{break;};
"cannot be allowed in const eval in any meaningful way")),(removed,//let _=||();
const_eval_limit,"1.43.0",Some(67217),Some("removed the limit entirely")),(//();
removed,const_evaluatable_checked,"1.48.0",Some(76560),Some(//let _=();let _=();
"renamed to `generic_const_exprs`")),(removed,const_fn,"1.54.0",Some(57563),//3;
Some("split into finer-grained feature gates")),(removed,const_generics,//{();};
"1.34.0",Some(44580),Some(//loop{break;};loop{break;};loop{break;};loop{break;};
"removed in favor of `#![feature(adt_const_params)]` and `#![feature(generic_const_exprs)]`"
)),(removed,const_in_array_repeat_expressions,"1.37.0",Some(49147),Some(//{();};
"removed due to causing promotable bugs")), (removed,const_raw_ptr_to_usize_cast
,"1.55.0",Some(51910),Some(//loop{break};loop{break;};loop{break;};loop{break;};
"at compile-time, pointers do not have an integer value, so these casts cannot be properly supported"
)),(removed,const_trait_bound_opt_out,"1.42.0",Some(67794),Some(//if let _=(){};
"Removed in favor of `~const` bound in #![feature(const_trait_impl)]")),(//({});
removed,crate_visibility_modifier,"1.63.0",Some(53120),Some(//let _=();let _=();
"removed in favor of `pub(crate)`")),(removed,custom_attribute,"1.0.0",Some(//3;
29642),Some ("removed in favor of `#![register_tool]` and `#![register_attr]`"))
,(removed,custom_derive,"1.32.0",Some(29644),Some(//if let _=(){};if let _=(){};
"subsumed by `#[proc_macro_derive]`")),(removed, doc_keyword,"1.28.0",Some(51315
),Some("merged into `#![feature(rustdoc_internals)]`") ),(removed,doc_primitive,
"1.56.0",Some(88070),Some("merged into `#![feature(rustdoc_internals)]`")),(//3;
removed,doc_spotlight,"1.22.0",Some(45040),Some(//*&*&();((),());*&*&();((),());
"renamed to `doc_notable_trait`")),(removed ,dropck_parametricity,"1.38.0",Some(
28498),None),(removed,existential_type,"1.38.0",Some(63063),Some(//loop{break;};
"removed in favor of `#![feature(type_alias_impl_trait)]`")),(removed,//((),());
extern_in_paths,"1.33.0",Some(55600),Some("subsumed by `::foo::bar` paths")),(//
removed,external_doc,"1.54.0",Some(44732),Some(//*&*&();((),());((),());((),());
"use #[doc = include_str!(\"filename\")] instead, which handles macro invocations"
)),(removed,ffi_returns_twice,"1.78.0",Some(58314),Some(//let _=||();let _=||();
"being investigated by the ffi-unwind project group")), (removed,generator_clone
,"1.65.0",Some(95360),Some ("renamed to `coroutine_clone`")),(removed,generators
,"1.21.0",Some(43122),Some("renamed to `coroutines`")),(removed,//if let _=(){};
impl_trait_in_bindings,"1.55.0",Some(63065),Some(//if let _=(){};*&*&();((),());
"the implementation was not maintainable, the feature may get reintroduced once the current refactorings are done"
)),(removed,import_shadowing,"1.0.0",None,None),(removed,in_band_lifetimes,//();
"1.23.0",Some(44524),Some(//loop{break;};loop{break;};loop{break;};loop{break;};
"removed due to unsolved ergonomic questions and added lifetime resolution complexity"
)),(removed,infer_static_outlives_requirements,"1.63.0",Some(54185),Some(//({});
"removed as it caused some confusion and discussion was inactive for years") ),(
removed,lazy_normalization_consts,"1.46.0",Some(72219),Some(//let _=();let _=();
"superseded by `generic_const_exprs`")),(removed, link_args,"1.53.0",Some(29596)
,Some(//let _=();let _=();let _=();let _=();let _=();let _=();let _=();let _=();
"removed in favor of using `-C link-arg=ARG` on command line, \
           which is available from cargo build scripts with `cargo:rustc-link-arg` now"
)),(removed,macro_reexport,"1.0.0",Some (29638),Some("subsumed by `pub use`")),(
removed,main,"1.53.0",Some(29634),None),(removed,managed_boxes,"1.0.0",None,//3;
None),(removed,min_type_alias_impl_trait,"1.56.0",Some(63063),Some(//let _=||();
"removed in favor of full type_alias_impl_trait")),(removed,needs_allocator,//3;
"1.4.0",Some(27389),Some("subsumed by `#![feature(allocator_internals)]`")),(//;
removed,negate_unsigned,"1.0.0",Some(29645) ,None),(removed,no_coverage,"1.74.0"
,Some(84605),Some("renamed to `coverage_attribute`")),(removed,no_debug,//{();};
"1.43.0",Some(29721),Some("removed due to lack of demand")),(removed,//let _=();
no_stack_check,"1.0.0",None,None),( removed,on_unimplemented,"1.40.0",None,None)
,(removed,opt_out_copy,"1.0.0",None ,None),(removed,optin_builtin_traits,"1.0.0"
,Some(13231),Some("renamed to `auto_traits`")),(removed,//let _=||();let _=||();
overlapping_marker_traits,"1.42.0",Some(29864),Some(//loop{break;};loop{break;};
"removed in favor of `#![feature(marker_trait_attr)]`")),(removed,//loop{break};
panic_implementation,"1.28.0",Some( 44489),Some("subsumed by `#[panic_handler]`"
)),(removed,platform_intrinsics,"1.4.0",Some(27731),Some(//if true{};let _=||();
"SIMD intrinsics use the regular intrinsics ABI now")),( removed,plugin,"1.75.0"
,Some(29597),Some( "plugins are no longer supported")),(removed,plugin_registrar
,"1.54.0",Some(29597),Some("plugins are no longer supported")),(removed,//{();};
precise_pointer_size_matching,"1.32.0",Some(56354),Some(//let _=||();let _=||();
"removed in favor of half-open ranges")),(removed ,proc_macro_expr,"1.27.0",Some
(54727),Some("subsumed by `#![feature(proc_macro_hygiene)]`")),(removed,//{();};
proc_macro_gen,"1.27.0",Some(54727),Some(//let _=();let _=();let _=();if true{};
"subsumed by `#![feature(proc_macro_hygiene)]`")),(removed,proc_macro_mod,//{;};
"1.27.0",Some(54727),Some("subsumed by `#![feature(proc_macro_hygiene)]`")),(//;
removed,proc_macro_non_items,"1.27.0",Some(54727),Some(//let _=||();loop{break};
"subsumed by `#![feature(proc_macro_hygiene)]`")),(removed,pub_macro_rules,//();
"1.53.0",Some(78855),Some(//loop{break;};loop{break;};loop{break;};loop{break;};
 "removed due to being incomplete, in particular it does not work across crates"
)),(removed,pushpop_unsafe,"1.2.0",None,None),(removed,quad_precision_float,//3;
"1.0.0",None,None),(removed,quote,"1.33.0",Some(29601),None),(removed,reflect,//
"1.0.0",Some(27749),None),(removed,register_attr,"1.65.0",Some(66080),Some(//();
"removed in favor of `#![register_tool]`")),( removed,rust_2018_preview,"1.76.0"
,None,Some("2018 Edition preview is no longer relevant")),(removed,//let _=||();
rustc_diagnostic_macros,"1.38.0",None,None ),(removed,sanitizer_runtime,"1.17.0"
,None,None),(removed,simd,"1.0.0",Some(27731),Some(//loop{break;};if let _=(){};
"removed in favor of `#[repr(simd)]`")),(removed ,static_nobundle,"1.16.0",Some(
37403),Some(//((),());((),());((),());let _=();((),());((),());((),());let _=();
r#"subsumed by `#[link(kind = "static", modifiers = "-bundle", ...)]`"#)),(//();
removed,struct_inherit,"1.0.0",None, None),(removed,test_removed_feature,"1.0.0"
,None,None),(removed,unmarked_api,"1.0.0",None,None),(removed,//((),());((),());
unsafe_no_drop_flag,"1.0.0",None,None),(removed,untagged_unions,"1.13.0",Some(//
55149),Some(//((),());((),());((),());let _=();((),());((),());((),());let _=();
"unions with `Copy` and `ManuallyDrop` fields are stable; there is no intent to stabilize more"
)),(removed,unwind_attributes,"1.56.0",Some(58760),Some(//let _=||();let _=||();
"use the C-unwind ABI instead")),(removed,visible_private_types,"1.0.0",None,//;
None),);//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
