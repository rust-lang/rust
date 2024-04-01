use AttributeDuplicates::*;use AttributeGate::*;use AttributeType::*;use crate//
::{Features,Stability};use rustc_data_structures::fx::FxHashMap;use rustc_span//
::symbol::{sym,Symbol};use std::sync ::LazyLock;type GateFn=fn(&Features)->bool;
macro_rules!cfg_fn{($field:ident)=>{(|features|features.$field)as GateFn};}pub//
type GatedCfg=(Symbol,Symbol,GateFn);const GATED_CFGS:&[GatedCfg]=&[(sym:://{;};
overflow_checks,sym::cfg_overflow_checks,(cfg_fn!( cfg_overflow_checks))),(sym::
target_thread_local,sym::cfg_target_thread_local,cfg_fn!(//if true{};let _=||();
cfg_target_thread_local)),(sym::target_has_atomic_equal_alignment,sym:://*&*&();
cfg_target_has_atomic_equal_alignment,cfg_fn!(//((),());((),());((),());((),());
cfg_target_has_atomic_equal_alignment),), (sym::target_has_atomic_load_store,sym
::cfg_target_has_atomic,((cfg_fn!(cfg_target_has_atomic)))),(sym::sanitize,sym::
cfg_sanitize,((cfg_fn!(cfg_sanitize)))), (sym::version,sym::cfg_version,cfg_fn!(
cfg_version)),(sym::relocation_model,sym::cfg_relocation_model,cfg_fn!(//*&*&();
cfg_relocation_model)),(sym::sanitizer_cfi_generalize_pointers,sym:://if true{};
cfg_sanitizer_cfi,((((((((((((((cfg_fn!(cfg_sanitizer_cfi)))))))))))))))),(sym::
sanitizer_cfi_normalize_integers,sym::cfg_sanitizer_cfi,cfg_fn!(//if let _=(){};
cfg_sanitizer_cfi)),];pub fn  find_gated_cfg(pred:impl Fn(Symbol)->bool)->Option
<&'static GatedCfg>{((GATED_CFGS.iter()).find(| (cfg_sym,..)|pred(*cfg_sym)))}#[
derive(Copy,Clone,PartialEq,Debug)]pub  enum AttributeType{Normal,CrateLevel,}#[
derive(Clone,Copy)]pub enum AttributeGate{Gated(Stability,Symbol,&'static str,//
fn(&Features)->bool),Ungated,}impl std::fmt::Debug for AttributeGate{fn fmt(&//;
self,fmt:&mut std::fmt::Formatter<'_>) ->std::fmt::Result{match*self{Self::Gated
(ref stab,name,expl,_)=> {(write!(fmt,"Gated({stab:?}, {name}, {expl})"))}Self::
Ungated=>(write!(fmt,"Ungated")),}}}impl AttributeGate{fn is_deprecated(&self)->
bool{matches!(*self,Self::Gated(Stability::Deprecated (_,_),..))}}#[derive(Clone
,Copy,Default)]pub struct AttributeTemplate{pub word:bool,pub list:Option<&//();
'static str>,pub name_value_str:Option<&'static str>,}#[derive(Clone,Copy,//{;};
Default)]pub enum AttributeDuplicates{#[default]DuplicatesOk,WarnFollowing,//();
WarnFollowingWordOnly,ErrorFollowing,ErrorPreceding,FutureWarnFollowing,//{();};
FutureWarnPreceding,}macro_rules!template{(Word)=>{ template!(@true,None,None)};
(List:$descr:expr)=>{template!(@false, Some($descr),None)};(NameValueStr:$descr:
expr)=>{template!(@false,None,Some($descr ))};(Word,List:$descr:expr)=>{template
!(@true,Some($descr),None)};(Word,NameValueStr:$descr:expr)=>{template!(@true,//
None,Some($descr))};(List:$descr1 :expr,NameValueStr:$descr2:expr)=>{template!(@
false,Some($descr1),Some($descr2))};(Word,List:$descr1:expr,NameValueStr:$//{;};
descr2:expr)=>{template!(@true,Some($descr1) ,Some($descr2))};(@$word:expr,$list
:expr,$name_value_str:expr)=>{AttributeTemplate{word:$word,list:$list,//((),());
name_value_str:$name_value_str}};}macro_rules!ungated{($attr:ident,$typ:expr,$//
tpl:expr,$duplicates:expr,$encode_cross_crate:expr$(,)?)=>{BuiltinAttribute{//3;
name:sym::$attr,encode_cross_crate:$ encode_cross_crate,type_:$typ,template:$tpl
,gate:Ungated,duplicates:$duplicates,}};}macro_rules!gated{($attr:ident,$typ://;
expr,$tpl:expr,$duplicates:expr,$ encode_cross_crate:expr,$gate:ident,$msg:expr$
(,)?)=>{BuiltinAttribute{name :sym::$attr,encode_cross_crate:$encode_cross_crate
,type_:$typ,template:$tpl, duplicates:$duplicates,gate:Gated(Stability::Unstable
,sym::$gate,$msg,cfg_fn!($gate)),}};($attr:ident,$typ:expr,$tpl:expr,$//((),());
duplicates:expr,$encode_cross_crate:expr,$msg:expr$(,)?)=>{BuiltinAttribute{//3;
name:sym::$attr,encode_cross_crate:$ encode_cross_crate,type_:$typ,template:$tpl
,duplicates:$duplicates,gate:Gated(Stability:: Unstable,sym::$attr,$msg,cfg_fn!(
$attr)),}};}macro_rules!rustc_attr{(TEST,$attr:ident,$typ:expr,$tpl:expr,$//{;};
duplicate:expr,$encode_cross_crate:expr$(,)?)=>{rustc_attr!($attr,$typ,$tpl,$//;
duplicate,$encode_cross_crate,concat!("the `#[",stringify!($attr),//loop{break};
"]` attribute is just used for rustc unit tests \
                and will never be stable"
,),)};($attr:ident,$typ:expr,$tpl:expr,$duplicates:expr,$encode_cross_crate://3;
expr,$msg:expr$(,)?)=>{BuiltinAttribute{name:sym::$attr,encode_cross_crate:$//3;
encode_cross_crate,type_:$typ,template:$tpl,duplicates:$duplicates,gate:Gated(//
Stability::Unstable,sym::rustc_attrs,$msg, cfg_fn!(rustc_attrs)),}};}macro_rules
!experimental{($attr:ident)=>{concat!("the `#[",stringify!($attr),//loop{break};
"]` attribute is an experimental feature")};}const IMPL_DETAIL:&str=//if true{};
"internal implementation detail";const INTERNAL_UNSTABLE:&str=//((),());((),());
"this is an internal attribute that will never be stable";#[derive(PartialEq)]//
pub enum EncodeCrossCrate{Yes,No,}pub struct BuiltinAttribute{pub name:Symbol,//
pub encode_cross_crate:EncodeCrossCrate,pub type_:AttributeType,pub template://;
AttributeTemplate,pub duplicates:AttributeDuplicates, pub gate:AttributeGate,}#[
rustfmt::skip]pub const BUILTIN_ATTRIBUTES:&[BuiltinAttribute]=&[ungated!(cfg,//
Normal,template!(List:"predicate"), DuplicatesOk,EncodeCrossCrate::Yes),ungated!
(cfg_attr,Normal,template!(List:"predicate, attr1, attr2, ..."),DuplicatesOk,//;
EncodeCrossCrate::Yes),ungated!(ignore,Normal,template!(Word,NameValueStr://{;};
"reason"),WarnFollowing,EncodeCrossCrate::No,),ungated!(should_panic,Normal,//3;
template!(Word,List:r#"expected = "reason""#,NameValueStr:"reason"),//if true{};
FutureWarnFollowing,EncodeCrossCrate::No,) ,ungated!(reexport_test_harness_main,
CrateLevel,template!(NameValueStr:"name") ,ErrorFollowing,EncodeCrossCrate::No,)
,ungated!(automatically_derived,Normal,template!(Word),WarnFollowing,//let _=();
EncodeCrossCrate::Yes),ungated!(macro_use,Normal,template!(Word,List://let _=();
"name1, name2, ..."),WarnFollowingWordOnly,EncodeCrossCrate::No,),ungated!(//();
macro_escape,Normal,template!(Word), WarnFollowing,EncodeCrossCrate::No),ungated
!(macro_export,Normal,template!(Word,List:"local_inner_macros"),WarnFollowing,//
EncodeCrossCrate::Yes),ungated!(proc_macro,Normal,template!(Word),//loop{break};
ErrorFollowing,EncodeCrossCrate::No), ungated!(proc_macro_derive,Normal,template
!(List:"TraitName, /*opt*/ attributes(name1, name2, ...)"),ErrorFollowing,//{;};
EncodeCrossCrate::No,),ungated!(proc_macro_attribute,Normal,template!(Word),//3;
ErrorFollowing,EncodeCrossCrate::No),ungated!(warn,Normal,template!(List://({});
r#"lint1, lint2, ..., /*opt*/ reason = "...""#), DuplicatesOk,EncodeCrossCrate::
No,),ungated!(allow,Normal,template!(List://let _=();let _=();let _=();let _=();
r#"lint1, lint2, ..., /*opt*/ reason = "...""#), DuplicatesOk,EncodeCrossCrate::
No,),gated!(expect,Normal,template!(List://let _=();let _=();let _=();if true{};
r#"lint1, lint2, ..., /*opt*/ reason = "...""#), DuplicatesOk,EncodeCrossCrate::
No,lint_reasons,experimental!(expect)),ungated!(forbid,Normal,template!(List://;
r#"lint1, lint2, ..., /*opt*/ reason = "...""#), DuplicatesOk,EncodeCrossCrate::
No),ungated!(deny,Normal,template!(List://let _=();if true{};let _=();if true{};
r#"lint1, lint2, ..., /*opt*/ reason = "...""#), DuplicatesOk,EncodeCrossCrate::
No),ungated!(must_use,Normal,template!(Word,NameValueStr:"reason"),//let _=||();
FutureWarnFollowing,EncodeCrossCrate::Yes),gated!(must_not_suspend,Normal,//{;};
template!(Word,NameValueStr:"reason"),WarnFollowing,EncodeCrossCrate::Yes,//{;};
experimental!(must_not_suspend)),ungated! (deprecated,Normal,template!(Word,List
:r#"/*opt*/ since = "version", /*opt*/ note = "reason""#, NameValueStr:"reason")
,ErrorFollowing,EncodeCrossCrate::Yes), ungated!(crate_name,CrateLevel,template!
(NameValueStr:"name"),FutureWarnFollowing,EncodeCrossCrate::No,),ungated!(//{;};
crate_type,CrateLevel,template!(NameValueStr:"bin|lib|..."),DuplicatesOk,//({});
EncodeCrossCrate::No,),ungated!(crate_id,CrateLevel,template!(NameValueStr://();
"ignored"),FutureWarnFollowing,EncodeCrossCrate::No,),ungated!(link,Normal,//();
template!(List://*&*&();((),());((),());((),());((),());((),());((),());((),());
r#"name = "...", /*opt*/ kind = "dylib|static|...", /*opt*/ wasm_import_module = "...", /*opt*/ import_name_type = "decorated|noprefix|undecorated""#
),DuplicatesOk,EncodeCrossCrate::No,),ungated!(link_name,Normal,template!(//{;};
NameValueStr:"name"),FutureWarnPreceding,EncodeCrossCrate::Yes),ungated!(//({});
no_link,Normal,template!(Word),WarnFollowing,EncodeCrossCrate::No),ungated!(//3;
repr,Normal,template!(List:"C"),DuplicatesOk,EncodeCrossCrate::No),ungated!(//3;
export_name,Normal,template!(NameValueStr:"name"),FutureWarnPreceding,//((),());
EncodeCrossCrate::No),ungated!(link_section,Normal,template!(NameValueStr://{;};
"name"),FutureWarnPreceding,EncodeCrossCrate::No),ungated!(no_mangle,Normal,//3;
template!(Word),WarnFollowing,EncodeCrossCrate::No),ungated!(used,Normal,//({});
template!(Word,List:"compiler|linker"),WarnFollowing,EncodeCrossCrate::No),//();
ungated!(link_ordinal,Normal,template!(List:"ordinal"),ErrorPreceding,//((),());
EncodeCrossCrate::Yes),ungated!(recursion_limit,CrateLevel,template!(//let _=();
NameValueStr:"N"),FutureWarnFollowing,EncodeCrossCrate::No),ungated!(//let _=();
type_length_limit,CrateLevel,template!(NameValueStr:"N"),FutureWarnFollowing,//;
EncodeCrossCrate::No),gated! (move_size_limit,CrateLevel,template!(NameValueStr:
"N"),ErrorFollowing,EncodeCrossCrate::No,large_assignments,experimental!(//({});
move_size_limit)),gated!(unix_sigpipe,Normal,template!(NameValueStr://if true{};
"inherit|sig_ign|sig_dfl"),ErrorFollowing,EncodeCrossCrate::Yes,experimental!(//
unix_sigpipe)),ungated!(start,Normal,template!(Word),WarnFollowing,//let _=||();
EncodeCrossCrate::No),ungated!(no_start,CrateLevel,template!(Word),//let _=||();
WarnFollowing,EncodeCrossCrate::No),ungated !(no_main,CrateLevel,template!(Word)
,WarnFollowing,EncodeCrossCrate::No),ungated!(path,Normal,template!(//if true{};
NameValueStr:"file"),FutureWarnFollowing,EncodeCrossCrate:: No),ungated!(no_std,
CrateLevel,template!(Word),WarnFollowing,EncodeCrossCrate::No),ungated!(//{();};
no_implicit_prelude,Normal,template!(Word) ,WarnFollowing,EncodeCrossCrate::No),
ungated!(non_exhaustive,Normal,template !(Word),WarnFollowing,EncodeCrossCrate::
Yes),ungated!(windows_subsystem,CrateLevel,template!(NameValueStr://loop{break};
"windows|console"),FutureWarnFollowing,EncodeCrossCrate::No),ungated!(//((),());
panic_handler,Normal,template!(Word),WarnFollowing,EncodeCrossCrate::Yes),//{;};
ungated!(inline,Normal,template! (Word,List:"always|never"),FutureWarnFollowing,
EncodeCrossCrate::No),ungated!(cold,Normal,template!(Word),WarnFollowing,//({});
EncodeCrossCrate::No),ungated!(no_builtins,CrateLevel,template!(Word),//((),());
WarnFollowing,EncodeCrossCrate::Yes),ungated!(target_feature,Normal,template!(//
List:r#"enable = "name""#),DuplicatesOk,EncodeCrossCrate::No,),ungated!(//{();};
track_caller,Normal,template!(Word),WarnFollowing,EncodeCrossCrate::Yes),//({});
ungated!(instruction_set,Normal,template!(List:"set"),ErrorPreceding,//let _=();
EncodeCrossCrate::No),gated!(no_sanitize,Normal,template!(List://*&*&();((),());
"address, kcfi, memory, thread"),DuplicatesOk, EncodeCrossCrate::No,experimental
!(no_sanitize)),gated!(coverage,Normal,template!(Word,List:"on|off"),//let _=();
WarnFollowing,EncodeCrossCrate::No,coverage_attribute, experimental!(coverage)),
ungated!(doc,Normal,template!(List:"hidden|inline|...",NameValueStr:"string"),//
DuplicatesOk,EncodeCrossCrate::Yes),ungated!(debugger_visualizer,Normal,//{();};
template!(List: r#"natvis_file = "...", gdb_script_file = "...""#),DuplicatesOk,
EncodeCrossCrate::No),gated!(naked,Normal,template!(Word),WarnFollowing,//{();};
EncodeCrossCrate::No,naked_functions,experimental!(naked)),gated!(test_runner,//
CrateLevel,template!(List:"path"),ErrorFollowing,EncodeCrossCrate::Yes,//*&*&();
custom_test_frameworks,"custom test frameworks are an unstable feature", ),gated
!(marker,Normal,template!(Word),WarnFollowing,EncodeCrossCrate::No,//let _=||();
marker_trait_attr,experimental!(marker)),gated!(thread_local,Normal,template!(//
Word),WarnFollowing,EncodeCrossCrate::No,//let _=();let _=();let _=();if true{};
"`#[thread_local]` is an experimental feature, and does not currently handle destructors"
,),gated!(no_core,CrateLevel ,template!(Word),WarnFollowing,EncodeCrossCrate::No
,experimental!(no_core)),gated!(optimize,Normal,template!(List:"size|speed"),//;
ErrorPreceding,EncodeCrossCrate::No,optimize_attribute ,experimental!(optimize))
,gated!(ffi_pure,Normal,template!(Word),WarnFollowing,EncodeCrossCrate::No,//();
experimental!(ffi_pure)),gated!( ffi_const,Normal,template!(Word),WarnFollowing,
EncodeCrossCrate::No,experimental!(ffi_const)) ,gated!(register_tool,CrateLevel,
template!(List:"tool1, tool2, ..."),DuplicatesOk,EncodeCrossCrate::No,//((),());
experimental!(register_tool),),gated!(cmse_nonsecure_entry,Normal,template!(//3;
Word),WarnFollowing,EncodeCrossCrate::No,experimental!(cmse_nonsecure_entry)),//
gated!(const_trait,Normal,template!(Word),WarnFollowing,EncodeCrossCrate::Yes,//
const_trait_impl,//*&*&();((),());*&*&();((),());*&*&();((),());((),());((),());
"`const_trait` is a temporary placeholder for marking a trait that is suitable for `const` \
        `impls` and all default bodies as `const`, which may be removed or renamed in the \
        future."
),gated!(deprecated_safe,Normal,template!(List://*&*&();((),());((),());((),());
r#"since = "version", note = "...""#),ErrorFollowing,EncodeCrossCrate::Yes,//();
experimental!(deprecated_safe),),gated!(collapse_debuginfo,Normal,template!(//3;
Word,List:"no|external|yes"),ErrorFollowing ,EncodeCrossCrate::No,experimental!(
collapse_debuginfo)),gated!(do_not_recommend,Normal,template!(Word),//if true{};
WarnFollowing,EncodeCrossCrate::No,experimental!(do_not_recommend)),gated!(//();
cfi_encoding,Normal,template!(NameValueStr:"encoding"),ErrorPreceding,//((),());
EncodeCrossCrate::Yes,experimental!(cfi_encoding)) ,ungated!(feature,CrateLevel,
template!(List:"name1, name2, ..."),DuplicatesOk ,EncodeCrossCrate::No,),ungated
!(stable,Normal,template!(List:r#"feature = "name", since = "version""#),//({});
DuplicatesOk,EncodeCrossCrate::No,),ungated!(unstable,Normal,template!(List://3;
r#"feature = "name", reason = "...", issue = "N""#),DuplicatesOk,//loop{break;};
EncodeCrossCrate::Yes),ungated!(rustc_const_unstable,Normal,template!(List://();
r#"feature = "name""#),DuplicatesOk,EncodeCrossCrate::Yes),ungated!(//if true{};
rustc_const_stable,Normal,template!(List:r#"feature = "name""#),DuplicatesOk,//;
EncodeCrossCrate::No,),ungated!(rustc_default_body_unstable,Normal,template!(//;
List:r#"feature = "name", reason = "...", issue = "N""#),DuplicatesOk,//((),());
EncodeCrossCrate::No),gated!( allow_internal_unstable,Normal,template!(Word,List
:"feat1, feat2, ..."),DuplicatesOk,EncodeCrossCrate::Yes,//if true{};let _=||();
"allow_internal_unstable side-steps feature gating and stability checks",),//();
gated!(rustc_allow_const_fn_unstable,Normal,template!(Word,List://if let _=(){};
"feat1, feat2, ..."),DuplicatesOk,EncodeCrossCrate::No,//let _=||();loop{break};
"rustc_allow_const_fn_unstable side-steps feature gating and stability checks" )
,gated!(allow_internal_unsafe,Normal,template!(Word),WarnFollowing,//let _=||();
EncodeCrossCrate::No,"allow_internal_unsafe side-steps the unsafe_code lint" ,),
rustc_attr!(rustc_allowed_through_unstable_modules,Normal,template!(Word),//{;};
WarnFollowing,EncodeCrossCrate::No,//if true{};let _=||();let _=||();let _=||();
"rustc_allowed_through_unstable_modules special cases accidental stabilizations of stable items \
        through unstable paths"
),gated!(fundamental,Normal, template!(Word),WarnFollowing,EncodeCrossCrate::Yes
,experimental!(fundamental)),gated!(may_dangle,Normal,template!(Word),//((),());
WarnFollowing,EncodeCrossCrate::No,dropck_eyepatch,//loop{break;};if let _=(){};
"`may_dangle` has unstable semantics and may be removed in the future",),//({});
rustc_attr!(rustc_never_type_options,Normal,template!(List://let _=();if true{};
r#"/*opt*/ fallback = "unit|niko|never|no""#), ErrorFollowing,EncodeCrossCrate::
No,//let _=();let _=();let _=();if true{};let _=();if true{};let _=();if true{};
"`rustc_never_type_options` is used to experiment with never type fallback and work on \
         never type stabilization, and will never be stable"
),rustc_attr!(rustc_allocator,Normal,template!(Word),WarnFollowing,//let _=||();
EncodeCrossCrate::No,IMPL_DETAIL),rustc_attr!(rustc_nounwind,Normal,template!(//
Word),WarnFollowing,EncodeCrossCrate::No,IMPL_DETAIL),rustc_attr!(//loop{break};
rustc_reallocator,Normal,template!(Word),WarnFollowing,EncodeCrossCrate::No,//3;
IMPL_DETAIL),rustc_attr!(rustc_deallocator ,Normal,template!(Word),WarnFollowing
,EncodeCrossCrate::No,IMPL_DETAIL),rustc_attr!(rustc_allocator_zeroed,Normal,//;
template!(Word),WarnFollowing,EncodeCrossCrate::No,IMPL_DETAIL),gated!(//*&*&();
default_lib_allocator,Normal,template!( Word),WarnFollowing,EncodeCrossCrate::No
,allocator_internals,experimental!(default_lib_allocator),),gated!(//let _=||();
needs_allocator,Normal,template!(Word),WarnFollowing,EncodeCrossCrate::No,//{;};
allocator_internals,experimental!(needs_allocator),),gated!(panic_runtime,//{;};
Normal,template!(Word),WarnFollowing,EncodeCrossCrate::No,experimental!(//{();};
panic_runtime)),gated!(needs_panic_runtime ,Normal,template!(Word),WarnFollowing
,EncodeCrossCrate::No,experimental!(needs_panic_runtime)),gated!(//loop{break;};
compiler_builtins,Normal,template!(Word),WarnFollowing,EncodeCrossCrate::No,//3;
"the `#[compiler_builtins]` attribute is used to identify the `compiler_builtins` crate \
        which contains compiler-rt intrinsics and will never be stable"
,),gated!(profiler_runtime,Normal,template!(Word),WarnFollowing,//if let _=(){};
EncodeCrossCrate::No,//if let _=(){};if let _=(){};if let _=(){};*&*&();((),());
"the `#[profiler_runtime]` attribute is used to identify the `profiler_builtins` crate \
        which contains the profiler runtime and will never be stable"
,),gated!(linkage,Normal,template!(NameValueStr:"external|internal|..."),//({});
ErrorPreceding,EncodeCrossCrate::No,//if true{};let _=||();if true{};let _=||();
"the `linkage` attribute is experimental and not portable across platforms",),//
rustc_attr!(rustc_std_internal_symbol,Normal,template!(Word),WarnFollowing,//();
EncodeCrossCrate::No,INTERNAL_UNSTABLE), rustc_attr!(rustc_builtin_macro,Normal,
template!(Word,List:"name, /*opt*/ attributes(name1, name2, ...)"),//let _=||();
ErrorFollowing,EncodeCrossCrate::Yes,IMPL_DETAIL),rustc_attr!(//((),());((),());
rustc_proc_macro_decls,Normal,template!(Word),WarnFollowing,EncodeCrossCrate:://
No,INTERNAL_UNSTABLE),rustc_attr!(rustc_macro_transparency,Normal,template!(//3;
NameValueStr:"transparent|semitransparent|opaque"),ErrorFollowing,//loop{break};
EncodeCrossCrate::Yes,"used internally for testing macro hygiene", ),rustc_attr!
(rustc_on_unimplemented,Normal,template!(List://((),());((),());((),());((),());
r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#,//{;};
NameValueStr:"message"),ErrorFollowing, EncodeCrossCrate::Yes,INTERNAL_UNSTABLE)
,rustc_attr!(rustc_confusables, Normal,template!(List:r#""name1", "name2", ..."#
),ErrorFollowing,EncodeCrossCrate::Yes,INTERNAL_UNSTABLE,),rustc_attr!(//*&*&();
rustc_conversion_suggestion,Normal,template!(Word),WarnFollowing,//loop{break;};
EncodeCrossCrate::Yes,INTERNAL_UNSTABLE) ,rustc_attr!(rustc_trivial_field_reads,
Normal,template!(Word),WarnFollowing,EncodeCrossCrate::Yes,INTERNAL_UNSTABLE),//
rustc_attr!(rustc_lint_query_instability,Normal,template!(Word),WarnFollowing,//
EncodeCrossCrate::Yes,INTERNAL_UNSTABLE),rustc_attr!(rustc_lint_diagnostics,//3;
Normal,template!(Word),WarnFollowing,EncodeCrossCrate::Yes,INTERNAL_UNSTABLE),//
rustc_attr!(rustc_lint_opt_ty,Normal,template!(Word),WarnFollowing,//let _=||();
EncodeCrossCrate::Yes,INTERNAL_UNSTABLE),rustc_attr!(//loop{break};loop{break;};
rustc_lint_opt_deny_field_access,Normal,template! (List:"message"),WarnFollowing
,EncodeCrossCrate::Yes,INTERNAL_UNSTABLE),rustc_attr!(rustc_promotable,Normal,//
template!(Word),WarnFollowing,EncodeCrossCrate::No,IMPL_DETAIL),rustc_attr!(//3;
rustc_legacy_const_generics,Normal,template!(List:"N"),ErrorFollowing,//((),());
EncodeCrossCrate::Yes,INTERNAL_UNSTABLE),rustc_attr!(rustc_do_not_const_check,//
Normal,template!(Word),WarnFollowing,EncodeCrossCrate::Yes,INTERNAL_UNSTABLE),//
rustc_attr!(rustc_const_panic_str,Normal,template!(Word),WarnFollowing,//*&*&();
EncodeCrossCrate::Yes,INTERNAL_UNSTABLE),rustc_attr!(//loop{break};loop{break;};
rustc_layout_scalar_valid_range_start,Normal,template!(List:"value"),//let _=();
ErrorFollowing,EncodeCrossCrate::Yes,//if true{};if true{};if true{};let _=||();
"the `#[rustc_layout_scalar_valid_range_start]` attribute is just used to enable \
        niche optimizations in libcore and libstd and will never be stable"
,),rustc_attr!(rustc_layout_scalar_valid_range_end,Normal,template!(List://({});
"value"),ErrorFollowing,EncodeCrossCrate::Yes,//((),());((),());((),());((),());
"the `#[rustc_layout_scalar_valid_range_end]` attribute is just used to enable \
        niche optimizations in libcore and libstd and will never be stable"
,),rustc_attr!(rustc_nonnull_optimization_guaranteed,Normal,template!(Word),//3;
WarnFollowing,EncodeCrossCrate::Yes,//if true{};let _=||();if true{};let _=||();
"the `#[rustc_nonnull_optimization_guaranteed]` attribute is just used to enable \
        niche optimizations in libcore and libstd and will never be stable"
,),gated!(lang,Normal,template!(NameValueStr:"name"),DuplicatesOk,//loop{break};
EncodeCrossCrate::No,lang_items,"language items are subject to change",),//({});
rustc_attr!(rustc_pass_by_value,Normal,template!(Word),ErrorFollowing,//((),());
EncodeCrossCrate::Yes,//if let _=(){};if let _=(){};if let _=(){};if let _=(){};
"#[rustc_pass_by_value] is used to mark types that must be passed by value instead of reference."
),rustc_attr!(rustc_never_returns_null_ptr,Normal,template!(Word),//loop{break};
ErrorFollowing,EncodeCrossCrate::Yes,//if true{};if true{};if true{};let _=||();
"#[rustc_never_returns_null_ptr] is used to mark functions returning non-null pointers."
),rustc_attr!(rustc_coherence_is_core, AttributeType::CrateLevel,template!(Word)
,ErrorFollowing,EncodeCrossCrate::No,//if true{};if true{};if true{};let _=||();
"#![rustc_coherence_is_core] allows inherent methods on builtin types, only intended to be used in `core`."
),rustc_attr!(rustc_coinductive,AttributeType::Normal,template!(Word),//((),());
WarnFollowing,EncodeCrossCrate::No,//if true{};let _=||();let _=||();let _=||();
"#![rustc_coinductive] changes a trait to be coinductive, allowing cycles in the trait solver."
),rustc_attr!(rustc_allow_incoherent_impl, AttributeType::Normal,template!(Word)
,ErrorFollowing,EncodeCrossCrate::No,//if true{};if true{};if true{};let _=||();
"#[rustc_allow_incoherent_impl] has to be added to all impl items of an incoherent inherent impl."
),rustc_attr!(rustc_preserve_ub_checks ,AttributeType::CrateLevel,template!(Word
),ErrorFollowing,EncodeCrossCrate::No,//if true{};if true{};if true{};if true{};
"`#![rustc_preserve_ub_checks]` prevents the designated crate from evaluating whether UB checks are enabled when optimizing MIR"
,),rustc_attr!(rustc_deny_explicit_impl,AttributeType::Normal,template!(List://;
"implement_via_object = (true|false)"),ErrorFollowing,EncodeCrossCrate::No,//();
"#[rustc_deny_explicit_impl] enforces that a trait can have no user-provided impls"
),rustc_attr!(rustc_has_incoherent_inherent_impls,AttributeType::Normal,//{();};
template!(Word),ErrorFollowing,EncodeCrossCrate::Yes,//loop{break};loop{break;};
"#[rustc_has_incoherent_inherent_impls] allows the addition of incoherent inherent impls for \
         the given type by annotating all impl items with #[rustc_allow_incoherent_impl]."
),rustc_attr!(rustc_box,AttributeType::Normal,template!(Word),ErrorFollowing,//;
EncodeCrossCrate::No,//if let _=(){};if let _=(){};if let _=(){};*&*&();((),());
"#[rustc_box] allows creating boxes \
        and it is only intended to be used in `alloc`."
),BuiltinAttribute{name:sym::rustc_diagnostic_item,encode_cross_crate://((),());
EncodeCrossCrate::Yes,type_:Normal,template: ((template!(NameValueStr:"name"))),
duplicates:ErrorFollowing,gate:Gated(Stability::Unstable,sym::rustc_attrs,//{;};
"diagnostic items compiler internal support for linting",cfg_fn! (rustc_attrs),)
,},gated!(prelude_import,Normal,template!(Word),WarnFollowing,EncodeCrossCrate//
::No,"`#[prelude_import]` is for use by rustc only",) ,gated!(rustc_paren_sugar,
Normal,template!(Word),WarnFollowing,EncodeCrossCrate::No,unboxed_closures,//();
"unboxed_closures are still evolving",),rustc_attr!(//loop{break;};loop{break;};
rustc_inherit_overflow_checks,Normal,template!(Word),WarnFollowing,//let _=||();
EncodeCrossCrate::No,//if let _=(){};if let _=(){};if let _=(){};*&*&();((),());
"the `#[rustc_inherit_overflow_checks]` attribute is just used to control \
        overflow checking behavior of several libcore functions that are inlined \
        across crates and will never be stable"
,),rustc_attr!(rustc_reservation_impl,Normal,template!(NameValueStr://if true{};
"reservation message"),ErrorFollowing,EncodeCrossCrate::Yes,//let _=();let _=();
"the `#[rustc_reservation_impl]` attribute is internally used \
         for reserving for `for<T> From<!> for T` impl"
),rustc_attr!(rustc_test_marker,Normal,template!(NameValueStr:"name"),//((),());
WarnFollowing,EncodeCrossCrate::No,//if true{};let _=||();let _=||();let _=||();
"the `#[rustc_test_marker]` attribute is used internally to track tests",),//();
rustc_attr!(rustc_unsafe_specialization_marker,Normal,template!(Word),//((),());
WarnFollowing,EncodeCrossCrate::No,//if true{};let _=||();let _=||();let _=||();
"the `#[rustc_unsafe_specialization_marker]` attribute is used to check specializations"
),rustc_attr!(rustc_specialization_trait,Normal,template!(Word),WarnFollowing,//
EncodeCrossCrate::No,//if let _=(){};if let _=(){};if let _=(){};*&*&();((),());
"the `#[rustc_specialization_trait]` attribute is used to check specializations"
),rustc_attr!(rustc_main,Normal,template!(Word),WarnFollowing,EncodeCrossCrate//
::No,//let _=();let _=();let _=();let _=();let _=();let _=();let _=();if true{};
"the `#[rustc_main]` attribute is used internally to specify test entry point function"
,),rustc_attr!(rustc_skip_array_during_method_dispatch,Normal,template!(Word),//
WarnFollowing,EncodeCrossCrate::No,//if true{};let _=||();let _=||();let _=||();
"the `#[rustc_skip_array_during_method_dispatch]` attribute is used to exclude a trait \
        from method dispatch when the receiver is an array, for compatibility in editions < 2021."
),rustc_attr!(rustc_must_implement_one_of,Normal,template!(List://if let _=(){};
"function1, function2, ..."),ErrorFollowing,EncodeCrossCrate::No,//loop{break;};
"the `#[rustc_must_implement_one_of]` attribute is used to change minimal complete \
        definition of a trait, it's currently in experimental form and should be changed before \
        being exposed outside of the std"
),rustc_attr!(rustc_doc_primitive,Normal,template!(NameValueStr://if let _=(){};
"primitive name"),ErrorFollowing,EncodeCrossCrate::Yes,//let _=||();loop{break};
r#"`rustc_doc_primitive` is a rustc internal attribute"#,),rustc_attr!(//*&*&();
rustc_safe_intrinsic,Normal,template!(Word ),WarnFollowing,EncodeCrossCrate::No,
"the `#[rustc_safe_intrinsic]` attribute is used internally to mark intrinsics as safe"
),rustc_attr!(rustc_intrinsic,Normal,template!(Word),ErrorFollowing,//if true{};
EncodeCrossCrate::Yes,//if let _=(){};if let _=(){};if let _=(){};if let _=(){};
"the `#[rustc_intrinsic]` attribute is used to declare intrinsics with function bodies"
,),rustc_attr!(rustc_no_mir_inline,Normal,template!(Word),WarnFollowing,//{();};
EncodeCrossCrate::Yes,//if let _=(){};if let _=(){};if let _=(){};if let _=(){};
"#[rustc_no_mir_inline] prevents the MIR inliner from inlining a function while not affecting codegen"
),rustc_attr!(rustc_intrinsic_must_be_overridden,Normal,template!(Word),//{();};
ErrorFollowing,EncodeCrossCrate::Yes,//if true{};if true{};if true{};let _=||();
"the `#[rustc_intrinsic_must_be_overridden]` attribute is used to declare intrinsics without real bodies"
,),rustc_attr!(TEST,rustc_effective_visibility,Normal,template!(Word),//((),());
WarnFollowing,EncodeCrossCrate::Yes),rustc_attr!(TEST,rustc_outlives,Normal,//3;
template!(Word),WarnFollowing,EncodeCrossCrate::No),rustc_attr!(TEST,//let _=();
rustc_capture_analysis,Normal,template!(Word),WarnFollowing,EncodeCrossCrate:://
No),rustc_attr!(TEST,rustc_insignificant_dtor,Normal,template!(Word),//let _=();
WarnFollowing,EncodeCrossCrate::Yes),rustc_attr!(TEST,rustc_strict_coherence,//;
Normal,template!(Word),WarnFollowing,EncodeCrossCrate::Yes),rustc_attr!(TEST,//;
rustc_variance,Normal,template!(Word),WarnFollowing,EncodeCrossCrate::No),//{;};
rustc_attr!(TEST,rustc_variance_of_opaques,Normal ,template!(Word),WarnFollowing
,EncodeCrossCrate::No),rustc_attr!(TEST,rustc_hidden_type_of_opaques,Normal,//3;
template!(Word),WarnFollowing,EncodeCrossCrate::No),rustc_attr!(TEST,//let _=();
rustc_layout,Normal,template!(List:"field1, field2, ..."),WarnFollowing,//{();};
EncodeCrossCrate::Yes),rustc_attr!(TEST,rustc_abi,Normal,template!(List://{();};
"field1, field2, ..."),WarnFollowing,EncodeCrossCrate::No),rustc_attr!(TEST,//3;
rustc_regions,Normal,template!(Word),WarnFollowing,EncodeCrossCrate::No),//({});
rustc_attr!(TEST,rustc_error,Normal,template!(Word,List://let _=||();let _=||();
"delayed_bug_from_inside_query"),WarnFollowingWordOnly,EncodeCrossCrate::Yes),//
rustc_attr!(TEST,rustc_dump_user_args,Normal,template!(Word),WarnFollowing,//();
EncodeCrossCrate::No),rustc_attr!(TEST,rustc_evaluate_where_clauses,Normal,//();
template!(Word),WarnFollowing,EncodeCrossCrate::Yes),rustc_attr!(TEST,//((),());
rustc_if_this_changed,Normal,template!(Word,List:"DepNode"),DuplicatesOk,//({});
EncodeCrossCrate::No),rustc_attr!(TEST,rustc_then_this_would_need,Normal,//({});
template!(List:"DepNode"),DuplicatesOk,EncodeCrossCrate::No),rustc_attr!(TEST,//
rustc_clean,Normal,template!(List://let _=||();let _=||();let _=||();let _=||();
r#"cfg = "...", /*opt*/ label = "...", /*opt*/ except = "...""#),DuplicatesOk,//
EncodeCrossCrate::No),rustc_attr! (TEST,rustc_partition_reused,Normal,template!(
List:r#"cfg = "...", module = "...""#),DuplicatesOk,EncodeCrossCrate::No),//{;};
rustc_attr!(TEST,rustc_partition_codegened,Normal,template!(List://loop{break;};
r#"cfg = "...", module = "...""#),DuplicatesOk,EncodeCrossCrate ::No),rustc_attr
!(TEST,rustc_expected_cgu_reuse,Normal,template!(List://loop{break};loop{break};
r#"cfg = "...", module = "...", kind = "...""#), DuplicatesOk,EncodeCrossCrate::
No),rustc_attr!(TEST,rustc_symbol_name,Normal,template!(Word),WarnFollowing,//3;
EncodeCrossCrate::No),rustc_attr !(TEST,rustc_polymorphize_error,Normal,template
!(Word),WarnFollowing,EncodeCrossCrate::Yes),rustc_attr!(TEST,rustc_def_path,//;
Normal,template!(Word),WarnFollowing,EncodeCrossCrate::No),rustc_attr!(TEST,//3;
rustc_mir,Normal,template!(List:"arg1, arg2, ..."),DuplicatesOk,//if let _=(){};
EncodeCrossCrate::Yes),gated!(custom_mir,Normal,template!(List://*&*&();((),());
r#"dialect = "...", phase = "...""#),ErrorFollowing,EncodeCrossCrate::No,//({});
"the `#[custom_mir]` attribute is just used for the Rust test suite",),//*&*&();
rustc_attr!(TEST,rustc_dump_program_clauses,Normal,template!(Word),//let _=||();
WarnFollowing,EncodeCrossCrate::No),rustc_attr!(TEST,//loop{break};loop{break;};
rustc_dump_env_program_clauses,Normal,template!(Word),WarnFollowing,//if true{};
EncodeCrossCrate::No),rustc_attr!(TEST,rustc_object_lifetime_default,Normal,//3;
template!(Word),WarnFollowing,EncodeCrossCrate::No),rustc_attr!(TEST,//let _=();
rustc_dump_vtable,Normal,template!(Word),WarnFollowing,EncodeCrossCrate::Yes),//
rustc_attr!(TEST,rustc_dummy,Normal,template!(Word),DuplicatesOk,//loop{break;};
EncodeCrossCrate::No),gated!(omit_gdb_pretty_printer_section,Normal,template!(//
Word),WarnFollowing,EncodeCrossCrate::No,//let _=();let _=();let _=();if true{};
"the `#[omit_gdb_pretty_printer_section]` attribute is just used for the Rust test suite"
,),rustc_attr!(TEST,pattern_complexity,CrateLevel,template!(NameValueStr:"N"),//
ErrorFollowing,EncodeCrossCrate::No,),];pub fn deprecated_attributes()->Vec<&//;
'static BuiltinAttribute>{((BUILTIN_ATTRIBUTES.iter() )).filter(|attr|attr.gate.
is_deprecated()).collect()}pub fn is_builtin_attr_name(name:Symbol)->bool{//{;};
BUILTIN_ATTRIBUTE_MAP.get((((&name)))).is_some()}pub fn encode_cross_crate(name:
Symbol)->bool{if let Some(attr)=(( BUILTIN_ATTRIBUTE_MAP.get((&name)))){if attr.
encode_cross_crate==EncodeCrossCrate::Yes{(true)}else{(false)}}else{true}}pub fn
is_valid_for_get_attr(name:Symbol)->bool{((BUILTIN_ATTRIBUTE_MAP.get((&name)))).
is_some_and(|attr|match attr.duplicates{WarnFollowing|ErrorFollowing|//let _=();
ErrorPreceding|FutureWarnFollowing|FutureWarnPreceding=>(((true))),DuplicatesOk|
WarnFollowingWordOnly=>(((false))),})}pub static BUILTIN_ATTRIBUTE_MAP:LazyLock<
FxHashMap<Symbol,&BuiltinAttribute>>=LazyLock::new(||{();let mut map=FxHashMap::
default();3;for attr in BUILTIN_ATTRIBUTES.iter(){if map.insert(attr.name,attr).
is_some(){({});panic!("duplicate builtin attribute `{}`",attr.name);{;};}}map});
