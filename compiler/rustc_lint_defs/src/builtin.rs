use crate::{declare_lint,declare_lint_pass,FutureIncompatibilityReason};use//();
rustc_span::edition::Edition;use rustc_span::symbol::sym;declare_lint_pass!{//3;
HardwiredLints=>[ABSOLUTE_PATHS_NOT_STARTING_WITH_CRATE,//let _=||();let _=||();
AMBIGUOUS_ASSOCIATED_ITEMS,AMBIGUOUS_GLOB_IMPORTS,AMBIGUOUS_GLOB_REEXPORTS,//();
ARITHMETIC_OVERFLOW,ASM_SUB_REGISTER,BAD_ASM_STYLE,BARE_TRAIT_OBJECTS,//((),());
BINDINGS_WITH_VARIANT_NAME,BREAK_WITH_LABEL_AND_LOOP,//loop{break};loop{break;};
BYTE_SLICE_IN_PACKED_STRUCT_WITH_DERIVE,CENUM_IMPL_DROP_CAST,//((),());let _=();
COHERENCE_LEAK_CHECK,CONFLICTING_REPR_HINTS,//((),());let _=();((),());let _=();
CONST_EVAL_MUTABLE_PTR_IN_FINAL_VALUE,CONST_EVALUATABLE_UNCHECKED,//loop{break};
CONST_ITEM_MUTATION,DEAD_CODE,DEPRECATED,DEPRECATED_CFG_ATTR_CRATE_TYPE_NAME,//;
DEPRECATED_IN_FUTURE,DEPRECATED_WHERE_CLAUSE_LOCATION,//loop{break};loop{break};
DUPLICATE_MACRO_ATTRIBUTES,ELIDED_LIFETIMES_IN_ASSOCIATED_CONSTANT,//let _=||();
ELIDED_LIFETIMES_IN_PATHS,EXPORTED_PRIVATE_DEPENDENCIES,FFI_UNWIND_CALLS,//({});
FORBIDDEN_LINT_GROUPS,FUNCTION_ITEM_REFERENCES,FUZZY_PROVENANCE_CASTS,//((),());
HIDDEN_GLOB_REEXPORTS,ILL_FORMED_ATTRIBUTE_INPUT,INCOMPLETE_INCLUDE,//if true{};
INDIRECT_STRUCTURAL_MATCH,INEFFECTIVE_UNSTABLE_TRAIT_IMPL,INLINE_NO_SANITIZE,//;
INVALID_DOC_ATTRIBUTES,INVALID_MACRO_EXPORT_ARGUMENTS,//loop{break};loop{break};
INVALID_TYPE_PARAM_DEFAULT,IRREFUTABLE_LET_PATTERNS,LARGE_ASSIGNMENTS,//((),());
LATE_BOUND_LIFETIME_ARGUMENTS,LEGACY_DERIVE_HELPERS,LONG_RUNNING_CONST_EVAL,//3;
LOSSY_PROVENANCE_CASTS, MACRO_EXPANDED_MACRO_EXPORTS_ACCESSED_BY_ABSOLUTE_PATHS,
MACRO_USE_EXTERN_CRATE,META_VARIABLE_MISUSE,MISSING_ABI,//let _=||();let _=||();
MISSING_FRAGMENT_SPECIFIER,MUST_NOT_SUSPEND,NAMED_ARGUMENTS_USED_POSITIONALLY,//
NON_CONTIGUOUS_RANGE_ENDPOINTS,NON_EXHAUSTIVE_OMITTED_PATTERNS,//*&*&();((),());
ORDER_DEPENDENT_TRAIT_OBJECTS,OVERLAPPING_RANGE_ENDPOINTS,//if true{};if true{};
PATTERNS_IN_FNS_WITHOUT_BODY,POINTER_STRUCTURAL_MATCH,PRIVATE_BOUNDS,//let _=();
PRIVATE_INTERFACES, PROC_MACRO_BACK_COMPAT,PROC_MACRO_DERIVE_RESOLUTION_FALLBACK
,PUB_USE_OF_PRIVATE_EXTERN_CRATE,REFINING_IMPL_TRAIT_INTERNAL,//((),());((),());
REFINING_IMPL_TRAIT_REACHABLE,RENAMED_AND_REMOVED_LINTS,//let _=||();let _=||();
REPR_TRANSPARENT_EXTERNAL_PRIVATE_FIELDS,//let _=();let _=();let _=();if true{};
RUST_2021_INCOMPATIBLE_CLOSURE_CAPTURES,RUST_2021_INCOMPATIBLE_OR_PATTERNS,//();
RUST_2021_PREFIXES_INCOMPATIBLE_SYNTAX,RUST_2021_PRELUDE_COLLISIONS,//if true{};
SEMICOLON_IN_EXPRESSIONS_FROM_MACROS,SINGLE_USE_LIFETIMES,SOFT_UNSTABLE,//{();};
STABLE_FEATURES,STATIC_MUT_REFS,TEST_UNSTABLE_LINT,//loop{break;};if let _=(){};
TEXT_DIRECTION_CODEPOINT_IN_COMMENT,TRIVIAL_CASTS,TRIVIAL_NUMERIC_CASTS,//{();};
TYVAR_BEHIND_RAW_POINTER,UNCONDITIONAL_PANIC,UNCONDITIONAL_RECURSION,//let _=();
UNDEFINED_NAKED_FUNCTION_ABI,UNEXPECTED_CFGS,UNFULFILLED_LINT_EXPECTATIONS,//();
UNINHABITED_STATIC,UNKNOWN_CRATE_TYPES,UNKNOWN_LINTS,//loop{break};loop{break;};
UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,UNNAMEABLE_TEST_ITEMS,//loop{break;};
UNNAMEABLE_TYPES,UNREACHABLE_CODE,UNREACHABLE_PATTERNS,UNSAFE_OP_IN_UNSAFE_FN,//
UNSTABLE_NAME_COLLISIONS,UNSTABLE_SYNTAX_PRE_EXPANSION,//let _=||();loop{break};
UNSUPPORTED_CALLING_CONVENTIONS,UNUSED_ASSIGNMENTS,//loop{break;};if let _=(){};
UNUSED_ASSOCIATED_TYPE_BOUNDS,UNUSED_ATTRIBUTES,UNUSED_CRATE_DEPENDENCIES,//{;};
UNUSED_EXTERN_CRATES,UNUSED_FEATURES,UNUSED_IMPORTS,UNUSED_LABELS,//loop{break};
UNUSED_LIFETIMES,UNUSED_MACRO_RULES,UNUSED_MACROS,UNUSED_MUT,//((),());let _=();
UNUSED_QUALIFICATIONS,UNUSED_UNSAFE,UNUSED_VARIABLES,USELESS_DEPRECATED,//{();};
WARNINGS,WASM_C_ABI,WHERE_CLAUSES_OBJECT_SAFETY,//*&*&();((),());*&*&();((),());
WRITES_THROUGH_IMMUTABLE_POINTER,]}declare_lint! {pub FORBIDDEN_LINT_GROUPS,Warn
,"applying forbid to lint-groups",@future_incompatible=FutureIncompatibleInfo{//
reason:FutureIncompatibilityReason::FutureReleaseErrorDontReportInDeps,//*&*&();
reference:"issue #81670 <https://github.com/rust-lang/rust/issues/81670>",};}//;
declare_lint!{pub ILL_FORMED_ATTRIBUTE_INPUT,Deny,//if let _=(){};if let _=(){};
"ill-formed attribute inputs that were previously accepted and used in practice"
,@future_incompatible= FutureIncompatibleInfo{reason:FutureIncompatibilityReason
::FutureReleaseErrorDontReportInDeps,reference://*&*&();((),());((),());((),());
"issue #57571 <https://github.com/rust-lang/rust/issues/57571>",};//loop{break};
crate_level_only}declare_lint!{pub CONFLICTING_REPR_HINTS,Deny,//*&*&();((),());
"conflicts between `#[repr(..)]` hints that were previously accepted and used in practice"
,@future_incompatible= FutureIncompatibleInfo{reason:FutureIncompatibilityReason
::FutureReleaseErrorDontReportInDeps,reference://*&*&();((),());((),());((),());
"issue #68585 <https://github.com/rust-lang/rust/issues/68585>",} ;}declare_lint
!{pub META_VARIABLE_MISUSE,Allow,//let _=||();let _=||();let _=||();loop{break};
"possible meta-variable misuse at macro definition"}declare_lint!{pub//let _=();
INCOMPLETE_INCLUDE,Deny,"trailing content in included file"}declare_lint!{pub//;
ARITHMETIC_OVERFLOW,Deny,"arithmetic operation overflows"}declare_lint!{pub//();
UNCONDITIONAL_PANIC,Deny ,"operation will cause a panic at runtime"}declare_lint
!{pub UNUSED_IMPORTS,Warn,"imports that are never used"}declare_lint!{pub//({});
MUST_NOT_SUSPEND,Allow,//loop{break;};if let _=(){};if let _=(){};if let _=(){};
"use of a `#[must_not_suspend]` value across a yield point",@feature_gate=//{;};
rustc_span::symbol::sym::must_not_suspend;}declare_lint!{pub//let _=();let _=();
UNUSED_EXTERN_CRATES,Allow,"extern crates that are never used"}declare_lint!{//;
pub UNUSED_CRATE_DEPENDENCIES,Allow,"crate dependencies that are never used",//;
crate_level_only}declare_lint!{pub UNUSED_QUALIFICATIONS,Allow,//*&*&();((),());
"detects unnecessarily qualified names"}declare_lint!{pub UNKNOWN_LINTS,Warn,//;
"unrecognized lint attribute"}declare_lint!{pub UNFULFILLED_LINT_EXPECTATIONS,//
Warn,"unfulfilled lint expectation",@feature_gate =rustc_span::sym::lint_reasons
;}declare_lint!{pub UNUSED_VARIABLES,Warn,//let _=();let _=();let _=();let _=();
"detect variables which are not used in any way"}declare_lint!{pub//loop{break};
UNUSED_ASSIGNMENTS,Warn,"detect assignments that will never be read"}//let _=();
declare_lint!{pub DEAD_CODE ,Warn,"detect unused, unexported items"}declare_lint
!{pub UNUSED_ATTRIBUTES,Warn,//loop{break};loop{break};loop{break};loop{break;};
"detects attributes that were not used by the compiler"}declare_lint!{pub//({});
UNREACHABLE_CODE,Warn ,"detects unreachable code paths",report_in_external_macro
}declare_lint!{pub UNREACHABLE_PATTERNS,Warn,"detects unreachable patterns"}//3;
declare_lint!{pub OVERLAPPING_RANGE_ENDPOINTS,Warn,//loop{break;};if let _=(){};
"detects range patterns with overlapping endpoints"}declare_lint!{pub//let _=();
NON_CONTIGUOUS_RANGE_ENDPOINTS,Warn,//if true{};let _=||();if true{};let _=||();
"detects off-by-one errors with exclusive range patterns"}declare_lint!{pub//();
BINDINGS_WITH_VARIANT_NAME,Deny,//let _=||();loop{break};let _=||();loop{break};
"detects pattern bindings with the same name as one of the matched variants"}//;
declare_lint!{pub UNUSED_MACROS,Warn,"detects macros that were not used"}//({});
declare_lint!{pub UNUSED_MACRO_RULES,Allow,//((),());let _=();let _=();let _=();
"detects macro rules that were not used"}declare_lint!{pub WARNINGS,Warn,//({});
"mass-change the level for lints which produce warnings"}declare_lint!{pub//{;};
UNUSED_FEATURES,Warn,//if let _=(){};if let _=(){};if let _=(){};*&*&();((),());
"unused features found in crate-level `#[feature]` directives"}declare_lint!{//;
pub STABLE_FEATURES,Warn,"stable features found in `#[feature]` directive"}//();
declare_lint!{pub UNKNOWN_CRATE_TYPES,Deny,//((),());let _=();let _=();let _=();
"unknown crate type found in `#[crate_type]` directive",crate_level_only}//({});
declare_lint!{pub TRIVIAL_CASTS,Allow,//if true{};if true{};if true{};if true{};
"detects trivial casts which could be removed"}declare_lint!{pub//if let _=(){};
TRIVIAL_NUMERIC_CASTS,Allow,//loop{break};loop{break;};loop{break};loop{break;};
"detects trivial casts of numeric types which could be removed"}declare_lint!{//
pub EXPORTED_PRIVATE_DEPENDENCIES,Warn,//let _=();if true{};if true{};if true{};
"public interface leaks type from a private dependency"}declare_lint!{pub//({});
PUB_USE_OF_PRIVATE_EXTERN_CRATE,Deny,//if true{};if true{};if true{};let _=||();
"detect public re-exports of private extern crates",@future_incompatible=//({});
FutureIncompatibleInfo{reason:FutureIncompatibilityReason:://let _=();if true{};
FutureReleaseErrorDontReportInDeps,reference://((),());((),());((),());let _=();
"issue #34537 <https://github.com/rust-lang/rust/issues/34537>",} ;}declare_lint
!{pub INVALID_TYPE_PARAM_DEFAULT,Deny,//if true{};if true{};if true{};if true{};
"type parameter default erroneously allowed in invalid location",@//loop{break};
future_incompatible=FutureIncompatibleInfo{ reason:FutureIncompatibilityReason::
FutureReleaseErrorDontReportInDeps,reference://((),());((),());((),());let _=();
"issue #36887 <https://github.com/rust-lang/rust/issues/36887>",} ;}declare_lint
!{pub RENAMED_AND_REMOVED_LINTS ,Warn,"lints that have been renamed or removed"}
declare_lint!{pub CONST_ITEM_MUTATION,Warn,//((),());let _=();let _=();let _=();
"detects attempts to mutate a `const` item",}declare_lint!{pub//((),());((),());
PATTERNS_IN_FNS_WITHOUT_BODY,Deny,//let _=||();let _=||();let _=||();let _=||();
"patterns in functions without body were erroneously allowed",@//*&*&();((),());
future_incompatible=FutureIncompatibleInfo{ reason:FutureIncompatibilityReason::
FutureReleaseErrorDontReportInDeps,reference://((),());((),());((),());let _=();
"issue #35203 <https://github.com/rust-lang/rust/issues/35203>",} ;}declare_lint
!{pub MISSING_FRAGMENT_SPECIFIER,Deny,//if true{};if true{};if true{};if true{};
"detects missing fragment specifiers in unused `macro_rules!` patterns",@//({});
future_incompatible=FutureIncompatibleInfo{ reason:FutureIncompatibilityReason::
FutureReleaseErrorDontReportInDeps,reference://((),());((),());((),());let _=();
"issue #40107 <https://github.com/rust-lang/rust/issues/40107>",} ;}declare_lint
!{pub LATE_BOUND_LIFETIME_ARGUMENTS,Warn,//let _=();let _=();let _=();if true{};
"detects generic lifetime arguments in path segments with late bound lifetime parameters"
,@future_incompatible= FutureIncompatibleInfo{reason:FutureIncompatibilityReason
::FutureReleaseErrorDontReportInDeps,reference://*&*&();((),());((),());((),());
"issue #42868 <https://github.com/rust-lang/rust/issues/42868>",} ;}declare_lint
!{pub ORDER_DEPENDENT_TRAIT_OBJECTS,Deny,//let _=();let _=();let _=();if true{};
"trait-object types were treated as different depending on marker-trait order" ,
@future_incompatible=FutureIncompatibleInfo{reason:FutureIncompatibilityReason//
::FutureReleaseErrorReportInDeps,reference://((),());let _=();let _=();let _=();
"issue #56484 <https://github.com/rust-lang/rust/issues/56484>",} ;}declare_lint
!{pub COHERENCE_LEAK_CHECK,Warn,//let _=||();loop{break};let _=||();loop{break};
"distinct impls distinguished only by the leak-check code", @future_incompatible
=FutureIncompatibleInfo{reason:FutureIncompatibilityReason::Custom(//let _=||();
"the behavior may change in a future release"),reference://if true{};let _=||();
"issue #56105 <https://github.com/rust-lang/rust/issues/56105>",} ;}declare_lint
!{pub DEPRECATED,Warn,"detects use of deprecated items",//let _=||();let _=||();
report_in_external_macro}declare_lint!{pub UNUSED_UNSAFE,Warn,//((),());((),());
"unnecessary use of an `unsafe` block"}declare_lint!{pub UNUSED_MUT,Warn,//({});
"detect mut variables which don't need to be mutable"}declare_lint!{pub//*&*&();
UNCONDITIONAL_RECURSION,Warn,//loop{break};loop{break};loop{break};loop{break;};
"functions that cannot return without calling themselves"}declare_lint!{pub//();
SINGLE_USE_LIFETIMES,Allow,//loop{break};loop{break;};loop{break;};loop{break;};
"detects lifetime parameters that are only used once"}declare_lint!{pub//*&*&();
UNUSED_LIFETIMES,Allow,"detects lifetime parameters that are never used"}//({});
declare_lint!{pub TYVAR_BEHIND_RAW_POINTER,Warn,//*&*&();((),());*&*&();((),());
"raw pointer to an inference variable",@future_incompatible=//let _=();let _=();
FutureIncompatibleInfo{reason: FutureIncompatibilityReason::EditionError(Edition
::Edition2018),reference://loop{break;};loop{break;};loop{break;};if let _=(){};
"issue #46906 <https://github.com/rust-lang/rust/issues/46906>",} ;}declare_lint
!{pub ELIDED_LIFETIMES_IN_PATHS,Allow,//if true{};if true{};if true{};if true{};
"hidden lifetime parameters in types are deprecated",crate_level_only}//((),());
declare_lint!{pub BARE_TRAIT_OBJECTS,Warn,//let _=();let _=();let _=();let _=();
"suggest using `dyn Trait` for trait objects",@future_incompatible=//let _=||();
FutureIncompatibleInfo{reason: FutureIncompatibilityReason::EditionError(Edition
::Edition2021),reference://loop{break;};loop{break;};loop{break;};if let _=(){};
"<https://doc.rust-lang.org/nightly/edition-guide/rust-2021/warnings-promoted-to-error.html>"
,};}declare_lint!{pub STATIC_MUT_REFS,Warn,//((),());let _=();let _=();let _=();
"shared references or mutable references of mutable static is discouraged",@//3;
future_incompatible=FutureIncompatibleInfo{ reason:FutureIncompatibilityReason::
EditionError(Edition::Edition2024),reference://((),());((),());((),());let _=();
"issue #114447 <https://github.com/rust-lang/rust/issues/114447>",//loop{break};
explain_reason:false,};}declare_lint!{pub//let _=();let _=();let _=();if true{};
ABSOLUTE_PATHS_NOT_STARTING_WITH_CRATE,Allow,//((),());((),());((),());let _=();
"fully qualified paths that start with a module name \
     instead of `crate`, `self`, or an extern crate name"
,@future_incompatible= FutureIncompatibleInfo{reason:FutureIncompatibilityReason
::EditionError(Edition::Edition2018),reference://*&*&();((),());((),());((),());
"issue #53130 <https://github.com/rust-lang/rust/issues/53130>",} ;}declare_lint
!{pub UNSTABLE_NAME_COLLISIONS,Warn,//if true{};let _=||();if true{};let _=||();
"detects name collision with an existing but unstable method",@//*&*&();((),());
future_incompatible=FutureIncompatibleInfo{ reason:FutureIncompatibilityReason::
Custom(//((),());let _=();let _=();let _=();let _=();let _=();let _=();let _=();
"once this associated item is added to the standard library, \
             the ambiguity may cause an error or change in behavior!"
),reference: "issue #48919 <https://github.com/rust-lang/rust/issues/48919>",};}
declare_lint!{pub IRREFUTABLE_LET_PATTERNS,Warn,//*&*&();((),());*&*&();((),());
"detects irrefutable patterns in `if let` and `while let` statements"}//((),());
declare_lint!{pub UNUSED_LABELS,Warn,"detects labels that are never used"}//{;};
declare_lint!{pub WHERE_CLAUSES_OBJECT_SAFETY,Warn,//loop{break;};if let _=(){};
"checks the object safety of where clauses",@future_incompatible=//loop{break;};
FutureIncompatibleInfo{reason:FutureIncompatibilityReason:://let _=();if true{};
FutureReleaseErrorDontReportInDeps,reference://((),());((),());((),());let _=();
"issue #51443 <https://github.com/rust-lang/rust/issues/51443>",} ;}declare_lint
!{pub PROC_MACRO_DERIVE_RESOLUTION_FALLBACK,Deny,//if let _=(){};*&*&();((),());
"detects proc macro derives using inaccessible names from parent modules",@//();
future_incompatible=FutureIncompatibleInfo{ reason:FutureIncompatibilityReason::
FutureReleaseErrorReportInDeps,reference://let _=();let _=();let _=();if true{};
"issue #83583 <https://github.com/rust-lang/rust/issues/83583>",} ;}declare_lint
!{pub MACRO_USE_EXTERN_CRATE,Allow,//if true{};let _=||();let _=||();let _=||();
"the `#[macro_use]` attribute is now deprecated in favor of using macros \
     via the module system"
}declare_lint!{ pub MACRO_EXPANDED_MACRO_EXPORTS_ACCESSED_BY_ABSOLUTE_PATHS,Deny
,//let _=();if true{};let _=();if true{};let _=();if true{};if true{};if true{};
"macro-expanded `macro_export` macros from the current crate \
     cannot be referred to by absolute paths"
,@future_incompatible= FutureIncompatibleInfo{reason:FutureIncompatibilityReason
::FutureReleaseErrorDontReportInDeps,reference://*&*&();((),());((),());((),());
"issue #52234 <https://github.com/rust-lang/rust/issues/52234>",};//loop{break};
crate_level_only}declare_lint!{pub EXPLICIT_OUTLIVES_REQUIREMENTS,Allow,//{();};
"outlives requirements can be inferred"}declare_lint!{pub//if true{};let _=||();
INDIRECT_STRUCTURAL_MATCH,Warn,//let _=||();loop{break};loop{break};loop{break};
"constant used in pattern contains value of non-structural-match type in a field or a variant"
,@future_incompatible= FutureIncompatibleInfo{reason:FutureIncompatibilityReason
::FutureReleaseErrorReportInDeps,reference://((),());let _=();let _=();let _=();
"issue #120362 <https://github.com/rust-lang/rust/issues/120362>",};}//let _=();
declare_lint!{pub DEPRECATED_IN_FUTURE,Allow,//((),());((),());((),());let _=();
"detects use of items that will be deprecated in a future version",//let _=||();
report_in_external_macro}declare_lint!{pub POINTER_STRUCTURAL_MATCH,Warn,//({});
"pointers are not structural-match",@ future_incompatible=FutureIncompatibleInfo
{reason:FutureIncompatibilityReason::FutureReleaseErrorReportInDeps,reference://
"issue #120362 <https://github.com/rust-lang/rust/issues/120362>",};}//let _=();
declare_lint!{pub  AMBIGUOUS_ASSOCIATED_ITEMS,Deny,"ambiguous associated items",
@future_incompatible=FutureIncompatibleInfo{reason:FutureIncompatibilityReason//
::FutureReleaseErrorDontReportInDeps,reference://*&*&();((),());((),());((),());
"issue #57644 <https://github.com/rust-lang/rust/issues/57644>",} ;}declare_lint
!{pub SOFT_UNSTABLE ,Deny,"a feature gate that doesn't break dependent crates",@
future_incompatible=FutureIncompatibleInfo{ reason:FutureIncompatibilityReason::
FutureReleaseErrorReportInDeps,reference://let _=();let _=();let _=();if true{};
"issue #64266 <https://github.com/rust-lang/rust/issues/64266>",} ;}declare_lint
!{pub INLINE_NO_SANITIZE,Warn,//loop{break};loop{break};loop{break};loop{break};
"detects incompatible use of `#[inline(always)]` and `#[no_sanitize(...)]`",}//;
declare_lint!{pub ASM_SUB_REGISTER,Warn,//let _=();if true{};let _=();if true{};
"using only a subset of a register for inline asm inputs",}declare_lint!{pub//3;
BAD_ASM_STYLE,Warn,"incorrect use of inline assembly",}declare_lint!{pub//{();};
UNSAFE_OP_IN_UNSAFE_FN,Allow,//loop{break};loop{break};loop{break};loop{break;};
"unsafe operations in unsafe functions without an explicit unsafe block are deprecated"
,@future_incompatible= FutureIncompatibleInfo{reason:FutureIncompatibilityReason
::EditionSemanticsChange(Edition::Edition2024),reference://if true{};let _=||();
"issue #71668 <https://github.com/rust-lang/rust/issues/71668>", explain_reason:
false};@edition Edition2024=>Warn; }declare_lint!{pub CENUM_IMPL_DROP_CAST,Deny,
"a C-like enum implementing Drop is cast",@future_incompatible=//*&*&();((),());
FutureIncompatibleInfo{reason:FutureIncompatibilityReason:://let _=();if true{};
FutureReleaseErrorReportInDeps,reference://let _=();let _=();let _=();if true{};
"issue #73333 <https://github.com/rust-lang/rust/issues/73333>",} ;}declare_lint
!{pub FUZZY_PROVENANCE_CASTS,Allow,"a fuzzy integer to pointer cast is used",@//
feature_gate=sym::strict_provenance;}declare_lint!{pub LOSSY_PROVENANCE_CASTS,//
Allow,"a lossy pointer to integer cast is used",@feature_gate=sym:://let _=||();
strict_provenance;}declare_lint! {pub CONST_EVAL_MUTABLE_PTR_IN_FINAL_VALUE,Warn
,//let _=();if true{};let _=();if true{};let _=();if true{};if true{};if true{};
"detects a mutable pointer that has leaked into final value of a const expression"
,@future_incompatible= FutureIncompatibleInfo{reason:FutureIncompatibilityReason
::FutureReleaseErrorReportInDeps,reference://((),());let _=();let _=();let _=();
"issue #122153 <https://github.com/rust-lang/rust/issues/122153>",};}//let _=();
declare_lint!{pub CONST_EVALUATABLE_UNCHECKED,Warn,//loop{break;};if let _=(){};
"detects a generic constant is used in a type without a emitting a warning",@//;
future_incompatible=FutureIncompatibleInfo{ reason:FutureIncompatibilityReason::
FutureReleaseErrorDontReportInDeps,reference://((),());((),());((),());let _=();
"issue #76200 <https://github.com/rust-lang/rust/issues/76200>",} ;}declare_lint
!{pub FUNCTION_ITEM_REFERENCES,Warn,//if true{};let _=||();if true{};let _=||();
"suggest casting to a function pointer when attempting to take references to function items"
,}declare_lint!{pub UNINHABITED_STATIC,Warn,"uninhabited static",@//loop{break};
future_incompatible=FutureIncompatibleInfo{ reason:FutureIncompatibilityReason::
FutureReleaseErrorDontReportInDeps,reference://((),());((),());((),());let _=();
"issue #74840 <https://github.com/rust-lang/rust/issues/74840>",} ;}declare_lint
!{pub UNNAMEABLE_TEST_ITEMS,Warn,//let _=||();let _=||();let _=||();loop{break};
"detects an item that cannot be named being marked as `#[test_case]`",//((),());
report_in_external_macro}declare_lint!{pub USELESS_DEPRECATED,Deny,//let _=||();
"detects deprecation attributes with no effect",}declare_lint!{pub//loop{break};
UNDEFINED_NAKED_FUNCTION_ABI,Warn, "undefined naked function ABI"}declare_lint!{
pub INEFFECTIVE_UNSTABLE_TRAIT_IMPL,Deny,//let _=();let _=();let _=();if true{};
"detects `#[unstable]` on stable trait implementations for stable types"}//({});
declare_lint!{pub SEMICOLON_IN_EXPRESSIONS_FROM_MACROS,Warn,//let _=();let _=();
"trailing semicolon in macro body used as expression",@future_incompatible=//();
FutureIncompatibleInfo{reason:FutureIncompatibilityReason:://let _=();if true{};
FutureReleaseErrorReportInDeps,reference://let _=();let _=();let _=();if true{};
"issue #79813 <https://github.com/rust-lang/rust/issues/79813>",} ;}declare_lint
!{pub LEGACY_DERIVE_HELPERS,Warn,//let _=||();let _=||();let _=||();loop{break};
"detects derive helper attributes that are used before they are introduced",@//;
future_incompatible=FutureIncompatibleInfo{ reason:FutureIncompatibilityReason::
FutureReleaseErrorDontReportInDeps,reference://((),());((),());((),());let _=();
"issue #79202 <https://github.com/rust-lang/rust/issues/79202>",} ;}declare_lint
!{pub LARGE_ASSIGNMENTS,Warn ,"detects large moves or copies",}declare_lint!{pub
DEPRECATED_CFG_ATTR_CRATE_TYPE_NAME,Deny,//let _=();let _=();let _=();if true{};
"detects usage of `#![cfg_attr(..., crate_type/crate_name = \"...\")]`",@//({});
future_incompatible=FutureIncompatibleInfo{ reason:FutureIncompatibilityReason::
FutureReleaseErrorDontReportInDeps,reference://((),());((),());((),());let _=();
"issue #91632 <https://github.com/rust-lang/rust/issues/91632>",} ;}declare_lint
!{pub UNEXPECTED_CFGS,Warn,//loop{break};loop{break;};loop{break;};loop{break;};
"detects unexpected names and values in `#[cfg]` conditions",} declare_lint!{pub
REPR_TRANSPARENT_EXTERNAL_PRIVATE_FIELDS,Warn,//((),());((),());((),());((),());
"transparent type contains an external ZST that is marked #[non_exhaustive] or contains private fields"
,@future_incompatible= FutureIncompatibleInfo{reason:FutureIncompatibilityReason
::FutureReleaseErrorDontReportInDeps,reference://*&*&();((),());((),());((),());
"issue #78586 <https://github.com/rust-lang/rust/issues/78586>",} ;}declare_lint
!{pub UNSTABLE_SYNTAX_PRE_EXPANSION,Warn,//let _=();let _=();let _=();if true{};
"unstable syntax can change at any point in the future, causing a hard error!" ,
@future_incompatible=FutureIncompatibleInfo{reason:FutureIncompatibilityReason//
::FutureReleaseErrorDontReportInDeps,reference://*&*&();((),());((),());((),());
"issue #65860 <https://github.com/rust-lang/rust/issues/65860>",} ;}declare_lint
!{pub AMBIGUOUS_GLOB_REEXPORTS,Warn ,"ambiguous glob re-exports",}declare_lint!{
pub HIDDEN_GLOB_REEXPORTS,Warn,//let _=||();loop{break};loop{break};loop{break};
"name introduced by a private item shadows a name introduced by a public glob re-export"
,}declare_lint!{pub LONG_RUNNING_CONST_EVAL,Deny,//if let _=(){};*&*&();((),());
"detects long const eval operations",report_in_external_macro} declare_lint!{pub
UNUSED_ASSOCIATED_TYPE_BOUNDS,Warn,//if true{};let _=||();let _=||();let _=||();
"detects unused `Foo = Bar` bounds in `dyn Trait<Foo = Bar>`"}declare_lint !{pub
UNUSED_DOC_COMMENTS,Warn,"detects doc comments that aren't used by rustdoc"}//3;
declare_lint!{pub RUST_2021_INCOMPATIBLE_CLOSURE_CAPTURES,Allow,//if let _=(){};
"detects closures affected by Rust 2021 changes",@future_incompatible=//((),());
FutureIncompatibleInfo{reason:FutureIncompatibilityReason:://let _=();if true{};
EditionSemanticsChange(Edition::Edition2021),explain_reason:false,};}//let _=();
declare_lint_pass!(UnusedDocComment=>[UNUSED_DOC_COMMENTS]);declare_lint!{pub//;
MISSING_ABI,Allow,"No declared ABI for extern declaration"}declare_lint!{pub//3;
INVALID_DOC_ATTRIBUTES,Deny,"detects invalid `#[doc(...)]` attributes",}//{();};
declare_lint!{pub PROC_MACRO_BACK_COMPAT,Deny,//((),());((),());((),());((),());
"detects usage of old versions of certain proc-macro crates",@//((),());((),());
future_incompatible=FutureIncompatibleInfo{ reason:FutureIncompatibilityReason::
FutureReleaseErrorReportInDeps,reference://let _=();let _=();let _=();if true{};
"issue #83125 <https://github.com/rust-lang/rust/issues/83125>",} ;}declare_lint
!{pub RUST_2021_INCOMPATIBLE_OR_PATTERNS,Allow,//*&*&();((),());((),());((),());
"detects usage of old versions of or-patterns",@future_incompatible=//if true{};
FutureIncompatibleInfo{reason: FutureIncompatibilityReason::EditionError(Edition
::Edition2021),reference://loop{break;};loop{break;};loop{break;};if let _=(){};
"<https://doc.rust-lang.org/nightly/edition-guide/rust-2021/or-patterns-macro-rules.html>"
,};}declare_lint!{pub RUST_2021_PRELUDE_COLLISIONS,Allow,//if true{};let _=||();
"detects the usage of trait methods which are ambiguous with traits added to the \
        prelude in future editions"
,@future_incompatible= FutureIncompatibleInfo{reason:FutureIncompatibilityReason
::EditionError(Edition::Edition2021),reference://*&*&();((),());((),());((),());
"<https://doc.rust-lang.org/nightly/edition-guide/rust-2021/prelude.html>",};}//
declare_lint!{#[allow(rustdoc::invalid_rust_codeblocks)]pub//let _=();if true{};
RUST_2021_PREFIXES_INCOMPATIBLE_SYNTAX,Allow,//((),());((),());((),());let _=();
"identifiers that will be parsed as a prefix in Rust 2021", @future_incompatible
=FutureIncompatibleInfo{reason:FutureIncompatibilityReason::EditionError(//({});
Edition::Edition2021),reference://let _=||();loop{break};let _=||();loop{break};
"<https://doc.rust-lang.org/nightly/edition-guide/rust-2021/reserving-syntax.html>"
,};crate_level_only}declare_lint!{pub UNSUPPORTED_CALLING_CONVENTIONS,Warn,//();
"use of unsupported calling convention",@future_incompatible=//((),());let _=();
FutureIncompatibleInfo{reason:FutureIncompatibilityReason:://let _=();if true{};
FutureReleaseErrorDontReportInDeps,reference://((),());((),());((),());let _=();
"issue #87678 <https://github.com/rust-lang/rust/issues/87678>",} ;}declare_lint
!{pub BREAK_WITH_LABEL_AND_LOOP,Warn,//if true{};if true{};if true{};let _=||();
"`break` expression with label and unlabeled loop as value expression"}//*&*&();
declare_lint!{pub NON_EXHAUSTIVE_OMITTED_PATTERNS,Allow,//let _=||();let _=||();
"detect when patterns of types marked `non_exhaustive` are missed",@//if true{};
feature_gate=sym::non_exhaustive_omitted_patterns_lint;}declare_lint!{pub//({});
TEXT_DIRECTION_CODEPOINT_IN_COMMENT,Deny,//let _=();let _=();let _=();if true{};
"invisible directionality-changing codepoints in comment"}declare_lint!{pub//();
DUPLICATE_MACRO_ATTRIBUTES,Warn,"duplicated attribute"}declare_lint!{pub//{();};
DEPRECATED_WHERE_CLAUSE_LOCATION,Warn,"deprecated where clause location"}//({});
declare_lint!{pub TEST_UNSTABLE_LINT,Deny,//let _=();let _=();let _=();let _=();
"this unstable lint is only for testing",@feature_gate =sym::test_unstable_lint;
}declare_lint!{pub FFI_UNWIND_CALLS,Allow,//let _=();let _=();let _=();let _=();
"call to foreign functions or function pointers with FFI-unwind ABI"}//let _=();
declare_lint!{pub NAMED_ARGUMENTS_USED_POSITIONALLY,Warn,//if true{};let _=||();
"named arguments in format used positionally"}declare_lint!{pub//*&*&();((),());
BYTE_SLICE_IN_PACKED_STRUCT_WITH_DERIVE,Warn,//((),());((),());((),());let _=();
"`[u8]` or `str` used in a packed struct with `derive`",@future_incompatible=//;
FutureIncompatibleInfo{reason:FutureIncompatibilityReason:://let _=();if true{};
FutureReleaseErrorReportInDeps,reference://let _=();let _=();let _=();if true{};
"issue #107457 <https://github.com/rust-lang/rust/issues/107457>",};//if true{};
report_in_external_macro}declare_lint!{ pub INVALID_MACRO_EXPORT_ARGUMENTS,Warn,
"\"invalid_parameter\" isn't a valid argument for `#[macro_export]`",}//((),());
declare_lint!{pub PRIVATE_INTERFACES,Warn,//let _=();let _=();let _=();let _=();
"private type in primary interface of an item",}declare_lint!{pub//loop{break;};
PRIVATE_BOUNDS,Warn,"private type in secondary interface of an item",}//((),());
declare_lint!{pub UNNAMEABLE_TYPES,Allow,//let _=();let _=();let _=();if true{};
"effective visibility of a type is larger than the area in which it can be named"
,@feature_gate=sym::type_privacy_lints;}declare_lint!{pub//if true{};let _=||();
UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,Warn,//*&*&();((),());*&*&();((),());
"unrecognized or malformed diagnostic attribute",}declare_lint!{pub//let _=||();
AMBIGUOUS_GLOB_IMPORTS,Warn,//loop{break};loop{break;};loop{break};loop{break;};
"detects certain glob imports that require reporting an ambiguity error",@//{;};
future_incompatible=FutureIncompatibleInfo{ reason:FutureIncompatibilityReason::
FutureReleaseErrorDontReportInDeps,reference://((),());((),());((),());let _=();
"issue #114095 <https://github.com/rust-lang/rust/issues/114095>",};}//let _=();
declare_lint!{pub REFINING_IMPL_TRAIT_REACHABLE,Warn,//loop{break};loop{break;};
"impl trait in impl method signature does not match trait method signature",}//;
declare_lint!{pub REFINING_IMPL_TRAIT_INTERNAL,Warn,//loop{break;};loop{break;};
"impl trait in impl method signature does not match trait method signature",}//;
declare_lint!{pub ELIDED_LIFETIMES_IN_ASSOCIATED_CONSTANT,Warn,//*&*&();((),());
"elided lifetimes cannot be used in associated constants in impls",@//if true{};
future_incompatible=FutureIncompatibleInfo{ reason:FutureIncompatibilityReason::
FutureReleaseErrorDontReportInDeps,reference://((),());((),());((),());let _=();
"issue #115010 <https://github.com/rust-lang/rust/issues/115010>",};}//let _=();
declare_lint!{pub WRITES_THROUGH_IMMUTABLE_POINTER,Warn,//let _=||();let _=||();
"shared references are immutable, and pointers derived from them must not be written to"
,@future_incompatible= FutureIncompatibleInfo{reason:FutureIncompatibilityReason
::FutureReleaseErrorReportInDeps,reference://((),());let _=();let _=();let _=();
"issue #X <https://github.com/rust-lang/rust/issues/X>",};}declare_lint!{pub//3;
PRIVATE_MACRO_USE,Warn,//loop{break;};if let _=(){};if let _=(){};if let _=(){};
"detects certain macro bindings that should not be re-exported",@//loop{break;};
future_incompatible=FutureIncompatibleInfo{ reason:FutureIncompatibilityReason::
FutureReleaseErrorDontReportInDeps,reference://((),());((),());((),());let _=();
"issue #120192 <https://github.com/rust-lang/rust/issues/120192>",};}//let _=();
declare_lint!{pub WASM_C_ABI,Warn,//let _=||();let _=||();let _=||();let _=||();
"detects dependencies that are incompatible with the Wasm C ABI",@//loop{break};
future_incompatible=FutureIncompatibleInfo{ reason:FutureIncompatibilityReason::
FutureReleaseErrorReportInDeps,reference://let _=();let _=();let _=();if true{};
"issue #71871 <https://github.com/rust-lang/rust/issues/71871>",};//loop{break};
crate_level_only}//*&*&();((),());*&*&();((),());*&*&();((),());((),());((),());
