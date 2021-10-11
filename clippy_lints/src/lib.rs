// error-pattern:cargo-clippy

#![feature(box_patterns)]
#![feature(drain_filter)]
#![feature(in_band_lifetimes)]
#![feature(iter_zip)]
#![feature(once_cell)]
#![feature(rustc_private)]
#![feature(stmt_expr_attributes)]
#![feature(control_flow_enum)]
#![recursion_limit = "512"]
#![cfg_attr(feature = "deny-warnings", deny(warnings))]
#![allow(clippy::missing_docs_in_private_items, clippy::must_use_candidate)]
#![warn(trivial_casts, trivial_numeric_casts)]
// warn on lints, that are included in `rust-lang/rust`s bootstrap
#![warn(rust_2018_idioms, unused_lifetimes)]
// warn on rustc internal lints
#![warn(rustc::internal)]

// FIXME: switch to something more ergonomic here, once available.
// (Currently there is no way to opt into sysroot crates without `extern crate`.)
extern crate rustc_ast;
extern crate rustc_ast_pretty;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_hir_pretty;
extern crate rustc_index;
extern crate rustc_infer;
extern crate rustc_lexer;
extern crate rustc_lint;
extern crate rustc_middle;
extern crate rustc_mir_dataflow;
extern crate rustc_parse;
extern crate rustc_parse_format;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_target;
extern crate rustc_trait_selection;
extern crate rustc_typeck;

#[macro_use]
extern crate clippy_utils;

use clippy_utils::parse_msrv;
use rustc_data_structures::fx::FxHashSet;
use rustc_lint::LintId;
use rustc_session::Session;

/// Macro used to declare a Clippy lint.
///
/// Every lint declaration consists of 4 parts:
///
/// 1. The documentation, which is used for the website
/// 2. The `LINT_NAME`. See [lint naming][lint_naming] on lint naming conventions.
/// 3. The `lint_level`, which is a mapping from *one* of our lint groups to `Allow`, `Warn` or
///    `Deny`. The lint level here has nothing to do with what lint groups the lint is a part of.
/// 4. The `description` that contains a short explanation on what's wrong with code where the
///    lint is triggered.
///
/// Currently the categories `style`, `correctness`, `suspicious`, `complexity` and `perf` are
/// enabled by default. As said in the README.md of this repository, if the lint level mapping
/// changes, please update README.md.
///
/// # Example
///
/// ```
/// #![feature(rustc_private)]
/// extern crate rustc_session;
/// use rustc_session::declare_tool_lint;
/// use clippy_lints::declare_clippy_lint;
///
/// declare_clippy_lint! {
///     /// ### What it does
///     /// Checks for ... (describe what the lint matches).
///     ///
///     /// ### Why is this bad?
///     /// Supply the reason for linting the code.
///     ///
///     /// ### Example
///     /// ```rust
///     /// // Bad
///     /// Insert a short example of code that triggers the lint
///     ///
///     /// // Good
///     /// Insert a short example of improved code that doesn't trigger the lint
///     /// ```
///     pub LINT_NAME,
///     pedantic,
///     "description"
/// }
/// ```
/// [lint_naming]: https://rust-lang.github.io/rfcs/0344-conventions-galore.html#lints
#[macro_export]
macro_rules! declare_clippy_lint {
    { $(#[$attr:meta])* pub $name:tt, style, $description:tt } => {
        declare_tool_lint! {
            $(#[$attr])* pub clippy::$name, Warn, $description, report_in_external_macro: true
        }
    };
    { $(#[$attr:meta])* pub $name:tt, correctness, $description:tt } => {
        declare_tool_lint! {
            $(#[$attr])* pub clippy::$name, Deny, $description, report_in_external_macro: true
        }
    };
    { $(#[$attr:meta])* pub $name:tt, suspicious, $description:tt } => {
        declare_tool_lint! {
            $(#[$attr])* pub clippy::$name, Warn, $description, report_in_external_macro: true
        }
    };
    { $(#[$attr:meta])* pub $name:tt, complexity, $description:tt } => {
        declare_tool_lint! {
            $(#[$attr])* pub clippy::$name, Warn, $description, report_in_external_macro: true
        }
    };
    { $(#[$attr:meta])* pub $name:tt, perf, $description:tt } => {
        declare_tool_lint! {
            $(#[$attr])* pub clippy::$name, Warn, $description, report_in_external_macro: true
        }
    };
    { $(#[$attr:meta])* pub $name:tt, pedantic, $description:tt } => {
        declare_tool_lint! {
            $(#[$attr])* pub clippy::$name, Allow, $description, report_in_external_macro: true
        }
    };
    { $(#[$attr:meta])* pub $name:tt, restriction, $description:tt } => {
        declare_tool_lint! {
            $(#[$attr])* pub clippy::$name, Allow, $description, report_in_external_macro: true
        }
    };
    { $(#[$attr:meta])* pub $name:tt, cargo, $description:tt } => {
        declare_tool_lint! {
            $(#[$attr])* pub clippy::$name, Allow, $description, report_in_external_macro: true
        }
    };
    { $(#[$attr:meta])* pub $name:tt, nursery, $description:tt } => {
        declare_tool_lint! {
            $(#[$attr])* pub clippy::$name, Allow, $description, report_in_external_macro: true
        }
    };
    { $(#[$attr:meta])* pub $name:tt, internal, $description:tt } => {
        declare_tool_lint! {
            $(#[$attr])* pub clippy::$name, Allow, $description, report_in_external_macro: true
        }
    };
    { $(#[$attr:meta])* pub $name:tt, internal_warn, $description:tt } => {
        declare_tool_lint! {
            $(#[$attr])* pub clippy::$name, Warn, $description, report_in_external_macro: true
        }
    };
}

#[cfg(feature = "metadata-collector-lint")]
mod deprecated_lints;
mod utils;

// begin lints modules, do not remove this comment, it’s used in `update_lints`
mod absurd_extreme_comparisons;
mod approx_const;
mod arithmetic;
mod as_conversions;
mod asm_syntax;
mod assertions_on_constants;
mod assign_ops;
mod async_yields_async;
mod attrs;
mod await_holding_invalid;
mod bit_mask;
mod blacklisted_name;
mod blocks_in_if_conditions;
mod bool_assert_comparison;
mod booleans;
mod bytecount;
mod cargo_common_metadata;
mod case_sensitive_file_extension_comparisons;
mod casts;
mod checked_conversions;
mod cognitive_complexity;
mod collapsible_if;
mod collapsible_match;
mod comparison_chain;
mod copies;
mod copy_iterator;
mod create_dir;
mod dbg_macro;
mod default;
mod default_numeric_fallback;
mod dereference;
mod derivable_impls;
mod derive;
mod disallowed_method;
mod disallowed_script_idents;
mod disallowed_type;
mod doc;
mod double_comparison;
mod double_parens;
mod drop_forget_ref;
mod duration_subsec;
mod else_if_without_else;
mod empty_enum;
mod entry;
mod enum_clike;
mod enum_variants;
mod eq_op;
mod equatable_if_let;
mod erasing_op;
mod escape;
mod eta_reduction;
mod eval_order_dependence;
mod excessive_bools;
mod exhaustive_items;
mod exit;
mod explicit_write;
mod fallible_impl_from;
mod feature_name;
mod float_equality_without_abs;
mod float_literal;
mod floating_point_arithmetic;
mod format;
mod formatting;
mod from_over_into;
mod from_str_radix_10;
mod functions;
mod future_not_send;
mod get_last_with_len;
mod identity_op;
mod if_let_mutex;
mod if_not_else;
mod if_then_panic;
mod if_then_some_else_none;
mod implicit_hasher;
mod implicit_return;
mod implicit_saturating_sub;
mod inconsistent_struct_constructor;
mod indexing_slicing;
mod infinite_iter;
mod inherent_impl;
mod inherent_to_string;
mod inline_fn_without_body;
mod int_plus_one;
mod integer_division;
mod invalid_upcast_comparisons;
mod items_after_statements;
mod iter_not_returning_iterator;
mod large_const_arrays;
mod large_enum_variant;
mod large_stack_arrays;
mod len_zero;
mod let_if_seq;
mod let_underscore;
mod lifetimes;
mod literal_representation;
mod loops;
mod macro_use;
mod main_recursion;
mod manual_async_fn;
mod manual_map;
mod manual_non_exhaustive;
mod manual_ok_or;
mod manual_strip;
mod manual_unwrap_or;
mod map_clone;
mod map_err_ignore;
mod map_unit_fn;
mod match_on_vec_items;
mod match_result_ok;
mod matches;
mod mem_forget;
mod mem_replace;
mod methods;
mod minmax;
mod misc;
mod misc_early;
mod missing_const_for_fn;
mod missing_doc;
mod missing_enforced_import_rename;
mod missing_inline;
mod module_style;
mod modulo_arithmetic;
mod multiple_crate_versions;
mod mut_key;
mod mut_mut;
mod mut_mutex_lock;
mod mut_reference;
mod mutable_debug_assertion;
mod mutex_atomic;
mod needless_arbitrary_self_type;
mod needless_bitwise_bool;
mod needless_bool;
mod needless_borrow;
mod needless_borrowed_ref;
mod needless_continue;
mod needless_for_each;
mod needless_option_as_deref;
mod needless_pass_by_value;
mod needless_question_mark;
mod needless_update;
mod neg_cmp_op_on_partial_ord;
mod neg_multiply;
mod new_without_default;
mod no_effect;
mod non_copy_const;
mod non_expressive_names;
mod non_octal_unix_permissions;
mod non_send_fields_in_send_ty;
mod nonstandard_macro_braces;
mod open_options;
mod option_env_unwrap;
mod option_if_let_else;
mod overflow_check_conditional;
mod panic_in_result_fn;
mod panic_unimplemented;
mod partialeq_ne_impl;
mod pass_by_ref_or_value;
mod path_buf_push_overwrite;
mod pattern_type_mismatch;
mod precedence;
mod ptr;
mod ptr_eq;
mod ptr_offset_with_cast;
mod question_mark;
mod ranges;
mod redundant_clone;
mod redundant_closure_call;
mod redundant_else;
mod redundant_field_names;
mod redundant_pub_crate;
mod redundant_slicing;
mod redundant_static_lifetimes;
mod ref_option_ref;
mod reference;
mod regex;
mod repeat_once;
mod returns;
mod same_name_method;
mod self_assignment;
mod self_named_constructors;
mod semicolon_if_nothing_returned;
mod serde_api;
mod shadow;
mod single_component_path_imports;
mod size_of_in_element_count;
mod slow_vector_initialization;
mod stable_sort_primitive;
mod strings;
mod strlen_on_c_strings;
mod suspicious_operation_groupings;
mod suspicious_trait_impl;
mod swap;
mod tabs_in_doc_comments;
mod temporary_assignment;
mod to_digit_is_some;
mod to_string_in_display;
mod trait_bounds;
mod transmute;
mod transmuting_null;
mod try_err;
mod types;
mod undropped_manually_drops;
mod unicode;
mod unit_return_expecting_ord;
mod unit_types;
mod unnamed_address;
mod unnecessary_self_imports;
mod unnecessary_sort_by;
mod unnecessary_wraps;
mod unnested_or_patterns;
mod unsafe_removed_from_name;
mod unused_async;
mod unused_io_amount;
mod unused_self;
mod unused_unit;
mod unwrap;
mod unwrap_in_result;
mod upper_case_acronyms;
mod use_self;
mod useless_conversion;
mod vec;
mod vec_init_then_push;
mod vec_resize_to_zero;
mod verbose_file_reads;
mod wildcard_dependencies;
mod wildcard_imports;
mod write;
mod zero_div_zero;
mod zero_sized_map_values;
// end lints modules, do not remove this comment, it’s used in `update_lints`

pub use crate::utils::conf::Conf;
use crate::utils::conf::TryConf;

/// Register all pre expansion lints
///
/// Pre-expansion lints run before any macro expansion has happened.
///
/// Note that due to the architecture of the compiler, currently `cfg_attr` attributes on crate
/// level (i.e `#![cfg_attr(...)]`) will still be expanded even when using a pre-expansion pass.
///
/// Used in `./src/driver.rs`.
pub fn register_pre_expansion_lints(store: &mut rustc_lint::LintStore) {
    // NOTE: Do not add any more pre-expansion passes. These should be removed eventually.
    store.register_pre_expansion_pass(|| Box::new(write::Write::default()));
    store.register_pre_expansion_pass(|| Box::new(attrs::EarlyAttributes));
    store.register_pre_expansion_pass(|| Box::new(dbg_macro::DbgMacro));
}

#[doc(hidden)]
pub fn read_conf(sess: &Session) -> Conf {
    let file_name = match utils::conf::lookup_conf_file() {
        Ok(Some(path)) => path,
        Ok(None) => return Conf::default(),
        Err(error) => {
            sess.struct_err(&format!("error finding Clippy's configuration file: {}", error))
                .emit();
            return Conf::default();
        },
    };

    let TryConf { conf, errors } = utils::conf::read(&file_name);
    // all conf errors are non-fatal, we just use the default conf in case of error
    for error in errors {
        sess.struct_err(&format!(
            "error reading Clippy's configuration file `{}`: {}",
            file_name.display(),
            error
        ))
        .emit();
    }

    conf
}

/// Register all lints and lint groups with the rustc plugin registry
///
/// Used in `./src/driver.rs`.
#[allow(clippy::too_many_lines)]
#[rustfmt::skip]
pub fn register_plugins(store: &mut rustc_lint::LintStore, sess: &Session, conf: &Conf) {
    register_removed_non_tool_lints(store);

    include!("lib.deprecated.rs");

    include!("lib.register_lints.rs");
    include!("lib.register_restriction.rs");
    include!("lib.register_pedantic.rs");

    #[cfg(feature = "internal-lints")]
    include!("lib.register_internal.rs");

    include!("lib.register_all.rs");
    include!("lib.register_style.rs");
    include!("lib.register_complexity.rs");
    include!("lib.register_correctness.rs");
    include!("lib.register_suspicious.rs");
    include!("lib.register_perf.rs");
    include!("lib.register_cargo.rs");
    include!("lib.register_nursery.rs");

    #[cfg(feature = "metadata-collector-lint")]
    {
        if std::env::var("ENABLE_METADATA_COLLECTION").eq(&Ok("1".to_string())) {
            store.register_late_pass(|| Box::new(utils::internal_lints::metadata_collector::MetadataCollector::new()));
            return;
        }
    }

    // all the internal lints
    #[cfg(feature = "internal-lints")]
    {
        store.register_early_pass(|| Box::new(utils::internal_lints::ClippyLintsInternal));
        store.register_early_pass(|| Box::new(utils::internal_lints::ProduceIce));
        store.register_late_pass(|| Box::new(utils::inspector::DeepCodeInspector));
        store.register_late_pass(|| Box::new(utils::internal_lints::CollapsibleCalls));
        store.register_late_pass(|| Box::new(utils::internal_lints::CompilerLintFunctions::new()));
        store.register_late_pass(|| Box::new(utils::internal_lints::IfChainStyle));
        store.register_late_pass(|| Box::new(utils::internal_lints::InvalidPaths));
        store.register_late_pass(|| Box::new(utils::internal_lints::InterningDefinedSymbol::default()));
        store.register_late_pass(|| Box::new(utils::internal_lints::LintWithoutLintPass::default()));
        store.register_late_pass(|| Box::new(utils::internal_lints::MatchTypeOnDiagItem));
        store.register_late_pass(|| Box::new(utils::internal_lints::OuterExpnDataPass));
    }

    store.register_late_pass(|| Box::new(utils::author::Author));
    store.register_late_pass(|| Box::new(await_holding_invalid::AwaitHolding));
    store.register_late_pass(|| Box::new(serde_api::SerdeApi));
    let vec_box_size_threshold = conf.vec_box_size_threshold;
    let type_complexity_threshold = conf.type_complexity_threshold;
    let avoid_breaking_exported_api = conf.avoid_breaking_exported_api;
    store.register_late_pass(move || Box::new(types::Types::new(
        vec_box_size_threshold,
        type_complexity_threshold,
        avoid_breaking_exported_api,
    )));
    store.register_late_pass(|| Box::new(booleans::NonminimalBool));
    store.register_late_pass(|| Box::new(needless_bitwise_bool::NeedlessBitwiseBool));
    store.register_late_pass(|| Box::new(eq_op::EqOp));
    store.register_late_pass(|| Box::new(enum_clike::UnportableVariant));
    store.register_late_pass(|| Box::new(float_literal::FloatLiteral));
    let verbose_bit_mask_threshold = conf.verbose_bit_mask_threshold;
    store.register_late_pass(move || Box::new(bit_mask::BitMask::new(verbose_bit_mask_threshold)));
    store.register_late_pass(|| Box::new(ptr::Ptr));
    store.register_late_pass(|| Box::new(ptr_eq::PtrEq));
    store.register_late_pass(|| Box::new(needless_bool::NeedlessBool));
    store.register_late_pass(|| Box::new(needless_option_as_deref::OptionNeedlessDeref));
    store.register_late_pass(|| Box::new(needless_bool::BoolComparison));
    store.register_late_pass(|| Box::new(needless_for_each::NeedlessForEach));
    store.register_late_pass(|| Box::new(misc::MiscLints));
    store.register_late_pass(|| Box::new(eta_reduction::EtaReduction));
    store.register_late_pass(|| Box::new(identity_op::IdentityOp));
    store.register_late_pass(|| Box::new(erasing_op::ErasingOp));
    store.register_late_pass(|| Box::new(mut_mut::MutMut));
    store.register_late_pass(|| Box::new(mut_reference::UnnecessaryMutPassed));
    store.register_late_pass(|| Box::new(len_zero::LenZero));
    store.register_late_pass(|| Box::new(attrs::Attributes));
    store.register_late_pass(|| Box::new(blocks_in_if_conditions::BlocksInIfConditions));
    store.register_late_pass(|| Box::new(collapsible_match::CollapsibleMatch));
    store.register_late_pass(|| Box::new(unicode::Unicode));
    store.register_late_pass(|| Box::new(unit_return_expecting_ord::UnitReturnExpectingOrd));
    store.register_late_pass(|| Box::new(strings::StringAdd));
    store.register_late_pass(|| Box::new(implicit_return::ImplicitReturn));
    store.register_late_pass(|| Box::new(implicit_saturating_sub::ImplicitSaturatingSub));
    store.register_late_pass(|| Box::new(default_numeric_fallback::DefaultNumericFallback));
    store.register_late_pass(|| Box::new(inconsistent_struct_constructor::InconsistentStructConstructor));
    store.register_late_pass(|| Box::new(non_octal_unix_permissions::NonOctalUnixPermissions));
    store.register_early_pass(|| Box::new(unnecessary_self_imports::UnnecessarySelfImports));

    let msrv = conf.msrv.as_ref().and_then(|s| {
        parse_msrv(s, None, None).or_else(|| {
            sess.err(&format!("error reading Clippy's configuration file. `{}` is not a valid Rust version", s));
            None
        })
    });

    let avoid_breaking_exported_api = conf.avoid_breaking_exported_api;
    store.register_late_pass(move || Box::new(approx_const::ApproxConstant::new(msrv)));
    store.register_late_pass(move || Box::new(methods::Methods::new(avoid_breaking_exported_api, msrv)));
    store.register_late_pass(move || Box::new(matches::Matches::new(msrv)));
    store.register_early_pass(move || Box::new(manual_non_exhaustive::ManualNonExhaustive::new(msrv)));
    store.register_late_pass(move || Box::new(manual_strip::ManualStrip::new(msrv)));
    store.register_early_pass(move || Box::new(redundant_static_lifetimes::RedundantStaticLifetimes::new(msrv)));
    store.register_early_pass(move || Box::new(redundant_field_names::RedundantFieldNames::new(msrv)));
    store.register_late_pass(move || Box::new(checked_conversions::CheckedConversions::new(msrv)));
    store.register_late_pass(move || Box::new(mem_replace::MemReplace::new(msrv)));
    store.register_late_pass(move || Box::new(ranges::Ranges::new(msrv)));
    store.register_late_pass(move || Box::new(from_over_into::FromOverInto::new(msrv)));
    store.register_late_pass(move || Box::new(use_self::UseSelf::new(msrv)));
    store.register_late_pass(move || Box::new(missing_const_for_fn::MissingConstForFn::new(msrv)));
    store.register_late_pass(move || Box::new(needless_question_mark::NeedlessQuestionMark));
    store.register_late_pass(move || Box::new(casts::Casts::new(msrv)));
    store.register_early_pass(move || Box::new(unnested_or_patterns::UnnestedOrPatterns::new(msrv)));

    store.register_late_pass(|| Box::new(size_of_in_element_count::SizeOfInElementCount));
    store.register_late_pass(|| Box::new(same_name_method::SameNameMethod));
    store.register_late_pass(|| Box::new(map_clone::MapClone));
    store.register_late_pass(|| Box::new(map_err_ignore::MapErrIgnore));
    store.register_late_pass(|| Box::new(shadow::Shadow::default()));
    store.register_late_pass(|| Box::new(unit_types::UnitTypes));
    store.register_late_pass(|| Box::new(loops::Loops));
    store.register_late_pass(|| Box::new(main_recursion::MainRecursion::default()));
    store.register_late_pass(|| Box::new(lifetimes::Lifetimes));
    store.register_late_pass(|| Box::new(entry::HashMapPass));
    store.register_late_pass(|| Box::new(minmax::MinMaxPass));
    store.register_late_pass(|| Box::new(open_options::OpenOptions));
    store.register_late_pass(|| Box::new(zero_div_zero::ZeroDiv));
    store.register_late_pass(|| Box::new(mutex_atomic::Mutex));
    store.register_late_pass(|| Box::new(needless_update::NeedlessUpdate));
    store.register_late_pass(|| Box::new(needless_borrow::NeedlessBorrow::default()));
    store.register_late_pass(|| Box::new(needless_borrowed_ref::NeedlessBorrowedRef));
    store.register_late_pass(|| Box::new(no_effect::NoEffect));
    store.register_late_pass(|| Box::new(temporary_assignment::TemporaryAssignment));
    store.register_late_pass(|| Box::new(transmute::Transmute));
    let cognitive_complexity_threshold = conf.cognitive_complexity_threshold;
    store.register_late_pass(move || Box::new(cognitive_complexity::CognitiveComplexity::new(cognitive_complexity_threshold)));
    let too_large_for_stack = conf.too_large_for_stack;
    store.register_late_pass(move || Box::new(escape::BoxedLocal{too_large_for_stack}));
    store.register_late_pass(move || Box::new(vec::UselessVec{too_large_for_stack}));
    store.register_late_pass(|| Box::new(panic_unimplemented::PanicUnimplemented));
    store.register_late_pass(|| Box::new(strings::StringLitAsBytes));
    store.register_late_pass(|| Box::new(derive::Derive));
    store.register_late_pass(|| Box::new(derivable_impls::DerivableImpls));
    store.register_late_pass(|| Box::new(get_last_with_len::GetLastWithLen));
    store.register_late_pass(|| Box::new(drop_forget_ref::DropForgetRef));
    store.register_late_pass(|| Box::new(empty_enum::EmptyEnum));
    store.register_late_pass(|| Box::new(absurd_extreme_comparisons::AbsurdExtremeComparisons));
    store.register_late_pass(|| Box::new(invalid_upcast_comparisons::InvalidUpcastComparisons));
    store.register_late_pass(|| Box::new(regex::Regex));
    store.register_late_pass(|| Box::new(copies::CopyAndPaste));
    store.register_late_pass(|| Box::new(copy_iterator::CopyIterator));
    store.register_late_pass(|| Box::new(format::UselessFormat));
    store.register_late_pass(|| Box::new(swap::Swap));
    store.register_late_pass(|| Box::new(overflow_check_conditional::OverflowCheckConditional));
    store.register_late_pass(|| Box::new(new_without_default::NewWithoutDefault::default()));
    let blacklisted_names = conf.blacklisted_names.iter().cloned().collect::<FxHashSet<_>>();
    store.register_late_pass(move || Box::new(blacklisted_name::BlacklistedName::new(blacklisted_names.clone())));
    let too_many_arguments_threshold = conf.too_many_arguments_threshold;
    let too_many_lines_threshold = conf.too_many_lines_threshold;
    store.register_late_pass(move || Box::new(functions::Functions::new(too_many_arguments_threshold, too_many_lines_threshold)));
    let doc_valid_idents = conf.doc_valid_idents.iter().cloned().collect::<FxHashSet<_>>();
    store.register_late_pass(move || Box::new(doc::DocMarkdown::new(doc_valid_idents.clone())));
    store.register_late_pass(|| Box::new(neg_multiply::NegMultiply));
    store.register_late_pass(|| Box::new(mem_forget::MemForget));
    store.register_late_pass(|| Box::new(arithmetic::Arithmetic::default()));
    store.register_late_pass(|| Box::new(assign_ops::AssignOps));
    store.register_late_pass(|| Box::new(let_if_seq::LetIfSeq));
    store.register_late_pass(|| Box::new(eval_order_dependence::EvalOrderDependence));
    store.register_late_pass(|| Box::new(missing_doc::MissingDoc::new()));
    store.register_late_pass(|| Box::new(missing_inline::MissingInline));
    store.register_late_pass(move || Box::new(exhaustive_items::ExhaustiveItems));
    store.register_late_pass(|| Box::new(match_result_ok::MatchResultOk));
    store.register_late_pass(|| Box::new(partialeq_ne_impl::PartialEqNeImpl));
    store.register_late_pass(|| Box::new(unused_io_amount::UnusedIoAmount));
    let enum_variant_size_threshold = conf.enum_variant_size_threshold;
    store.register_late_pass(move || Box::new(large_enum_variant::LargeEnumVariant::new(enum_variant_size_threshold)));
    store.register_late_pass(|| Box::new(explicit_write::ExplicitWrite));
    store.register_late_pass(|| Box::new(needless_pass_by_value::NeedlessPassByValue));
    let pass_by_ref_or_value = pass_by_ref_or_value::PassByRefOrValue::new(
        conf.trivial_copy_size_limit,
        conf.pass_by_value_size_limit,
        conf.avoid_breaking_exported_api,
        &sess.target,
    );
    store.register_late_pass(move || Box::new(pass_by_ref_or_value));
    store.register_late_pass(|| Box::new(ref_option_ref::RefOptionRef));
    store.register_late_pass(|| Box::new(try_err::TryErr));
    store.register_late_pass(|| Box::new(bytecount::ByteCount));
    store.register_late_pass(|| Box::new(infinite_iter::InfiniteIter));
    store.register_late_pass(|| Box::new(inline_fn_without_body::InlineFnWithoutBody));
    store.register_late_pass(|| Box::new(useless_conversion::UselessConversion::default()));
    store.register_late_pass(|| Box::new(implicit_hasher::ImplicitHasher));
    store.register_late_pass(|| Box::new(fallible_impl_from::FallibleImplFrom));
    store.register_late_pass(|| Box::new(double_comparison::DoubleComparisons));
    store.register_late_pass(|| Box::new(question_mark::QuestionMark));
    store.register_early_pass(|| Box::new(suspicious_operation_groupings::SuspiciousOperationGroupings));
    store.register_late_pass(|| Box::new(suspicious_trait_impl::SuspiciousImpl));
    store.register_late_pass(|| Box::new(map_unit_fn::MapUnit));
    store.register_late_pass(|| Box::new(inherent_impl::MultipleInherentImpl));
    store.register_late_pass(|| Box::new(neg_cmp_op_on_partial_ord::NoNegCompOpForPartialOrd));
    store.register_late_pass(|| Box::new(unwrap::Unwrap));
    store.register_late_pass(|| Box::new(duration_subsec::DurationSubsec));
    store.register_late_pass(|| Box::new(indexing_slicing::IndexingSlicing));
    store.register_late_pass(|| Box::new(non_copy_const::NonCopyConst));
    store.register_late_pass(|| Box::new(ptr_offset_with_cast::PtrOffsetWithCast));
    store.register_late_pass(|| Box::new(redundant_clone::RedundantClone));
    store.register_late_pass(|| Box::new(slow_vector_initialization::SlowVectorInit));
    store.register_late_pass(|| Box::new(unnecessary_sort_by::UnnecessarySortBy));
    store.register_late_pass(move || Box::new(unnecessary_wraps::UnnecessaryWraps::new(avoid_breaking_exported_api)));
    store.register_late_pass(|| Box::new(assertions_on_constants::AssertionsOnConstants));
    store.register_late_pass(|| Box::new(transmuting_null::TransmutingNull));
    store.register_late_pass(|| Box::new(path_buf_push_overwrite::PathBufPushOverwrite));
    store.register_late_pass(|| Box::new(integer_division::IntegerDivision));
    store.register_late_pass(|| Box::new(inherent_to_string::InherentToString));
    let max_trait_bounds = conf.max_trait_bounds;
    store.register_late_pass(move || Box::new(trait_bounds::TraitBounds::new(max_trait_bounds)));
    store.register_late_pass(|| Box::new(comparison_chain::ComparisonChain));
    store.register_late_pass(|| Box::new(mut_key::MutableKeyType));
    store.register_late_pass(|| Box::new(modulo_arithmetic::ModuloArithmetic));
    store.register_early_pass(|| Box::new(reference::DerefAddrOf));
    store.register_early_pass(|| Box::new(reference::RefInDeref));
    store.register_early_pass(|| Box::new(double_parens::DoubleParens));
    store.register_late_pass(|| Box::new(to_string_in_display::ToStringInDisplay::new()));
    store.register_early_pass(|| Box::new(unsafe_removed_from_name::UnsafeNameRemoval));
    store.register_early_pass(|| Box::new(if_not_else::IfNotElse));
    store.register_early_pass(|| Box::new(else_if_without_else::ElseIfWithoutElse));
    store.register_early_pass(|| Box::new(int_plus_one::IntPlusOne));
    store.register_early_pass(|| Box::new(formatting::Formatting));
    store.register_early_pass(|| Box::new(misc_early::MiscEarlyLints));
    store.register_early_pass(|| Box::new(redundant_closure_call::RedundantClosureCall));
    store.register_late_pass(|| Box::new(redundant_closure_call::RedundantClosureCall));
    store.register_early_pass(|| Box::new(unused_unit::UnusedUnit));
    store.register_late_pass(|| Box::new(returns::Return));
    store.register_early_pass(|| Box::new(collapsible_if::CollapsibleIf));
    store.register_early_pass(|| Box::new(items_after_statements::ItemsAfterStatements));
    store.register_early_pass(|| Box::new(precedence::Precedence));
    store.register_early_pass(|| Box::new(needless_continue::NeedlessContinue));
    store.register_early_pass(|| Box::new(redundant_else::RedundantElse));
    store.register_late_pass(|| Box::new(create_dir::CreateDir));
    store.register_early_pass(|| Box::new(needless_arbitrary_self_type::NeedlessArbitrarySelfType));
    let cargo_ignore_publish = conf.cargo_ignore_publish;
    store.register_late_pass(move || Box::new(cargo_common_metadata::CargoCommonMetadata::new(cargo_ignore_publish)));
    store.register_late_pass(|| Box::new(multiple_crate_versions::MultipleCrateVersions));
    store.register_late_pass(|| Box::new(wildcard_dependencies::WildcardDependencies));
    let literal_representation_lint_fraction_readability = conf.unreadable_literal_lint_fractions;
    store.register_early_pass(move || Box::new(literal_representation::LiteralDigitGrouping::new(literal_representation_lint_fraction_readability)));
    let literal_representation_threshold = conf.literal_representation_threshold;
    store.register_early_pass(move || Box::new(literal_representation::DecimalLiteralRepresentation::new(literal_representation_threshold)));
    let enum_variant_name_threshold = conf.enum_variant_name_threshold;
    store.register_late_pass(move || Box::new(enum_variants::EnumVariantNames::new(enum_variant_name_threshold, avoid_breaking_exported_api)));
    store.register_early_pass(|| Box::new(tabs_in_doc_comments::TabsInDocComments));
    let upper_case_acronyms_aggressive = conf.upper_case_acronyms_aggressive;
    store.register_late_pass(move || Box::new(upper_case_acronyms::UpperCaseAcronyms::new(avoid_breaking_exported_api, upper_case_acronyms_aggressive)));
    store.register_late_pass(|| Box::new(default::Default::default()));
    store.register_late_pass(|| Box::new(unused_self::UnusedSelf));
    store.register_late_pass(|| Box::new(mutable_debug_assertion::DebugAssertWithMutCall));
    store.register_late_pass(|| Box::new(exit::Exit));
    store.register_late_pass(|| Box::new(to_digit_is_some::ToDigitIsSome));
    let array_size_threshold = conf.array_size_threshold;
    store.register_late_pass(move || Box::new(large_stack_arrays::LargeStackArrays::new(array_size_threshold)));
    store.register_late_pass(move || Box::new(large_const_arrays::LargeConstArrays::new(array_size_threshold)));
    store.register_late_pass(|| Box::new(floating_point_arithmetic::FloatingPointArithmetic));
    store.register_early_pass(|| Box::new(as_conversions::AsConversions));
    store.register_late_pass(|| Box::new(let_underscore::LetUnderscore));
    store.register_early_pass(|| Box::new(single_component_path_imports::SingleComponentPathImports));
    let max_fn_params_bools = conf.max_fn_params_bools;
    let max_struct_bools = conf.max_struct_bools;
    store.register_early_pass(move || Box::new(excessive_bools::ExcessiveBools::new(max_struct_bools, max_fn_params_bools)));
    store.register_early_pass(|| Box::new(option_env_unwrap::OptionEnvUnwrap));
    let warn_on_all_wildcard_imports = conf.warn_on_all_wildcard_imports;
    store.register_late_pass(move || Box::new(wildcard_imports::WildcardImports::new(warn_on_all_wildcard_imports)));
    store.register_late_pass(|| Box::new(verbose_file_reads::VerboseFileReads));
    store.register_late_pass(|| Box::new(redundant_pub_crate::RedundantPubCrate::default()));
    store.register_late_pass(|| Box::new(unnamed_address::UnnamedAddress));
    store.register_late_pass(|| Box::new(dereference::Dereferencing::default()));
    store.register_late_pass(|| Box::new(option_if_let_else::OptionIfLetElse));
    store.register_late_pass(|| Box::new(future_not_send::FutureNotSend));
    store.register_late_pass(|| Box::new(if_let_mutex::IfLetMutex));
    store.register_late_pass(|| Box::new(equatable_if_let::PatternEquality));
    store.register_late_pass(|| Box::new(mut_mutex_lock::MutMutexLock));
    store.register_late_pass(|| Box::new(match_on_vec_items::MatchOnVecItems));
    store.register_late_pass(|| Box::new(manual_async_fn::ManualAsyncFn));
    store.register_late_pass(|| Box::new(vec_resize_to_zero::VecResizeToZero));
    store.register_late_pass(|| Box::new(panic_in_result_fn::PanicInResultFn));
    let single_char_binding_names_threshold = conf.single_char_binding_names_threshold;
    store.register_early_pass(move || Box::new(non_expressive_names::NonExpressiveNames {
        single_char_binding_names_threshold,
    }));
    let macro_matcher = conf.standard_macro_braces.iter().cloned().collect::<FxHashSet<_>>();
    store.register_early_pass(move || Box::new(nonstandard_macro_braces::MacroBraces::new(&macro_matcher)));
    store.register_late_pass(|| Box::new(macro_use::MacroUseImports::default()));
    store.register_late_pass(|| Box::new(pattern_type_mismatch::PatternTypeMismatch));
    store.register_late_pass(|| Box::new(stable_sort_primitive::StableSortPrimitive));
    store.register_late_pass(|| Box::new(repeat_once::RepeatOnce));
    store.register_late_pass(|| Box::new(unwrap_in_result::UnwrapInResult));
    store.register_late_pass(|| Box::new(self_assignment::SelfAssignment));
    store.register_late_pass(|| Box::new(manual_unwrap_or::ManualUnwrapOr));
    store.register_late_pass(|| Box::new(manual_ok_or::ManualOkOr));
    store.register_late_pass(|| Box::new(float_equality_without_abs::FloatEqualityWithoutAbs));
    store.register_late_pass(|| Box::new(semicolon_if_nothing_returned::SemicolonIfNothingReturned));
    store.register_late_pass(|| Box::new(async_yields_async::AsyncYieldsAsync));
    let disallowed_methods = conf.disallowed_methods.clone();
    store.register_late_pass(move || Box::new(disallowed_method::DisallowedMethod::new(disallowed_methods.clone())));
    store.register_early_pass(|| Box::new(asm_syntax::InlineAsmX86AttSyntax));
    store.register_early_pass(|| Box::new(asm_syntax::InlineAsmX86IntelSyntax));
    store.register_late_pass(|| Box::new(undropped_manually_drops::UndroppedManuallyDrops));
    store.register_late_pass(|| Box::new(strings::StrToString));
    store.register_late_pass(|| Box::new(strings::StringToString));
    store.register_late_pass(|| Box::new(zero_sized_map_values::ZeroSizedMapValues));
    store.register_late_pass(|| Box::new(vec_init_then_push::VecInitThenPush::default()));
    store.register_late_pass(|| Box::new(case_sensitive_file_extension_comparisons::CaseSensitiveFileExtensionComparisons));
    store.register_late_pass(|| Box::new(redundant_slicing::RedundantSlicing));
    store.register_late_pass(|| Box::new(from_str_radix_10::FromStrRadix10));
    store.register_late_pass(|| Box::new(manual_map::ManualMap));
    store.register_late_pass(move || Box::new(if_then_some_else_none::IfThenSomeElseNone::new(msrv)));
    store.register_late_pass(|| Box::new(bool_assert_comparison::BoolAssertComparison));
    store.register_early_pass(move || Box::new(module_style::ModStyle));
    store.register_late_pass(|| Box::new(unused_async::UnusedAsync));
    let disallowed_types = conf.disallowed_types.iter().cloned().collect::<FxHashSet<_>>();
    store.register_late_pass(move || Box::new(disallowed_type::DisallowedType::new(&disallowed_types)));
    let import_renames = conf.enforced_import_renames.clone();
    store.register_late_pass(move || Box::new(missing_enforced_import_rename::ImportRename::new(import_renames.clone())));
    let scripts = conf.allowed_scripts.clone();
    store.register_early_pass(move || Box::new(disallowed_script_idents::DisallowedScriptIdents::new(&scripts)));
    store.register_late_pass(|| Box::new(strlen_on_c_strings::StrlenOnCStrings));
    store.register_late_pass(move || Box::new(self_named_constructors::SelfNamedConstructors));
    store.register_late_pass(move || Box::new(feature_name::FeatureName));
    store.register_late_pass(move || Box::new(iter_not_returning_iterator::IterNotReturningIterator));
    store.register_late_pass(move || Box::new(if_then_panic::IfThenPanic));
    let enable_raw_pointer_heuristic_for_send = conf.enable_raw_pointer_heuristic_for_send;
    store.register_late_pass(move || Box::new(non_send_fields_in_send_ty::NonSendFieldInSendTy::new(enable_raw_pointer_heuristic_for_send)));
}

#[rustfmt::skip]
fn register_removed_non_tool_lints(store: &mut rustc_lint::LintStore) {
    store.register_removed(
        "should_assert_eq",
        "`assert!()` will be more flexible with RFC 2011",
    );
    store.register_removed(
        "extend_from_slice",
        "`.extend_from_slice(_)` is a faster way to extend a Vec by a slice",
    );
    store.register_removed(
        "range_step_by_zero",
        "`iterator.step_by(0)` panics nowadays",
    );
    store.register_removed(
        "unstable_as_slice",
        "`Vec::as_slice` has been stabilized in 1.7",
    );
    store.register_removed(
        "unstable_as_mut_slice",
        "`Vec::as_mut_slice` has been stabilized in 1.7",
    );
    store.register_removed(
        "misaligned_transmute",
        "this lint has been split into cast_ptr_alignment and transmute_ptr_to_ptr",
    );
    store.register_removed(
        "assign_ops",
        "using compound assignment operators (e.g., `+=`) is harmless",
    );
    store.register_removed(
        "if_let_redundant_pattern_matching",
        "this lint has been changed to redundant_pattern_matching",
    );
    store.register_removed(
        "unsafe_vector_initialization",
        "the replacement suggested by this lint had substantially different behavior",
    );
    store.register_removed(
        "reverse_range_loop",
        "this lint is now included in reversed_empty_ranges",
    );
}

/// Register renamed lints.
///
/// Used in `./src/driver.rs`.
pub fn register_renamed(ls: &mut rustc_lint::LintStore) {
    ls.register_renamed("clippy::stutter", "clippy::module_name_repetitions");
    ls.register_renamed("clippy::new_without_default_derive", "clippy::new_without_default");
    ls.register_renamed("clippy::cyclomatic_complexity", "clippy::cognitive_complexity");
    ls.register_renamed("clippy::const_static_lifetime", "clippy::redundant_static_lifetimes");
    ls.register_renamed("clippy::option_and_then_some", "clippy::bind_instead_of_map");
    ls.register_renamed("clippy::box_vec", "clippy::box_collection");
    ls.register_renamed("clippy::block_in_if_condition_expr", "clippy::blocks_in_if_conditions");
    ls.register_renamed("clippy::block_in_if_condition_stmt", "clippy::blocks_in_if_conditions");
    ls.register_renamed("clippy::option_map_unwrap_or", "clippy::map_unwrap_or");
    ls.register_renamed("clippy::option_map_unwrap_or_else", "clippy::map_unwrap_or");
    ls.register_renamed("clippy::result_map_unwrap_or_else", "clippy::map_unwrap_or");
    ls.register_renamed("clippy::option_unwrap_used", "clippy::unwrap_used");
    ls.register_renamed("clippy::result_unwrap_used", "clippy::unwrap_used");
    ls.register_renamed("clippy::option_expect_used", "clippy::expect_used");
    ls.register_renamed("clippy::result_expect_used", "clippy::expect_used");
    ls.register_renamed("clippy::for_loop_over_option", "clippy::for_loops_over_fallibles");
    ls.register_renamed("clippy::for_loop_over_result", "clippy::for_loops_over_fallibles");
    ls.register_renamed("clippy::identity_conversion", "clippy::useless_conversion");
    ls.register_renamed("clippy::zero_width_space", "clippy::invisible_characters");
    ls.register_renamed("clippy::single_char_push_str", "clippy::single_char_add_str");
    ls.register_renamed("clippy::if_let_some_result", "clippy::match_result_ok");

    // uplifted lints
    ls.register_renamed("clippy::invalid_ref", "invalid_value");
    ls.register_renamed("clippy::into_iter_on_array", "array_into_iter");
    ls.register_renamed("clippy::unused_label", "unused_labels");
    ls.register_renamed("clippy::drop_bounds", "drop_bounds");
    ls.register_renamed("clippy::temporary_cstring_as_ptr", "temporary_cstring_as_ptr");
    ls.register_renamed("clippy::panic_params", "non_fmt_panics");
    ls.register_renamed("clippy::unknown_clippy_lints", "unknown_lints");
    ls.register_renamed("clippy::invalid_atomic_ordering", "invalid_atomic_ordering");
    ls.register_renamed("clippy::mem_discriminant_non_enum", "enum_intrinsics_non_enums");
}

// only exists to let the dogfood integration test works.
// Don't run clippy as an executable directly
#[allow(dead_code)]
fn main() {
    panic!("Please use the cargo-clippy executable");
}
