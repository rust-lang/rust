#![feature(array_windows)]
#![feature(binary_heap_into_iter_sorted)]
#![feature(box_patterns)]
#![feature(drain_filter)]
#![feature(if_let_guard)]
#![feature(iter_intersperse)]
#![feature(let_chains)]
#![feature(lint_reasons)]
#![feature(never_type)]
#![feature(once_cell)]
#![feature(rustc_private)]
#![feature(stmt_expr_attributes)]
#![recursion_limit = "512"]
#![cfg_attr(feature = "deny-warnings", deny(warnings))]
#![allow(clippy::missing_docs_in_private_items, clippy::must_use_candidate)]
#![warn(trivial_casts, trivial_numeric_casts)]
// warn on lints, that are included in `rust-lang/rust`s bootstrap
#![warn(rust_2018_idioms, unused_lifetimes)]
// warn on rustc internal lints
#![warn(rustc::internal)]
// Disable this rustc lint for now, as it was also done in rustc
#![allow(rustc::potential_query_instability)]

// FIXME: switch to something more ergonomic here, once available.
// (Currently there is no way to opt into sysroot crates without `extern crate`.)
extern crate rustc_arena;
extern crate rustc_ast;
extern crate rustc_ast_pretty;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_hir_analysis;
extern crate rustc_hir_pretty;
extern crate rustc_hir_typeck;
extern crate rustc_index;
extern crate rustc_infer;
extern crate rustc_lexer;
extern crate rustc_lint;
extern crate rustc_middle;
extern crate rustc_parse;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_target;
extern crate rustc_trait_selection;

#[macro_use]
extern crate clippy_utils;
#[macro_use]
extern crate declare_clippy_lint;

use std::io;
use std::path::PathBuf;

use clippy_utils::msrvs::Msrv;
use rustc_data_structures::fx::FxHashSet;
use rustc_lint::{Lint, LintId};
use rustc_session::Session;

#[cfg(feature = "internal")]
pub mod deprecated_lints;
#[cfg_attr(feature = "internal", allow(clippy::missing_clippy_version_attribute))]
mod utils;

mod declared_lints;
mod renamed_lints;

// begin lints modules, do not remove this comment, it’s used in `update_lints`
mod almost_complete_range;
mod approx_const;
mod as_conversions;
mod asm_syntax;
mod assertions_on_constants;
mod assertions_on_result_states;
mod async_yields_async;
mod attrs;
mod await_holding_invalid;
mod blocks_in_if_conditions;
mod bool_assert_comparison;
mod bool_to_int_with_if;
mod booleans;
mod borrow_deref_ref;
mod box_default;
mod cargo;
mod casts;
mod checked_conversions;
mod cognitive_complexity;
mod collapsible_if;
mod comparison_chain;
mod copies;
mod copy_iterator;
mod crate_in_macro_def;
mod create_dir;
mod dbg_macro;
mod default;
mod default_instead_of_iter_empty;
mod default_numeric_fallback;
mod default_union_representation;
mod dereference;
mod derivable_impls;
mod derive;
mod disallowed_macros;
mod disallowed_methods;
mod disallowed_names;
mod disallowed_script_idents;
mod disallowed_types;
mod doc;
mod double_parens;
mod drop_forget_ref;
mod duplicate_mod;
mod else_if_without_else;
mod empty_drop;
mod empty_enum;
mod empty_structs_with_brackets;
mod entry;
mod enum_clike;
mod enum_variants;
mod equatable_if_let;
mod escape;
mod eta_reduction;
mod excessive_bools;
mod exhaustive_items;
mod exit;
mod explicit_write;
mod extra_unused_type_parameters;
mod fallible_impl_from;
mod float_literal;
mod floating_point_arithmetic;
mod fn_null_check;
mod format;
mod format_args;
mod format_impl;
mod format_push_string;
mod formatting;
mod from_over_into;
mod from_raw_with_void_ptr;
mod from_str_radix_10;
mod functions;
mod future_not_send;
mod if_let_mutex;
mod if_not_else;
mod if_then_some_else_none;
mod implicit_hasher;
mod implicit_return;
mod implicit_saturating_add;
mod implicit_saturating_sub;
mod inconsistent_struct_constructor;
mod index_refutable_slice;
mod indexing_slicing;
mod infinite_iter;
mod inherent_impl;
mod inherent_to_string;
mod init_numbered_fields;
mod inline_fn_without_body;
mod instant_subtraction;
mod int_plus_one;
mod invalid_upcast_comparisons;
mod invalid_utf8_in_unchecked;
mod items_after_statements;
mod iter_not_returning_iterator;
mod large_const_arrays;
mod large_enum_variant;
mod large_include_file;
mod large_stack_arrays;
mod len_zero;
mod let_if_seq;
mod let_underscore;
mod lifetimes;
mod literal_representation;
mod loops;
mod macro_use;
mod main_recursion;
mod manual_assert;
mod manual_async_fn;
mod manual_bits;
mod manual_clamp;
mod manual_is_ascii_check;
mod manual_let_else;
mod manual_non_exhaustive;
mod manual_rem_euclid;
mod manual_retain;
mod manual_string_new;
mod manual_strip;
mod map_unit_fn;
mod match_result_ok;
mod matches;
mod mem_forget;
mod mem_replace;
mod methods;
mod minmax;
mod misc;
mod misc_early;
mod mismatching_type_param_order;
mod missing_const_for_fn;
mod missing_doc;
mod missing_enforced_import_rename;
mod missing_inline;
mod missing_trait_methods;
mod mixed_read_write_in_expression;
mod module_style;
mod multi_assignments;
mod multiple_unsafe_ops_per_block;
mod mut_key;
mod mut_mut;
mod mut_reference;
mod mutable_debug_assertion;
mod mutex_atomic;
mod needless_arbitrary_self_type;
mod needless_bool;
mod needless_borrowed_ref;
mod needless_continue;
mod needless_for_each;
mod needless_late_init;
mod needless_parens_on_range_literals;
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
mod octal_escapes;
mod only_used_in_recursion;
mod operators;
mod option_env_unwrap;
mod option_if_let_else;
mod overflow_check_conditional;
mod panic_in_result_fn;
mod panic_unimplemented;
mod partial_pub_fields;
mod partialeq_ne_impl;
mod partialeq_to_none;
mod pass_by_ref_or_value;
mod pattern_type_mismatch;
mod permissions_set_readonly_false;
mod precedence;
mod ptr;
mod ptr_offset_with_cast;
mod pub_use;
mod question_mark;
mod question_mark_used;
mod ranges;
mod rc_clone_in_vec_init;
mod read_zero_byte_vec;
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
mod return_self_not_must_use;
mod returns;
mod same_name_method;
mod self_named_constructors;
mod semicolon_block;
mod semicolon_if_nothing_returned;
mod serde_api;
mod shadow;
mod significant_drop_tightening;
mod single_char_lifetime_names;
mod single_component_path_imports;
mod size_of_in_element_count;
mod size_of_ref;
mod slow_vector_initialization;
mod std_instead_of_core;
mod strings;
mod strlen_on_c_strings;
mod suspicious_operation_groupings;
mod suspicious_trait_impl;
mod suspicious_xor_used_as_pow;
mod swap;
mod swap_ptr_to_ref;
mod tabs_in_doc_comments;
mod temporary_assignment;
mod to_digit_is_some;
mod trailing_empty_array;
mod trait_bounds;
mod transmute;
mod types;
mod undocumented_unsafe_blocks;
mod unicode;
mod uninit_vec;
mod unit_return_expecting_ord;
mod unit_types;
mod unnamed_address;
mod unnecessary_owned_empty_strings;
mod unnecessary_self_imports;
mod unnecessary_wraps;
mod unnested_or_patterns;
mod unsafe_removed_from_name;
mod unused_async;
mod unused_io_amount;
mod unused_peekable;
mod unused_rounding;
mod unused_self;
mod unused_unit;
mod unwrap;
mod unwrap_in_result;
mod upper_case_acronyms;
mod use_self;
mod useless_conversion;
mod vec;
mod vec_init_then_push;
mod wildcard_imports;
mod write;
mod zero_div_zero;
mod zero_sized_map_values;
// end lints modules, do not remove this comment, it’s used in `update_lints`

use crate::utils::conf::{format_error, TryConf};
pub use crate::utils::conf::{lookup_conf_file, Conf};

/// Register all pre expansion lints
///
/// Pre-expansion lints run before any macro expansion has happened.
///
/// Note that due to the architecture of the compiler, currently `cfg_attr` attributes on crate
/// level (i.e `#![cfg_attr(...)]`) will still be expanded even when using a pre-expansion pass.
///
/// Used in `./src/driver.rs`.
pub fn register_pre_expansion_lints(store: &mut rustc_lint::LintStore, sess: &Session, conf: &Conf) {
    // NOTE: Do not add any more pre-expansion passes. These should be removed eventually.
    let msrv = Msrv::read(&conf.msrv, sess);
    let msrv = move || msrv.clone();

    store.register_pre_expansion_pass(move || Box::new(attrs::EarlyAttributes { msrv: msrv() }));
}

#[doc(hidden)]
pub fn read_conf(sess: &Session, path: &io::Result<Option<PathBuf>>) -> Conf {
    let file_name = match path {
        Ok(Some(path)) => path,
        Ok(None) => return Conf::default(),
        Err(error) => {
            sess.struct_err(format!("error finding Clippy's configuration file: {error}"))
                .emit();
            return Conf::default();
        },
    };

    let TryConf { conf, errors, warnings } = utils::conf::read(file_name);
    // all conf errors are non-fatal, we just use the default conf in case of error
    for error in errors {
        sess.err(format!(
            "error reading Clippy's configuration file `{}`: {}",
            file_name.display(),
            format_error(error)
        ));
    }

    for warning in warnings {
        sess.struct_warn(format!(
            "error reading Clippy's configuration file `{}`: {}",
            file_name.display(),
            format_error(warning)
        ))
        .emit();
    }

    conf
}

#[derive(Default)]
struct RegistrationGroups {
    all: Vec<LintId>,
    cargo: Vec<LintId>,
    complexity: Vec<LintId>,
    correctness: Vec<LintId>,
    nursery: Vec<LintId>,
    pedantic: Vec<LintId>,
    perf: Vec<LintId>,
    restriction: Vec<LintId>,
    style: Vec<LintId>,
    suspicious: Vec<LintId>,
    #[cfg(feature = "internal")]
    internal: Vec<LintId>,
}

impl RegistrationGroups {
    #[rustfmt::skip]
    fn register(self, store: &mut rustc_lint::LintStore) {
        store.register_group(true, "clippy::all", Some("clippy_all"), self.all);
        store.register_group(true, "clippy::cargo", Some("clippy_cargo"), self.cargo);
        store.register_group(true, "clippy::complexity", Some("clippy_complexity"), self.complexity);
        store.register_group(true, "clippy::correctness", Some("clippy_correctness"), self.correctness);
        store.register_group(true, "clippy::nursery", Some("clippy_nursery"), self.nursery);
        store.register_group(true, "clippy::pedantic", Some("clippy_pedantic"), self.pedantic);
        store.register_group(true, "clippy::perf", Some("clippy_perf"), self.perf);
        store.register_group(true, "clippy::restriction", Some("clippy_restriction"), self.restriction);
        store.register_group(true, "clippy::style", Some("clippy_style"), self.style);
        store.register_group(true, "clippy::suspicious", Some("clippy_suspicious"), self.suspicious);
        #[cfg(feature = "internal")]
        store.register_group(true, "clippy::internal", Some("clippy_internal"), self.internal);
    }
}

#[derive(Copy, Clone)]
pub(crate) enum LintCategory {
    Cargo,
    Complexity,
    Correctness,
    Nursery,
    Pedantic,
    Perf,
    Restriction,
    Style,
    Suspicious,
    #[cfg(feature = "internal")]
    Internal,
}
#[allow(clippy::enum_glob_use)]
use LintCategory::*;

impl LintCategory {
    fn is_all(self) -> bool {
        matches!(self, Correctness | Suspicious | Style | Complexity | Perf)
    }

    fn group(self, groups: &mut RegistrationGroups) -> &mut Vec<LintId> {
        match self {
            Cargo => &mut groups.cargo,
            Complexity => &mut groups.complexity,
            Correctness => &mut groups.correctness,
            Nursery => &mut groups.nursery,
            Pedantic => &mut groups.pedantic,
            Perf => &mut groups.perf,
            Restriction => &mut groups.restriction,
            Style => &mut groups.style,
            Suspicious => &mut groups.suspicious,
            #[cfg(feature = "internal")]
            Internal => &mut groups.internal,
        }
    }
}

pub(crate) struct LintInfo {
    /// Double reference to maintain pointer equality
    lint: &'static &'static Lint,
    category: LintCategory,
    explanation: &'static str,
}

pub fn explain(name: &str) {
    let target = format!("clippy::{}", name.to_ascii_uppercase());
    match declared_lints::LINTS.iter().find(|info| info.lint.name == target) {
        Some(info) => print!("{}", info.explanation),
        None => println!("unknown lint: {name}"),
    }
}

fn register_categories(store: &mut rustc_lint::LintStore) {
    let mut groups = RegistrationGroups::default();

    for LintInfo { lint, category, .. } in declared_lints::LINTS {
        if category.is_all() {
            groups.all.push(LintId::of(lint));
        }

        category.group(&mut groups).push(LintId::of(lint));
    }

    let lints: Vec<&'static Lint> = declared_lints::LINTS.iter().map(|info| *info.lint).collect();

    store.register_lints(&lints);
    groups.register(store);
}

/// Register all lints and lint groups with the rustc plugin registry
///
/// Used in `./src/driver.rs`.
#[expect(clippy::too_many_lines)]
pub fn register_plugins(store: &mut rustc_lint::LintStore, sess: &Session, conf: &Conf) {
    register_removed_non_tool_lints(store);
    register_categories(store);

    include!("lib.deprecated.rs");

    #[cfg(feature = "internal")]
    {
        if std::env::var("ENABLE_METADATA_COLLECTION").eq(&Ok("1".to_string())) {
            store.register_late_pass(|_| Box::new(utils::internal_lints::metadata_collector::MetadataCollector::new()));
            return;
        }
    }

    // all the internal lints
    #[cfg(feature = "internal")]
    {
        store.register_early_pass(|| Box::new(utils::internal_lints::clippy_lints_internal::ClippyLintsInternal));
        store.register_early_pass(|| Box::new(utils::internal_lints::produce_ice::ProduceIce));
        store.register_late_pass(|_| Box::new(utils::internal_lints::collapsible_calls::CollapsibleCalls));
        store.register_late_pass(|_| {
            Box::new(utils::internal_lints::compiler_lint_functions::CompilerLintFunctions::new())
        });
        store.register_late_pass(|_| Box::new(utils::internal_lints::if_chain_style::IfChainStyle));
        store.register_late_pass(|_| Box::new(utils::internal_lints::invalid_paths::InvalidPaths));
        store.register_late_pass(|_| {
            Box::<utils::internal_lints::interning_defined_symbol::InterningDefinedSymbol>::default()
        });
        store.register_late_pass(|_| {
            Box::<utils::internal_lints::lint_without_lint_pass::LintWithoutLintPass>::default()
        });
        store.register_late_pass(|_| Box::<utils::internal_lints::unnecessary_def_path::UnnecessaryDefPath>::default());
        store.register_late_pass(|_| Box::new(utils::internal_lints::outer_expn_data_pass::OuterExpnDataPass));
        store.register_late_pass(|_| Box::new(utils::internal_lints::msrv_attr_impl::MsrvAttrImpl));
    }

    let arithmetic_side_effects_allowed = conf.arithmetic_side_effects_allowed.clone();
    let arithmetic_side_effects_allowed_binary = conf.arithmetic_side_effects_allowed_binary.clone();
    let arithmetic_side_effects_allowed_unary = conf.arithmetic_side_effects_allowed_unary.clone();
    store.register_late_pass(move |_| {
        Box::new(operators::arithmetic_side_effects::ArithmeticSideEffects::new(
            arithmetic_side_effects_allowed
                .iter()
                .flat_map(|el| [[el.clone(), "*".to_string()], ["*".to_string(), el.clone()]])
                .chain(arithmetic_side_effects_allowed_binary.clone())
                .collect(),
            arithmetic_side_effects_allowed
                .iter()
                .chain(arithmetic_side_effects_allowed_unary.iter())
                .cloned()
                .collect(),
        ))
    });
    store.register_late_pass(|_| Box::new(utils::dump_hir::DumpHir));
    store.register_late_pass(|_| Box::new(utils::author::Author));
    let await_holding_invalid_types = conf.await_holding_invalid_types.clone();
    store.register_late_pass(move |_| {
        Box::new(await_holding_invalid::AwaitHolding::new(
            await_holding_invalid_types.clone(),
        ))
    });
    store.register_late_pass(|_| Box::new(serde_api::SerdeApi));
    let vec_box_size_threshold = conf.vec_box_size_threshold;
    let type_complexity_threshold = conf.type_complexity_threshold;
    let avoid_breaking_exported_api = conf.avoid_breaking_exported_api;
    store.register_late_pass(move |_| {
        Box::new(types::Types::new(
            vec_box_size_threshold,
            type_complexity_threshold,
            avoid_breaking_exported_api,
        ))
    });
    store.register_late_pass(|_| Box::new(booleans::NonminimalBool));
    store.register_late_pass(|_| Box::new(enum_clike::UnportableVariant));
    store.register_late_pass(|_| Box::new(float_literal::FloatLiteral));
    store.register_late_pass(|_| Box::new(ptr::Ptr));
    store.register_late_pass(|_| Box::new(needless_bool::NeedlessBool));
    store.register_late_pass(|_| Box::new(needless_bool::BoolComparison));
    store.register_late_pass(|_| Box::new(needless_for_each::NeedlessForEach));
    store.register_late_pass(|_| Box::<misc::LintPass>::default());
    store.register_late_pass(|_| Box::new(eta_reduction::EtaReduction));
    store.register_late_pass(|_| Box::new(mut_mut::MutMut));
    store.register_late_pass(|_| Box::new(mut_reference::UnnecessaryMutPassed));
    store.register_late_pass(|_| Box::<significant_drop_tightening::SignificantDropTightening<'_>>::default());
    store.register_late_pass(|_| Box::new(len_zero::LenZero));
    store.register_late_pass(|_| Box::new(attrs::Attributes));
    store.register_late_pass(|_| Box::new(blocks_in_if_conditions::BlocksInIfConditions));
    store.register_late_pass(|_| Box::new(unicode::Unicode));
    store.register_late_pass(|_| Box::new(uninit_vec::UninitVec));
    store.register_late_pass(|_| Box::new(unit_return_expecting_ord::UnitReturnExpectingOrd));
    store.register_late_pass(|_| Box::new(strings::StringAdd));
    store.register_late_pass(|_| Box::new(implicit_return::ImplicitReturn));
    store.register_late_pass(|_| Box::new(implicit_saturating_sub::ImplicitSaturatingSub));
    store.register_late_pass(|_| Box::new(default_numeric_fallback::DefaultNumericFallback));
    store.register_late_pass(|_| Box::new(inconsistent_struct_constructor::InconsistentStructConstructor));
    store.register_late_pass(|_| Box::new(non_octal_unix_permissions::NonOctalUnixPermissions));
    store.register_early_pass(|| Box::new(unnecessary_self_imports::UnnecessarySelfImports));

    let msrv = Msrv::read(&conf.msrv, sess);
    let msrv = move || msrv.clone();
    let avoid_breaking_exported_api = conf.avoid_breaking_exported_api;
    let allow_expect_in_tests = conf.allow_expect_in_tests;
    let allow_unwrap_in_tests = conf.allow_unwrap_in_tests;
    let suppress_restriction_lint_in_const = conf.suppress_restriction_lint_in_const;
    store.register_late_pass(move |_| Box::new(approx_const::ApproxConstant::new(msrv())));
    store.register_late_pass(move |_| {
        Box::new(methods::Methods::new(
            avoid_breaking_exported_api,
            msrv(),
            allow_expect_in_tests,
            allow_unwrap_in_tests,
        ))
    });
    store.register_late_pass(move |_| Box::new(matches::Matches::new(msrv())));
    let matches_for_let_else = conf.matches_for_let_else;
    store.register_late_pass(move |_| Box::new(manual_let_else::ManualLetElse::new(msrv(), matches_for_let_else)));
    store.register_early_pass(move || Box::new(manual_non_exhaustive::ManualNonExhaustiveStruct::new(msrv())));
    store.register_late_pass(move |_| Box::new(manual_non_exhaustive::ManualNonExhaustiveEnum::new(msrv())));
    store.register_late_pass(move |_| Box::new(manual_strip::ManualStrip::new(msrv())));
    store.register_early_pass(move || Box::new(redundant_static_lifetimes::RedundantStaticLifetimes::new(msrv())));
    store.register_early_pass(move || Box::new(redundant_field_names::RedundantFieldNames::new(msrv())));
    store.register_late_pass(move |_| Box::new(checked_conversions::CheckedConversions::new(msrv())));
    store.register_late_pass(move |_| Box::new(mem_replace::MemReplace::new(msrv())));
    store.register_late_pass(move |_| Box::new(ranges::Ranges::new(msrv())));
    store.register_late_pass(move |_| Box::new(from_over_into::FromOverInto::new(msrv())));
    store.register_late_pass(move |_| Box::new(use_self::UseSelf::new(msrv())));
    store.register_late_pass(move |_| Box::new(missing_const_for_fn::MissingConstForFn::new(msrv())));
    store.register_late_pass(move |_| Box::new(needless_question_mark::NeedlessQuestionMark));
    store.register_late_pass(move |_| Box::new(casts::Casts::new(msrv())));
    store.register_early_pass(move || Box::new(unnested_or_patterns::UnnestedOrPatterns::new(msrv())));
    store.register_late_pass(|_| Box::new(size_of_in_element_count::SizeOfInElementCount));
    store.register_late_pass(|_| Box::new(same_name_method::SameNameMethod));
    let max_suggested_slice_pattern_length = conf.max_suggested_slice_pattern_length;
    store.register_late_pass(move |_| {
        Box::new(index_refutable_slice::IndexRefutableSlice::new(
            max_suggested_slice_pattern_length,
            msrv(),
        ))
    });
    store.register_late_pass(|_| Box::<shadow::Shadow>::default());
    store.register_late_pass(|_| Box::new(unit_types::UnitTypes));
    store.register_late_pass(|_| Box::new(loops::Loops));
    store.register_late_pass(|_| Box::<main_recursion::MainRecursion>::default());
    store.register_late_pass(|_| Box::new(lifetimes::Lifetimes));
    store.register_late_pass(|_| Box::new(entry::HashMapPass));
    store.register_late_pass(|_| Box::new(minmax::MinMaxPass));
    store.register_late_pass(|_| Box::new(zero_div_zero::ZeroDiv));
    store.register_late_pass(|_| Box::new(mutex_atomic::Mutex));
    store.register_late_pass(|_| Box::new(needless_update::NeedlessUpdate));
    store.register_late_pass(|_| Box::new(needless_borrowed_ref::NeedlessBorrowedRef));
    store.register_late_pass(|_| Box::new(borrow_deref_ref::BorrowDerefRef));
    store.register_late_pass(|_| Box::new(no_effect::NoEffect));
    store.register_late_pass(|_| Box::new(temporary_assignment::TemporaryAssignment));
    store.register_late_pass(move |_| Box::new(transmute::Transmute::new(msrv())));
    let cognitive_complexity_threshold = conf.cognitive_complexity_threshold;
    store.register_late_pass(move |_| {
        Box::new(cognitive_complexity::CognitiveComplexity::new(
            cognitive_complexity_threshold,
        ))
    });
    let too_large_for_stack = conf.too_large_for_stack;
    store.register_late_pass(move |_| Box::new(escape::BoxedLocal { too_large_for_stack }));
    store.register_late_pass(move |_| Box::new(vec::UselessVec { too_large_for_stack }));
    store.register_late_pass(|_| Box::new(panic_unimplemented::PanicUnimplemented));
    store.register_late_pass(|_| Box::new(strings::StringLitAsBytes));
    store.register_late_pass(|_| Box::new(derive::Derive));
    store.register_late_pass(move |_| Box::new(derivable_impls::DerivableImpls::new(msrv())));
    store.register_late_pass(|_| Box::new(drop_forget_ref::DropForgetRef));
    store.register_late_pass(|_| Box::new(empty_enum::EmptyEnum));
    store.register_late_pass(|_| Box::new(invalid_upcast_comparisons::InvalidUpcastComparisons));
    store.register_late_pass(|_| Box::new(regex::Regex));
    store.register_late_pass(|_| Box::new(copies::CopyAndPaste));
    store.register_late_pass(|_| Box::new(copy_iterator::CopyIterator));
    store.register_late_pass(|_| Box::new(format::UselessFormat));
    store.register_late_pass(|_| Box::new(swap::Swap));
    store.register_late_pass(|_| Box::new(overflow_check_conditional::OverflowCheckConditional));
    store.register_late_pass(|_| Box::<new_without_default::NewWithoutDefault>::default());
    let disallowed_names = conf.disallowed_names.iter().cloned().collect::<FxHashSet<_>>();
    store.register_late_pass(move |_| Box::new(disallowed_names::DisallowedNames::new(disallowed_names.clone())));
    let too_many_arguments_threshold = conf.too_many_arguments_threshold;
    let too_many_lines_threshold = conf.too_many_lines_threshold;
    let large_error_threshold = conf.large_error_threshold;
    store.register_late_pass(move |_| {
        Box::new(functions::Functions::new(
            too_many_arguments_threshold,
            too_many_lines_threshold,
            large_error_threshold,
        ))
    });
    let doc_valid_idents = conf.doc_valid_idents.iter().cloned().collect::<FxHashSet<_>>();
    store.register_late_pass(move |_| Box::new(doc::DocMarkdown::new(doc_valid_idents.clone())));
    store.register_late_pass(|_| Box::new(neg_multiply::NegMultiply));
    store.register_late_pass(|_| Box::new(mem_forget::MemForget));
    store.register_late_pass(|_| Box::new(let_if_seq::LetIfSeq));
    store.register_late_pass(|_| Box::new(mixed_read_write_in_expression::EvalOrderDependence));
    store.register_late_pass(|_| Box::new(missing_doc::MissingDoc::new()));
    store.register_late_pass(|_| Box::new(missing_inline::MissingInline));
    store.register_late_pass(move |_| Box::new(exhaustive_items::ExhaustiveItems));
    store.register_late_pass(|_| Box::new(match_result_ok::MatchResultOk));
    store.register_late_pass(|_| Box::new(partialeq_ne_impl::PartialEqNeImpl));
    store.register_late_pass(|_| Box::new(unused_io_amount::UnusedIoAmount));
    let enum_variant_size_threshold = conf.enum_variant_size_threshold;
    store.register_late_pass(move |_| Box::new(large_enum_variant::LargeEnumVariant::new(enum_variant_size_threshold)));
    store.register_late_pass(|_| Box::new(explicit_write::ExplicitWrite));
    store.register_late_pass(|_| Box::new(needless_pass_by_value::NeedlessPassByValue));
    let pass_by_ref_or_value = pass_by_ref_or_value::PassByRefOrValue::new(
        conf.trivial_copy_size_limit,
        conf.pass_by_value_size_limit,
        conf.avoid_breaking_exported_api,
        &sess.target,
    );
    store.register_late_pass(move |_| Box::new(pass_by_ref_or_value));
    store.register_late_pass(|_| Box::new(ref_option_ref::RefOptionRef));
    store.register_late_pass(|_| Box::new(infinite_iter::InfiniteIter));
    store.register_late_pass(|_| Box::new(inline_fn_without_body::InlineFnWithoutBody));
    store.register_late_pass(|_| Box::<useless_conversion::UselessConversion>::default());
    store.register_late_pass(|_| Box::new(implicit_hasher::ImplicitHasher));
    store.register_late_pass(|_| Box::new(fallible_impl_from::FallibleImplFrom));
    store.register_late_pass(|_| Box::new(question_mark::QuestionMark));
    store.register_late_pass(|_| Box::new(question_mark_used::QuestionMarkUsed));
    store.register_early_pass(|| Box::new(suspicious_operation_groupings::SuspiciousOperationGroupings));
    store.register_late_pass(|_| Box::new(suspicious_trait_impl::SuspiciousImpl));
    store.register_late_pass(|_| Box::new(map_unit_fn::MapUnit));
    store.register_late_pass(|_| Box::new(inherent_impl::MultipleInherentImpl));
    store.register_late_pass(|_| Box::new(neg_cmp_op_on_partial_ord::NoNegCompOpForPartialOrd));
    store.register_late_pass(|_| Box::new(unwrap::Unwrap));
    store.register_late_pass(move |_| {
        Box::new(indexing_slicing::IndexingSlicing::new(
            suppress_restriction_lint_in_const,
        ))
    });
    store.register_late_pass(|_| Box::new(non_copy_const::NonCopyConst));
    store.register_late_pass(|_| Box::new(ptr_offset_with_cast::PtrOffsetWithCast));
    store.register_late_pass(|_| Box::new(redundant_clone::RedundantClone));
    store.register_late_pass(|_| Box::new(slow_vector_initialization::SlowVectorInit));
    store.register_late_pass(move |_| Box::new(unnecessary_wraps::UnnecessaryWraps::new(avoid_breaking_exported_api)));
    store.register_late_pass(|_| Box::new(assertions_on_constants::AssertionsOnConstants));
    store.register_late_pass(|_| Box::new(assertions_on_result_states::AssertionsOnResultStates));
    store.register_late_pass(|_| Box::new(inherent_to_string::InherentToString));
    let max_trait_bounds = conf.max_trait_bounds;
    store.register_late_pass(move |_| Box::new(trait_bounds::TraitBounds::new(max_trait_bounds)));
    store.register_late_pass(|_| Box::new(comparison_chain::ComparisonChain));
    let ignore_interior_mutability = conf.ignore_interior_mutability.clone();
    store.register_late_pass(move |_| Box::new(mut_key::MutableKeyType::new(ignore_interior_mutability.clone())));
    store.register_early_pass(|| Box::new(reference::DerefAddrOf));
    store.register_early_pass(|| Box::new(double_parens::DoubleParens));
    store.register_late_pass(|_| Box::new(format_impl::FormatImpl::new()));
    store.register_early_pass(|| Box::new(unsafe_removed_from_name::UnsafeNameRemoval));
    store.register_early_pass(|| Box::new(else_if_without_else::ElseIfWithoutElse));
    store.register_early_pass(|| Box::new(int_plus_one::IntPlusOne));
    store.register_early_pass(|| Box::new(formatting::Formatting));
    store.register_early_pass(|| Box::new(misc_early::MiscEarlyLints));
    store.register_early_pass(|| Box::new(redundant_closure_call::RedundantClosureCall));
    store.register_late_pass(|_| Box::new(redundant_closure_call::RedundantClosureCall));
    store.register_early_pass(|| Box::new(unused_unit::UnusedUnit));
    store.register_late_pass(|_| Box::new(returns::Return));
    store.register_early_pass(|| Box::new(collapsible_if::CollapsibleIf));
    store.register_early_pass(|| Box::new(items_after_statements::ItemsAfterStatements));
    store.register_early_pass(|| Box::new(precedence::Precedence));
    store.register_late_pass(|_| Box::new(needless_parens_on_range_literals::NeedlessParensOnRangeLiterals));
    store.register_early_pass(|| Box::new(needless_continue::NeedlessContinue));
    store.register_early_pass(|| Box::new(redundant_else::RedundantElse));
    store.register_late_pass(|_| Box::new(create_dir::CreateDir));
    store.register_early_pass(|| Box::new(needless_arbitrary_self_type::NeedlessArbitrarySelfType));
    let literal_representation_lint_fraction_readability = conf.unreadable_literal_lint_fractions;
    store.register_early_pass(move || {
        Box::new(literal_representation::LiteralDigitGrouping::new(
            literal_representation_lint_fraction_readability,
        ))
    });
    let literal_representation_threshold = conf.literal_representation_threshold;
    store.register_early_pass(move || {
        Box::new(literal_representation::DecimalLiteralRepresentation::new(
            literal_representation_threshold,
        ))
    });
    let enum_variant_name_threshold = conf.enum_variant_name_threshold;
    store.register_late_pass(move |_| {
        Box::new(enum_variants::EnumVariantNames::new(
            enum_variant_name_threshold,
            avoid_breaking_exported_api,
        ))
    });
    store.register_early_pass(|| Box::new(tabs_in_doc_comments::TabsInDocComments));
    let upper_case_acronyms_aggressive = conf.upper_case_acronyms_aggressive;
    store.register_late_pass(move |_| {
        Box::new(upper_case_acronyms::UpperCaseAcronyms::new(
            avoid_breaking_exported_api,
            upper_case_acronyms_aggressive,
        ))
    });
    store.register_late_pass(|_| Box::<default::Default>::default());
    store.register_late_pass(move |_| Box::new(unused_self::UnusedSelf::new(avoid_breaking_exported_api)));
    store.register_late_pass(|_| Box::new(mutable_debug_assertion::DebugAssertWithMutCall));
    store.register_late_pass(|_| Box::new(exit::Exit));
    store.register_late_pass(|_| Box::new(to_digit_is_some::ToDigitIsSome));
    let array_size_threshold = conf.array_size_threshold;
    store.register_late_pass(move |_| Box::new(large_stack_arrays::LargeStackArrays::new(array_size_threshold)));
    store.register_late_pass(move |_| Box::new(large_const_arrays::LargeConstArrays::new(array_size_threshold)));
    store.register_late_pass(|_| Box::new(floating_point_arithmetic::FloatingPointArithmetic));
    store.register_early_pass(|| Box::new(as_conversions::AsConversions));
    store.register_late_pass(|_| Box::new(let_underscore::LetUnderscore));
    store.register_early_pass(|| Box::<single_component_path_imports::SingleComponentPathImports>::default());
    let max_fn_params_bools = conf.max_fn_params_bools;
    let max_struct_bools = conf.max_struct_bools;
    store.register_late_pass(move |_| {
        Box::new(excessive_bools::ExcessiveBools::new(
            max_struct_bools,
            max_fn_params_bools,
        ))
    });
    store.register_early_pass(|| Box::new(option_env_unwrap::OptionEnvUnwrap));
    let warn_on_all_wildcard_imports = conf.warn_on_all_wildcard_imports;
    store.register_late_pass(move |_| Box::new(wildcard_imports::WildcardImports::new(warn_on_all_wildcard_imports)));
    store.register_late_pass(|_| Box::<redundant_pub_crate::RedundantPubCrate>::default());
    store.register_late_pass(|_| Box::new(unnamed_address::UnnamedAddress));
    store.register_late_pass(move |_| Box::new(dereference::Dereferencing::new(msrv())));
    store.register_late_pass(|_| Box::new(option_if_let_else::OptionIfLetElse));
    store.register_late_pass(|_| Box::new(future_not_send::FutureNotSend));
    store.register_late_pass(|_| Box::new(if_let_mutex::IfLetMutex));
    store.register_late_pass(|_| Box::new(if_not_else::IfNotElse));
    store.register_late_pass(|_| Box::new(equatable_if_let::PatternEquality));
    store.register_late_pass(|_| Box::new(manual_async_fn::ManualAsyncFn));
    store.register_late_pass(|_| Box::new(panic_in_result_fn::PanicInResultFn));
    let single_char_binding_names_threshold = conf.single_char_binding_names_threshold;
    store.register_early_pass(move || {
        Box::new(non_expressive_names::NonExpressiveNames {
            single_char_binding_names_threshold,
        })
    });
    let macro_matcher = conf.standard_macro_braces.iter().cloned().collect::<FxHashSet<_>>();
    store.register_early_pass(move || Box::new(nonstandard_macro_braces::MacroBraces::new(&macro_matcher)));
    store.register_late_pass(|_| Box::<macro_use::MacroUseImports>::default());
    store.register_late_pass(|_| Box::new(pattern_type_mismatch::PatternTypeMismatch));
    store.register_late_pass(|_| Box::new(unwrap_in_result::UnwrapInResult));
    store.register_late_pass(|_| Box::new(semicolon_if_nothing_returned::SemicolonIfNothingReturned));
    store.register_late_pass(|_| Box::new(async_yields_async::AsyncYieldsAsync));
    let disallowed_macros = conf.disallowed_macros.clone();
    store.register_late_pass(move |_| Box::new(disallowed_macros::DisallowedMacros::new(disallowed_macros.clone())));
    let disallowed_methods = conf.disallowed_methods.clone();
    store.register_late_pass(move |_| Box::new(disallowed_methods::DisallowedMethods::new(disallowed_methods.clone())));
    store.register_early_pass(|| Box::new(asm_syntax::InlineAsmX86AttSyntax));
    store.register_early_pass(|| Box::new(asm_syntax::InlineAsmX86IntelSyntax));
    store.register_late_pass(|_| Box::new(empty_drop::EmptyDrop));
    store.register_late_pass(|_| Box::new(strings::StrToString));
    store.register_late_pass(|_| Box::new(strings::StringToString));
    store.register_late_pass(|_| Box::new(zero_sized_map_values::ZeroSizedMapValues));
    store.register_late_pass(|_| Box::<vec_init_then_push::VecInitThenPush>::default());
    store.register_late_pass(|_| Box::new(redundant_slicing::RedundantSlicing));
    store.register_late_pass(|_| Box::new(from_str_radix_10::FromStrRadix10));
    store.register_late_pass(move |_| Box::new(if_then_some_else_none::IfThenSomeElseNone::new(msrv())));
    store.register_late_pass(|_| Box::new(bool_assert_comparison::BoolAssertComparison));
    store.register_early_pass(move || Box::new(module_style::ModStyle));
    store.register_late_pass(|_| Box::new(unused_async::UnusedAsync));
    let disallowed_types = conf.disallowed_types.clone();
    store.register_late_pass(move |_| Box::new(disallowed_types::DisallowedTypes::new(disallowed_types.clone())));
    let import_renames = conf.enforced_import_renames.clone();
    store.register_late_pass(move |_| {
        Box::new(missing_enforced_import_rename::ImportRename::new(
            import_renames.clone(),
        ))
    });
    let scripts = conf.allowed_scripts.clone();
    store.register_early_pass(move || Box::new(disallowed_script_idents::DisallowedScriptIdents::new(&scripts)));
    store.register_late_pass(|_| Box::new(strlen_on_c_strings::StrlenOnCStrings));
    store.register_late_pass(move |_| Box::new(self_named_constructors::SelfNamedConstructors));
    store.register_late_pass(move |_| Box::new(iter_not_returning_iterator::IterNotReturningIterator));
    store.register_late_pass(move |_| Box::new(manual_assert::ManualAssert));
    let enable_raw_pointer_heuristic_for_send = conf.enable_raw_pointer_heuristic_for_send;
    store.register_late_pass(move |_| {
        Box::new(non_send_fields_in_send_ty::NonSendFieldInSendTy::new(
            enable_raw_pointer_heuristic_for_send,
        ))
    });
    store.register_late_pass(move |_| Box::new(undocumented_unsafe_blocks::UndocumentedUnsafeBlocks));
    let allow_mixed_uninlined = conf.allow_mixed_uninlined_format_args;
    store.register_late_pass(move |_| Box::new(format_args::FormatArgs::new(msrv(), allow_mixed_uninlined)));
    store.register_late_pass(|_| Box::new(trailing_empty_array::TrailingEmptyArray));
    store.register_early_pass(|| Box::new(octal_escapes::OctalEscapes));
    store.register_late_pass(|_| Box::new(needless_late_init::NeedlessLateInit));
    store.register_late_pass(|_| Box::new(return_self_not_must_use::ReturnSelfNotMustUse));
    store.register_late_pass(|_| Box::new(init_numbered_fields::NumberedFields));
    store.register_early_pass(|| Box::new(single_char_lifetime_names::SingleCharLifetimeNames));
    store.register_late_pass(move |_| Box::new(manual_bits::ManualBits::new(msrv())));
    store.register_late_pass(|_| Box::new(default_union_representation::DefaultUnionRepresentation));
    store.register_late_pass(|_| Box::<only_used_in_recursion::OnlyUsedInRecursion>::default());
    let allow_dbg_in_tests = conf.allow_dbg_in_tests;
    store.register_late_pass(move |_| Box::new(dbg_macro::DbgMacro::new(allow_dbg_in_tests)));
    let allow_print_in_tests = conf.allow_print_in_tests;
    store.register_late_pass(move |_| Box::new(write::Write::new(allow_print_in_tests)));
    let cargo_ignore_publish = conf.cargo_ignore_publish;
    store.register_late_pass(move |_| {
        Box::new(cargo::Cargo {
            ignore_publish: cargo_ignore_publish,
        })
    });
    store.register_early_pass(|| Box::new(crate_in_macro_def::CrateInMacroDef));
    store.register_early_pass(|| Box::new(empty_structs_with_brackets::EmptyStructsWithBrackets));
    store.register_late_pass(|_| Box::new(unnecessary_owned_empty_strings::UnnecessaryOwnedEmptyStrings));
    store.register_early_pass(|| Box::new(pub_use::PubUse));
    store.register_late_pass(|_| Box::new(format_push_string::FormatPushString));
    let max_include_file_size = conf.max_include_file_size;
    store.register_late_pass(move |_| Box::new(large_include_file::LargeIncludeFile::new(max_include_file_size)));
    store.register_late_pass(|_| Box::new(strings::TrimSplitWhitespace));
    store.register_late_pass(|_| Box::new(rc_clone_in_vec_init::RcCloneInVecInit));
    store.register_early_pass(|| Box::<duplicate_mod::DuplicateMod>::default());
    store.register_early_pass(|| Box::new(unused_rounding::UnusedRounding));
    store.register_early_pass(move || Box::new(almost_complete_range::AlmostCompleteRange::new(msrv())));
    store.register_late_pass(|_| Box::new(swap_ptr_to_ref::SwapPtrToRef));
    store.register_late_pass(|_| Box::new(mismatching_type_param_order::TypeParamMismatch));
    store.register_late_pass(|_| Box::new(read_zero_byte_vec::ReadZeroByteVec));
    store.register_late_pass(|_| Box::new(default_instead_of_iter_empty::DefaultIterEmpty));
    store.register_late_pass(move |_| Box::new(manual_rem_euclid::ManualRemEuclid::new(msrv())));
    store.register_late_pass(move |_| Box::new(manual_retain::ManualRetain::new(msrv())));
    let verbose_bit_mask_threshold = conf.verbose_bit_mask_threshold;
    store.register_late_pass(move |_| Box::new(operators::Operators::new(verbose_bit_mask_threshold)));
    store.register_late_pass(|_| Box::new(invalid_utf8_in_unchecked::InvalidUtf8InUnchecked));
    store.register_late_pass(|_| Box::<std_instead_of_core::StdReexports>::default());
    store.register_late_pass(move |_| Box::new(instant_subtraction::InstantSubtraction::new(msrv())));
    store.register_late_pass(|_| Box::new(partialeq_to_none::PartialeqToNone));
    store.register_late_pass(move |_| Box::new(manual_clamp::ManualClamp::new(msrv())));
    store.register_late_pass(|_| Box::new(manual_string_new::ManualStringNew));
    store.register_late_pass(|_| Box::new(unused_peekable::UnusedPeekable));
    store.register_early_pass(|| Box::new(multi_assignments::MultiAssignments));
    store.register_late_pass(|_| Box::new(bool_to_int_with_if::BoolToIntWithIf));
    store.register_late_pass(|_| Box::new(box_default::BoxDefault));
    store.register_late_pass(|_| Box::new(implicit_saturating_add::ImplicitSaturatingAdd));
    store.register_early_pass(|| Box::new(partial_pub_fields::PartialPubFields));
    store.register_late_pass(|_| Box::new(missing_trait_methods::MissingTraitMethods));
    store.register_late_pass(|_| Box::new(from_raw_with_void_ptr::FromRawWithVoidPtr));
    store.register_late_pass(|_| Box::new(suspicious_xor_used_as_pow::ConfusingXorAndPow));
    store.register_late_pass(move |_| Box::new(manual_is_ascii_check::ManualIsAsciiCheck::new(msrv())));
    store.register_late_pass(|_| Box::new(semicolon_block::SemicolonBlock));
    store.register_late_pass(|_| Box::new(fn_null_check::FnNullCheck));
    store.register_late_pass(|_| Box::new(permissions_set_readonly_false::PermissionsSetReadonlyFalse));
    store.register_late_pass(|_| Box::new(size_of_ref::SizeOfRef));
    store.register_late_pass(|_| Box::new(multiple_unsafe_ops_per_block::MultipleUnsafeOpsPerBlock));
    store.register_late_pass(|_| Box::new(extra_unused_type_parameters::ExtraUnusedTypeParameters));
    // add lints here, do not remove this comment, it's used in `new_lint`
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
    for (old_name, new_name) in renamed_lints::RENAMED_LINTS {
        ls.register_renamed(old_name, new_name);
    }
}

// only exists to let the dogfood integration test works.
// Don't run clippy as an executable directly
#[allow(dead_code)]
fn main() {
    panic!("Please use the cargo-clippy executable");
}
