#![feature(array_windows)]
#![feature(box_patterns)]
#![feature(macro_metavar_expr_concat)]
#![feature(f128)]
#![feature(f16)]
#![feature(if_let_guard)]
#![feature(iter_intersperse)]
#![feature(iter_partition_in_place)]
#![feature(never_type)]
#![feature(round_char_boundary)]
#![feature(rustc_private)]
#![feature(stmt_expr_attributes)]
#![feature(unwrap_infallible)]
#![recursion_limit = "512"]
#![allow(
    clippy::missing_docs_in_private_items,
    clippy::must_use_candidate,
    rustc::diagnostic_outside_of_impl,
    rustc::untranslatable_diagnostic,
    clippy::literal_string_with_formatting_args
)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    rust_2018_idioms,
    unused_lifetimes,
    unused_qualifications,
    rustc::internal
)]

// FIXME: switch to something more ergonomic here, once available.
// (Currently there is no way to opt into sysroot crates without `extern crate`.)
extern crate pulldown_cmark;
extern crate rustc_abi;
extern crate rustc_arena;
extern crate rustc_ast;
extern crate rustc_ast_pretty;
extern crate rustc_attr_data_structures;
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
extern crate rustc_parse_format;
extern crate rustc_resolve;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_target;
extern crate rustc_trait_selection;
extern crate thin_vec;

#[macro_use]
mod declare_clippy_lint;

#[macro_use]
extern crate clippy_utils;

mod utils;

pub mod ctfe; // Very important lint, do not remove (rust#125116)
pub mod declared_lints;
pub mod deprecated_lints;

// begin lints modules, do not remove this comment, it’s used in `update_lints`
mod absolute_paths;
mod almost_complete_range;
mod approx_const;
mod arbitrary_source_item_ordering;
mod arc_with_non_send_sync;
mod as_conversions;
mod asm_syntax;
mod assertions_on_constants;
mod assertions_on_result_states;
mod assigning_clones;
mod async_yields_async;
mod attrs;
mod await_holding_invalid;
mod blocks_in_conditions;
mod bool_assert_comparison;
mod bool_to_int_with_if;
mod booleans;
mod borrow_deref_ref;
mod box_default;
mod byte_char_slices;
mod cargo;
mod casts;
mod cfg_not_test;
mod checked_conversions;
mod cloned_ref_to_slice_refs;
mod cognitive_complexity;
mod collapsible_if;
mod collection_is_never_read;
mod comparison_chain;
mod copies;
mod copy_iterator;
mod crate_in_macro_def;
mod create_dir;
mod dbg_macro;
mod default;
mod default_constructed_unit_structs;
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
mod empty_line_after;
mod empty_with_brackets;
mod endian_bytes;
mod entry;
mod enum_clike;
mod equatable_if_let;
mod error_impl_error;
mod escape;
mod eta_reduction;
mod excessive_bools;
mod excessive_nesting;
mod exhaustive_items;
mod exit;
mod explicit_write;
mod extra_unused_type_parameters;
mod fallible_impl_from;
mod field_scoped_visibility_modifiers;
mod float_literal;
mod floating_point_arithmetic;
mod format;
mod format_args;
mod format_impl;
mod format_push_string;
mod formatting;
mod four_forward_slashes;
mod from_over_into;
mod from_raw_with_void_ptr;
mod from_str_radix_10;
mod functions;
mod future_not_send;
mod if_let_mutex;
mod if_not_else;
mod if_then_some_else_none;
mod ignored_unit_patterns;
mod impl_hash_with_borrow_str_and_bytes;
mod implicit_hasher;
mod implicit_return;
mod implicit_saturating_add;
mod implicit_saturating_sub;
mod implied_bounds_in_impls;
mod incompatible_msrv;
mod inconsistent_struct_constructor;
mod index_refutable_slice;
mod indexing_slicing;
mod ineffective_open_options;
mod infinite_iter;
mod inherent_impl;
mod inherent_to_string;
mod init_numbered_fields;
mod inline_fn_without_body;
mod instant_subtraction;
mod int_plus_one;
mod integer_division_remainder_used;
mod invalid_upcast_comparisons;
mod item_name_repetitions;
mod items_after_statements;
mod items_after_test_module;
mod iter_not_returning_iterator;
mod iter_over_hash_type;
mod iter_without_into_iter;
mod large_const_arrays;
mod large_enum_variant;
mod large_futures;
mod large_include_file;
mod large_stack_arrays;
mod large_stack_frames;
mod legacy_numeric_constants;
mod len_zero;
mod let_if_seq;
mod let_underscore;
mod let_with_type_underscore;
mod lifetimes;
mod lines_filter_map_ok;
mod literal_representation;
mod literal_string_with_formatting_args;
mod loops;
mod macro_metavars_in_unsafe;
mod macro_use;
mod main_recursion;
mod manual_abs_diff;
mod manual_assert;
mod manual_async_fn;
mod manual_bits;
mod manual_clamp;
mod manual_div_ceil;
mod manual_float_methods;
mod manual_hash_one;
mod manual_ignore_case_cmp;
mod manual_is_ascii_check;
mod manual_is_power_of_two;
mod manual_let_else;
mod manual_main_separator_str;
mod manual_non_exhaustive;
mod manual_option_as_slice;
mod manual_range_patterns;
mod manual_rem_euclid;
mod manual_retain;
mod manual_rotate;
mod manual_slice_size_calculation;
mod manual_string_new;
mod manual_strip;
mod map_unit_fn;
mod match_result_ok;
mod matches;
mod mem_replace;
mod methods;
mod min_ident_chars;
mod minmax;
mod misc;
mod misc_early;
mod mismatching_type_param_order;
mod missing_assert_message;
mod missing_asserts_for_indexing;
mod missing_const_for_fn;
mod missing_const_for_thread_local;
mod missing_doc;
mod missing_enforced_import_rename;
mod missing_fields_in_debug;
mod missing_inline;
mod missing_trait_methods;
mod mixed_read_write_in_expression;
mod module_style;
mod multi_assignments;
mod multiple_bound_locations;
mod multiple_unsafe_ops_per_block;
mod mut_key;
mod mut_mut;
mod mut_reference;
mod mutable_debug_assertion;
mod mutex_atomic;
mod needless_arbitrary_self_type;
mod needless_bool;
mod needless_borrowed_ref;
mod needless_borrows_for_generic_args;
mod needless_continue;
mod needless_else;
mod needless_for_each;
mod needless_if;
mod needless_late_init;
mod needless_maybe_sized;
mod needless_parens_on_range_literals;
mod needless_pass_by_ref_mut;
mod needless_pass_by_value;
mod needless_question_mark;
mod needless_update;
mod neg_cmp_op_on_partial_ord;
mod neg_multiply;
mod new_without_default;
mod no_effect;
mod no_mangle_with_rust_abi;
mod non_canonical_impls;
mod non_copy_const;
mod non_expressive_names;
mod non_octal_unix_permissions;
mod non_send_fields_in_send_ty;
mod non_std_lazy_statics;
mod non_zero_suggestions;
mod nonstandard_macro_braces;
mod octal_escapes;
mod only_used_in_recursion;
mod operators;
mod option_env_unwrap;
mod option_if_let_else;
mod panic_in_result_fn;
mod panic_unimplemented;
mod panicking_overflow_checks;
mod partial_pub_fields;
mod partialeq_ne_impl;
mod partialeq_to_none;
mod pass_by_ref_or_value;
mod pathbuf_init_then_push;
mod pattern_type_mismatch;
mod permissions_set_readonly_false;
mod pointers_in_nomem_asm_block;
mod precedence;
mod ptr;
mod ptr_offset_with_cast;
mod pub_underscore_fields;
mod pub_use;
mod question_mark;
mod question_mark_used;
mod ranges;
mod raw_strings;
mod rc_clone_in_vec_init;
mod read_zero_byte_vec;
mod redundant_async_block;
mod redundant_clone;
mod redundant_closure_call;
mod redundant_else;
mod redundant_field_names;
mod redundant_locals;
mod redundant_pub_crate;
mod redundant_slicing;
mod redundant_static_lifetimes;
mod redundant_test_prefix;
mod redundant_type_annotations;
mod ref_option_ref;
mod ref_patterns;
mod reference;
mod regex;
mod repeat_vec_with_capacity;
mod reserve_after_initialization;
mod return_self_not_must_use;
mod returns;
mod same_name_method;
mod self_named_constructors;
mod semicolon_block;
mod semicolon_if_nothing_returned;
mod serde_api;
mod set_contains_or_insert;
mod shadow;
mod significant_drop_tightening;
mod single_call_fn;
mod single_char_lifetime_names;
mod single_component_path_imports;
mod single_option_map;
mod single_range_in_vec_init;
mod size_of_in_element_count;
mod size_of_ref;
mod slow_vector_initialization;
mod std_instead_of_core;
mod string_patterns;
mod strings;
mod strlen_on_c_strings;
mod suspicious_operation_groupings;
mod suspicious_trait_impl;
mod suspicious_xor_used_as_pow;
mod swap;
mod swap_ptr_to_ref;
mod tabs_in_doc_comments;
mod temporary_assignment;
mod tests_outside_test_module;
mod to_digit_is_some;
mod to_string_trait_impl;
mod trailing_empty_array;
mod trait_bounds;
mod transmute;
mod tuple_array_conversions;
mod types;
mod unconditional_recursion;
mod undocumented_unsafe_blocks;
mod unicode;
mod uninhabited_references;
mod uninit_vec;
mod unit_return_expecting_ord;
mod unit_types;
mod unnecessary_box_returns;
mod unnecessary_literal_bound;
mod unnecessary_map_on_constructor;
mod unnecessary_owned_empty_strings;
mod unnecessary_self_imports;
mod unnecessary_semicolon;
mod unnecessary_struct_initialization;
mod unnecessary_wraps;
mod unneeded_struct_pattern;
mod unnested_or_patterns;
mod unsafe_removed_from_name;
mod unused_async;
mod unused_io_amount;
mod unused_peekable;
mod unused_result_ok;
mod unused_rounding;
mod unused_self;
mod unused_trait_names;
mod unused_unit;
mod unwrap;
mod unwrap_in_result;
mod upper_case_acronyms;
mod use_self;
mod useless_concat;
mod useless_conversion;
mod vec;
mod vec_init_then_push;
mod visibility;
mod wildcard_imports;
mod write;
mod zero_div_zero;
mod zero_repeat_side_effects;
mod zero_sized_map_values;
mod zombie_processes;
// end lints modules, do not remove this comment, it’s used in `update_lints`

use clippy_config::{Conf, get_configuration_metadata, sanitize_explanation};
use clippy_utils::macros::FormatArgsStorage;
use rustc_data_structures::fx::FxHashSet;
use rustc_lint::{Lint, LintId};
use utils::attr_collector::{AttrCollector, AttrStorage};

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
    }
}

#[derive(Copy, Clone, Debug)]
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
        }
    }
}

pub struct LintInfo {
    /// Double reference to maintain pointer equality
    pub lint: &'static &'static Lint,
    category: LintCategory,
    pub explanation: &'static str,
    /// e.g. `clippy_lints/src/absolute_paths.rs#43`
    pub location: &'static str,
    pub version: Option<&'static str>,
}

impl LintInfo {
    /// Returns the lint name in lowercase without the `clippy::` prefix
    #[allow(clippy::missing_panics_doc)]
    pub fn name_lower(&self) -> String {
        self.lint.name.strip_prefix("clippy::").unwrap().to_ascii_lowercase()
    }

    /// Returns the name of the lint's category in lowercase (`style`, `pedantic`)
    pub fn category_str(&self) -> &'static str {
        match self.category {
            Cargo => "cargo",
            Complexity => "complexity",
            Correctness => "correctness",
            Nursery => "nursery",
            Pedantic => "pedantic",
            Perf => "perf",
            Restriction => "restriction",
            Style => "style",
            Suspicious => "suspicious",
        }
    }
}

pub fn explain(name: &str) -> i32 {
    let target = format!("clippy::{}", name.to_ascii_uppercase());

    if let Some(info) = declared_lints::LINTS.iter().find(|info| info.lint.name == target) {
        println!("{}", sanitize_explanation(info.explanation));
        // Check if the lint has configuration
        let mut mdconf = get_configuration_metadata();
        let name = name.to_ascii_lowercase();
        mdconf.retain(|cconf| cconf.lints.contains(&&*name));
        if !mdconf.is_empty() {
            println!("### Configuration for {}:\n", info.lint.name_lower());
            for conf in mdconf {
                println!("{conf}");
            }
        }
        0
    } else {
        println!("unknown lint: {name}");
        1
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

/// Register all lints and lint groups with the rustc lint store
///
/// Used in `./src/driver.rs`.
#[expect(clippy::too_many_lines)]
pub fn register_lints(store: &mut rustc_lint::LintStore, conf: &'static Conf) {
    register_categories(store);

    for (old_name, new_name) in deprecated_lints::RENAMED {
        store.register_renamed(old_name, new_name);
    }
    for (name, reason) in deprecated_lints::DEPRECATED {
        store.register_removed(name, reason);
    }

    // NOTE: Do not add any more pre-expansion passes. These should be removed eventually.
    // Due to the architecture of the compiler, currently `cfg_attr` attributes on crate
    // level (i.e `#![cfg_attr(...)]`) will still be expanded even when using a pre-expansion pass.
    store.register_pre_expansion_pass(move || Box::new(attrs::EarlyAttributes::new(conf)));

    store.register_early_pass(move || Box::new(attrs::PostExpansionEarlyAttributes::new(conf)));

    let format_args_storage = FormatArgsStorage::default();
    let format_args = format_args_storage.clone();
    store.register_early_pass(move || {
        Box::new(utils::format_args_collector::FormatArgsCollector::new(
            format_args.clone(),
        ))
    });

    let attr_storage = AttrStorage::default();
    let attrs = attr_storage.clone();
    store.register_early_pass(move || Box::new(AttrCollector::new(attrs.clone())));

    store.register_late_pass(|_| Box::new(ctfe::ClippyCtfe));

    store.register_late_pass(move |_| Box::new(operators::arithmetic_side_effects::ArithmeticSideEffects::new(conf)));
    store.register_late_pass(|_| Box::new(utils::dump_hir::DumpHir));
    store.register_late_pass(|_| Box::new(utils::author::Author));
    store.register_late_pass(move |tcx| Box::new(await_holding_invalid::AwaitHolding::new(tcx, conf)));
    store.register_late_pass(|_| Box::new(serde_api::SerdeApi));
    store.register_late_pass(move |_| Box::new(types::Types::new(conf)));
    store.register_late_pass(move |_| Box::new(booleans::NonminimalBool::new(conf)));
    store.register_late_pass(|_| Box::new(enum_clike::UnportableVariant));
    store.register_late_pass(|_| Box::new(float_literal::FloatLiteral));
    store.register_late_pass(|_| Box::new(ptr::Ptr));
    store.register_late_pass(|_| Box::new(needless_bool::NeedlessBool));
    store.register_late_pass(|_| Box::new(needless_bool::BoolComparison));
    store.register_late_pass(|_| Box::new(needless_for_each::NeedlessForEach));
    store.register_late_pass(|_| Box::new(misc::LintPass));
    store.register_late_pass(|_| Box::new(eta_reduction::EtaReduction));
    store.register_late_pass(|_| Box::new(mut_mut::MutMut));
    store.register_late_pass(|_| Box::new(mut_reference::UnnecessaryMutPassed));
    store.register_late_pass(|_| Box::<significant_drop_tightening::SignificantDropTightening<'_>>::default());
    store.register_late_pass(|_| Box::new(len_zero::LenZero));
    store.register_late_pass(move |_| Box::new(attrs::Attributes::new(conf)));
    store.register_late_pass(|_| Box::new(blocks_in_conditions::BlocksInConditions));
    store.register_late_pass(|_| Box::new(unicode::Unicode));
    store.register_late_pass(|_| Box::new(uninit_vec::UninitVec));
    store.register_late_pass(|_| Box::new(unit_return_expecting_ord::UnitReturnExpectingOrd));
    store.register_late_pass(|_| Box::new(strings::StringAdd));
    store.register_late_pass(|_| Box::new(implicit_return::ImplicitReturn));
    store.register_late_pass(move |_| Box::new(implicit_saturating_sub::ImplicitSaturatingSub::new(conf)));
    store.register_late_pass(|_| Box::new(default_numeric_fallback::DefaultNumericFallback));
    store.register_late_pass(move |_| {
        Box::new(inconsistent_struct_constructor::InconsistentStructConstructor::new(
            conf,
        ))
    });
    store.register_late_pass(|_| Box::new(non_octal_unix_permissions::NonOctalUnixPermissions));
    store.register_early_pass(|| Box::new(unnecessary_self_imports::UnnecessarySelfImports));
    store.register_late_pass(move |_| Box::new(approx_const::ApproxConstant::new(conf)));
    let format_args = format_args_storage.clone();
    store.register_late_pass(move |_| Box::new(methods::Methods::new(conf, format_args.clone())));
    store.register_late_pass(move |_| Box::new(matches::Matches::new(conf)));
    store.register_late_pass(move |_| Box::new(manual_non_exhaustive::ManualNonExhaustive::new(conf)));
    store.register_late_pass(move |_| Box::new(manual_strip::ManualStrip::new(conf)));
    store.register_early_pass(move || Box::new(redundant_static_lifetimes::RedundantStaticLifetimes::new(conf)));
    store.register_early_pass(move || Box::new(redundant_field_names::RedundantFieldNames::new(conf)));
    store.register_late_pass(move |_| Box::new(checked_conversions::CheckedConversions::new(conf)));
    store.register_late_pass(move |_| Box::new(mem_replace::MemReplace::new(conf)));
    store.register_late_pass(move |_| Box::new(ranges::Ranges::new(conf)));
    store.register_late_pass(move |_| Box::new(from_over_into::FromOverInto::new(conf)));
    store.register_late_pass(move |_| Box::new(use_self::UseSelf::new(conf)));
    store.register_late_pass(move |_| Box::new(missing_const_for_fn::MissingConstForFn::new(conf)));
    store.register_late_pass(move |_| Box::new(needless_question_mark::NeedlessQuestionMark));
    store.register_late_pass(move |_| Box::new(casts::Casts::new(conf)));
    store.register_early_pass(move || Box::new(unnested_or_patterns::UnnestedOrPatterns::new(conf)));
    store.register_late_pass(|_| Box::new(size_of_in_element_count::SizeOfInElementCount));
    store.register_late_pass(|_| Box::new(same_name_method::SameNameMethod));
    store.register_late_pass(move |_| Box::new(index_refutable_slice::IndexRefutableSlice::new(conf)));
    store.register_late_pass(|_| Box::<shadow::Shadow>::default());
    store.register_late_pass(|_| Box::new(unit_types::UnitTypes));
    store.register_late_pass(move |_| Box::new(loops::Loops::new(conf)));
    store.register_late_pass(|_| Box::<main_recursion::MainRecursion>::default());
    store.register_late_pass(move |_| Box::new(lifetimes::Lifetimes::new(conf)));
    store.register_late_pass(|_| Box::new(entry::HashMapPass));
    store.register_late_pass(|_| Box::new(minmax::MinMaxPass));
    store.register_late_pass(|_| Box::new(zero_div_zero::ZeroDiv));
    store.register_late_pass(|_| Box::new(mutex_atomic::Mutex));
    store.register_late_pass(|_| Box::new(needless_update::NeedlessUpdate));
    store.register_late_pass(|_| Box::new(needless_borrowed_ref::NeedlessBorrowedRef));
    store.register_late_pass(|_| Box::new(borrow_deref_ref::BorrowDerefRef));
    store.register_late_pass(|_| Box::<no_effect::NoEffect>::default());
    store.register_late_pass(|_| Box::new(temporary_assignment::TemporaryAssignment));
    store.register_late_pass(move |_| Box::new(transmute::Transmute::new(conf)));
    store.register_late_pass(move |_| Box::new(cognitive_complexity::CognitiveComplexity::new(conf)));
    store.register_late_pass(move |_| Box::new(escape::BoxedLocal::new(conf)));
    store.register_late_pass(move |_| Box::new(vec::UselessVec::new(conf)));
    store.register_late_pass(move |_| Box::new(panic_unimplemented::PanicUnimplemented::new(conf)));
    store.register_late_pass(|_| Box::new(strings::StringLitAsBytes));
    store.register_late_pass(|_| Box::new(derive::Derive));
    store.register_late_pass(move |_| Box::new(derivable_impls::DerivableImpls::new(conf)));
    store.register_late_pass(|_| Box::new(drop_forget_ref::DropForgetRef));
    store.register_late_pass(|_| Box::new(empty_enum::EmptyEnum));
    store.register_late_pass(|_| Box::new(invalid_upcast_comparisons::InvalidUpcastComparisons));
    store.register_late_pass(|_| Box::<regex::Regex>::default());
    store.register_late_pass(move |tcx| Box::new(copies::CopyAndPaste::new(tcx, conf)));
    store.register_late_pass(|_| Box::new(copy_iterator::CopyIterator));
    let format_args = format_args_storage.clone();
    store.register_late_pass(move |_| Box::new(format::UselessFormat::new(format_args.clone())));
    store.register_late_pass(|_| Box::new(swap::Swap));
    store.register_late_pass(|_| Box::new(panicking_overflow_checks::PanickingOverflowChecks));
    store.register_late_pass(|_| Box::<new_without_default::NewWithoutDefault>::default());
    store.register_late_pass(move |_| Box::new(disallowed_names::DisallowedNames::new(conf)));
    store.register_late_pass(move |tcx| Box::new(functions::Functions::new(tcx, conf)));
    store.register_late_pass(move |_| Box::new(doc::Documentation::new(conf)));
    store.register_early_pass(move || Box::new(doc::Documentation::new(conf)));
    store.register_late_pass(|_| Box::new(neg_multiply::NegMultiply));
    store.register_late_pass(|_| Box::new(let_if_seq::LetIfSeq));
    store.register_late_pass(|_| Box::new(mixed_read_write_in_expression::EvalOrderDependence));
    store.register_late_pass(move |_| Box::new(missing_doc::MissingDoc::new(conf)));
    store.register_late_pass(|_| Box::new(missing_inline::MissingInline));
    store.register_late_pass(move |_| Box::new(exhaustive_items::ExhaustiveItems));
    store.register_late_pass(|_| Box::new(unused_result_ok::UnusedResultOk));
    store.register_late_pass(|_| Box::new(match_result_ok::MatchResultOk));
    store.register_late_pass(|_| Box::new(partialeq_ne_impl::PartialEqNeImpl));
    store.register_late_pass(|_| Box::new(unused_io_amount::UnusedIoAmount));
    store.register_late_pass(move |_| Box::new(large_enum_variant::LargeEnumVariant::new(conf)));
    let format_args = format_args_storage.clone();
    store.register_late_pass(move |_| Box::new(explicit_write::ExplicitWrite::new(format_args.clone())));
    store.register_late_pass(|_| Box::new(needless_pass_by_value::NeedlessPassByValue));
    store.register_late_pass(move |tcx| Box::new(pass_by_ref_or_value::PassByRefOrValue::new(tcx, conf)));
    store.register_late_pass(|_| Box::new(ref_option_ref::RefOptionRef));
    store.register_late_pass(|_| Box::new(infinite_iter::InfiniteIter));
    store.register_late_pass(|_| Box::new(inline_fn_without_body::InlineFnWithoutBody));
    store.register_late_pass(|_| Box::<useless_conversion::UselessConversion>::default());
    store.register_late_pass(|_| Box::new(implicit_hasher::ImplicitHasher));
    store.register_late_pass(|_| Box::new(fallible_impl_from::FallibleImplFrom));
    store.register_late_pass(move |_| Box::new(question_mark::QuestionMark::new(conf)));
    store.register_late_pass(|_| Box::new(question_mark_used::QuestionMarkUsed));
    store.register_early_pass(|| Box::new(suspicious_operation_groupings::SuspiciousOperationGroupings));
    store.register_late_pass(|_| Box::new(suspicious_trait_impl::SuspiciousImpl));
    store.register_late_pass(|_| Box::new(map_unit_fn::MapUnit));
    store.register_late_pass(|_| Box::new(inherent_impl::MultipleInherentImpl));
    store.register_late_pass(|_| Box::new(neg_cmp_op_on_partial_ord::NoNegCompOpForPartialOrd));
    store.register_late_pass(|_| Box::new(unwrap::Unwrap));
    store.register_late_pass(move |_| Box::new(indexing_slicing::IndexingSlicing::new(conf)));
    store.register_late_pass(move |tcx| Box::new(non_copy_const::NonCopyConst::new(tcx, conf)));
    store.register_late_pass(|_| Box::new(ptr_offset_with_cast::PtrOffsetWithCast));
    store.register_late_pass(|_| Box::new(redundant_clone::RedundantClone));
    store.register_late_pass(|_| Box::new(slow_vector_initialization::SlowVectorInit));
    store.register_late_pass(move |_| Box::new(unnecessary_wraps::UnnecessaryWraps::new(conf)));
    store.register_late_pass(|_| Box::new(assertions_on_constants::AssertionsOnConstants));
    store.register_late_pass(|_| Box::new(assertions_on_result_states::AssertionsOnResultStates));
    store.register_late_pass(|_| Box::new(inherent_to_string::InherentToString));
    store.register_late_pass(move |_| Box::new(trait_bounds::TraitBounds::new(conf)));
    store.register_late_pass(|_| Box::new(comparison_chain::ComparisonChain));
    store.register_late_pass(move |tcx| Box::new(mut_key::MutableKeyType::new(tcx, conf)));
    store.register_early_pass(|| Box::new(reference::DerefAddrOf));
    store.register_early_pass(|| Box::new(double_parens::DoubleParens));
    let format_args = format_args_storage.clone();
    store.register_late_pass(move |_| Box::new(format_impl::FormatImpl::new(format_args.clone())));
    store.register_early_pass(|| Box::new(unsafe_removed_from_name::UnsafeNameRemoval));
    store.register_early_pass(|| Box::new(else_if_without_else::ElseIfWithoutElse));
    store.register_early_pass(|| Box::new(int_plus_one::IntPlusOne));
    store.register_early_pass(|| Box::new(formatting::Formatting));
    store.register_early_pass(|| Box::new(misc_early::MiscEarlyLints));
    store.register_late_pass(|_| Box::new(redundant_closure_call::RedundantClosureCall));
    store.register_early_pass(|| Box::new(unused_unit::UnusedUnit));
    store.register_late_pass(|_| Box::new(unused_unit::UnusedUnit));
    store.register_late_pass(|_| Box::new(returns::Return));
    store.register_late_pass(move |_| Box::new(collapsible_if::CollapsibleIf::new(conf)));
    store.register_late_pass(|_| Box::new(items_after_statements::ItemsAfterStatements));
    store.register_early_pass(|| Box::new(precedence::Precedence));
    store.register_late_pass(|_| Box::new(needless_parens_on_range_literals::NeedlessParensOnRangeLiterals));
    store.register_early_pass(|| Box::new(needless_continue::NeedlessContinue));
    store.register_early_pass(|| Box::new(redundant_else::RedundantElse));
    store.register_late_pass(|_| Box::new(create_dir::CreateDir));
    store.register_early_pass(|| Box::new(needless_arbitrary_self_type::NeedlessArbitrarySelfType));
    store.register_early_pass(move || Box::new(literal_representation::LiteralDigitGrouping::new(conf)));
    store.register_early_pass(move || Box::new(literal_representation::DecimalLiteralRepresentation::new(conf)));
    store.register_late_pass(move |_| Box::new(item_name_repetitions::ItemNameRepetitions::new(conf)));
    store.register_early_pass(|| Box::new(tabs_in_doc_comments::TabsInDocComments));
    store.register_late_pass(move |_| Box::new(upper_case_acronyms::UpperCaseAcronyms::new(conf)));
    store.register_late_pass(|_| Box::<default::Default>::default());
    store.register_late_pass(move |_| Box::new(unused_self::UnusedSelf::new(conf)));
    store.register_late_pass(|_| Box::new(mutable_debug_assertion::DebugAssertWithMutCall));
    store.register_late_pass(|_| Box::new(exit::Exit));
    store.register_late_pass(move |_| Box::new(to_digit_is_some::ToDigitIsSome::new(conf)));
    store.register_late_pass(move |_| Box::new(large_stack_arrays::LargeStackArrays::new(conf)));
    store.register_late_pass(move |_| Box::new(large_const_arrays::LargeConstArrays::new(conf)));
    store.register_late_pass(|_| Box::new(floating_point_arithmetic::FloatingPointArithmetic));
    store.register_late_pass(|_| Box::new(as_conversions::AsConversions));
    store.register_late_pass(|_| Box::new(let_underscore::LetUnderscore));
    store.register_early_pass(|| Box::<single_component_path_imports::SingleComponentPathImports>::default());
    store.register_late_pass(move |_| Box::new(excessive_bools::ExcessiveBools::new(conf)));
    store.register_early_pass(|| Box::new(option_env_unwrap::OptionEnvUnwrap));
    store.register_late_pass(move |_| Box::new(wildcard_imports::WildcardImports::new(conf)));
    store.register_late_pass(|_| Box::<redundant_pub_crate::RedundantPubCrate>::default());
    store.register_late_pass(|_| Box::<dereference::Dereferencing<'_>>::default());
    store.register_late_pass(|_| Box::new(option_if_let_else::OptionIfLetElse));
    store.register_late_pass(|_| Box::new(future_not_send::FutureNotSend));
    store.register_late_pass(move |_| Box::new(large_futures::LargeFuture::new(conf)));
    store.register_late_pass(|_| Box::new(if_let_mutex::IfLetMutex));
    store.register_late_pass(|_| Box::new(if_not_else::IfNotElse));
    store.register_late_pass(|_| Box::new(equatable_if_let::PatternEquality));
    store.register_late_pass(|_| Box::new(manual_async_fn::ManualAsyncFn));
    store.register_late_pass(|_| Box::new(panic_in_result_fn::PanicInResultFn));
    store.register_early_pass(move || Box::new(non_expressive_names::NonExpressiveNames::new(conf)));
    store.register_early_pass(move || Box::new(nonstandard_macro_braces::MacroBraces::new(conf)));
    store.register_late_pass(|_| Box::<macro_use::MacroUseImports>::default());
    store.register_late_pass(|_| Box::new(pattern_type_mismatch::PatternTypeMismatch));
    store.register_late_pass(|_| Box::new(unwrap_in_result::UnwrapInResult));
    store.register_late_pass(|_| Box::new(semicolon_if_nothing_returned::SemicolonIfNothingReturned));
    store.register_late_pass(|_| Box::new(async_yields_async::AsyncYieldsAsync));
    let attrs = attr_storage.clone();
    store.register_late_pass(move |tcx| Box::new(disallowed_macros::DisallowedMacros::new(tcx, conf, attrs.clone())));
    store.register_late_pass(move |tcx| Box::new(disallowed_methods::DisallowedMethods::new(tcx, conf)));
    store.register_early_pass(|| Box::new(asm_syntax::InlineAsmX86AttSyntax));
    store.register_early_pass(|| Box::new(asm_syntax::InlineAsmX86IntelSyntax));
    store.register_late_pass(|_| Box::new(empty_drop::EmptyDrop));
    store.register_late_pass(|_| Box::new(strings::StrToString));
    store.register_late_pass(|_| Box::new(strings::StringToString));
    store.register_late_pass(|_| Box::new(zero_sized_map_values::ZeroSizedMapValues));
    store.register_late_pass(|_| Box::<vec_init_then_push::VecInitThenPush>::default());
    store.register_late_pass(|_| Box::new(redundant_slicing::RedundantSlicing));
    store.register_late_pass(|_| Box::new(from_str_radix_10::FromStrRadix10));
    store.register_late_pass(move |_| Box::new(if_then_some_else_none::IfThenSomeElseNone::new(conf)));
    store.register_late_pass(|_| Box::new(bool_assert_comparison::BoolAssertComparison));
    store.register_early_pass(move || Box::new(module_style::ModStyle));
    store.register_late_pass(|_| Box::<unused_async::UnusedAsync>::default());
    store.register_late_pass(move |tcx| Box::new(disallowed_types::DisallowedTypes::new(tcx, conf)));
    store.register_late_pass(move |tcx| Box::new(missing_enforced_import_rename::ImportRename::new(tcx, conf)));
    store.register_early_pass(move || Box::new(disallowed_script_idents::DisallowedScriptIdents::new(conf)));
    store.register_late_pass(|_| Box::new(strlen_on_c_strings::StrlenOnCStrings));
    store.register_late_pass(move |_| Box::new(self_named_constructors::SelfNamedConstructors));
    store.register_late_pass(move |_| Box::new(iter_not_returning_iterator::IterNotReturningIterator));
    store.register_late_pass(move |_| Box::new(manual_assert::ManualAssert));
    store.register_late_pass(move |_| Box::new(non_send_fields_in_send_ty::NonSendFieldInSendTy::new(conf)));
    store.register_late_pass(move |_| Box::new(undocumented_unsafe_blocks::UndocumentedUnsafeBlocks::new(conf)));
    let format_args = format_args_storage.clone();
    store.register_late_pass(move |tcx| Box::new(format_args::FormatArgs::new(tcx, conf, format_args.clone())));
    store.register_late_pass(|_| Box::new(trailing_empty_array::TrailingEmptyArray));
    store.register_early_pass(|| Box::new(octal_escapes::OctalEscapes));
    store.register_late_pass(|_| Box::new(needless_late_init::NeedlessLateInit));
    store.register_late_pass(|_| Box::new(return_self_not_must_use::ReturnSelfNotMustUse));
    store.register_late_pass(|_| Box::new(init_numbered_fields::NumberedFields));
    store.register_early_pass(|| Box::new(single_char_lifetime_names::SingleCharLifetimeNames));
    store.register_late_pass(move |_| Box::new(manual_bits::ManualBits::new(conf)));
    store.register_late_pass(|_| Box::new(default_union_representation::DefaultUnionRepresentation));
    store.register_late_pass(|_| Box::<only_used_in_recursion::OnlyUsedInRecursion>::default());
    store.register_late_pass(move |_| Box::new(dbg_macro::DbgMacro::new(conf)));
    let format_args = format_args_storage.clone();
    store.register_late_pass(move |_| Box::new(write::Write::new(conf, format_args.clone())));
    store.register_late_pass(move |_| Box::new(cargo::Cargo::new(conf)));
    store.register_early_pass(|| Box::new(crate_in_macro_def::CrateInMacroDef));
    store.register_late_pass(|_| Box::new(empty_with_brackets::EmptyWithBrackets::default()));
    store.register_late_pass(|_| Box::new(unnecessary_owned_empty_strings::UnnecessaryOwnedEmptyStrings));
    store.register_early_pass(|| Box::new(pub_use::PubUse));
    store.register_late_pass(|_| Box::new(format_push_string::FormatPushString));
    store.register_late_pass(move |_| Box::new(large_include_file::LargeIncludeFile::new(conf)));
    store.register_early_pass(move || Box::new(large_include_file::LargeIncludeFile::new(conf)));
    store.register_late_pass(|_| Box::new(strings::TrimSplitWhitespace));
    store.register_late_pass(|_| Box::new(rc_clone_in_vec_init::RcCloneInVecInit));
    store.register_early_pass(|| Box::<duplicate_mod::DuplicateMod>::default());
    store.register_early_pass(|| Box::new(unused_rounding::UnusedRounding));
    store.register_early_pass(move || Box::new(almost_complete_range::AlmostCompleteRange::new(conf)));
    store.register_late_pass(|_| Box::new(swap_ptr_to_ref::SwapPtrToRef));
    store.register_late_pass(|_| Box::new(mismatching_type_param_order::TypeParamMismatch));
    store.register_late_pass(|_| Box::new(read_zero_byte_vec::ReadZeroByteVec));
    store.register_late_pass(|_| Box::new(default_instead_of_iter_empty::DefaultIterEmpty));
    store.register_late_pass(move |_| Box::new(manual_rem_euclid::ManualRemEuclid::new(conf)));
    store.register_late_pass(move |_| Box::new(manual_retain::ManualRetain::new(conf)));
    store.register_late_pass(move |_| Box::new(manual_rotate::ManualRotate));
    store.register_late_pass(move |_| Box::new(operators::Operators::new(conf)));
    store.register_late_pass(move |_| Box::new(std_instead_of_core::StdReexports::new(conf)));
    store.register_late_pass(move |_| Box::new(instant_subtraction::InstantSubtraction::new(conf)));
    store.register_late_pass(|_| Box::new(partialeq_to_none::PartialeqToNone));
    store.register_late_pass(move |_| Box::new(manual_abs_diff::ManualAbsDiff::new(conf)));
    store.register_late_pass(move |_| Box::new(manual_clamp::ManualClamp::new(conf)));
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
    store.register_late_pass(move |_| Box::new(manual_is_ascii_check::ManualIsAsciiCheck::new(conf)));
    store.register_late_pass(move |_| Box::new(semicolon_block::SemicolonBlock::new(conf)));
    store.register_late_pass(|_| Box::new(permissions_set_readonly_false::PermissionsSetReadonlyFalse));
    store.register_late_pass(|_| Box::new(size_of_ref::SizeOfRef));
    store.register_late_pass(|_| Box::new(multiple_unsafe_ops_per_block::MultipleUnsafeOpsPerBlock));
    store.register_late_pass(move |_| Box::new(extra_unused_type_parameters::ExtraUnusedTypeParameters::new(conf)));
    store.register_late_pass(|_| Box::new(no_mangle_with_rust_abi::NoMangleWithRustAbi));
    store.register_late_pass(|_| Box::new(collection_is_never_read::CollectionIsNeverRead));
    store.register_late_pass(|_| Box::new(missing_assert_message::MissingAssertMessage));
    store.register_late_pass(|_| Box::new(needless_maybe_sized::NeedlessMaybeSized));
    store.register_late_pass(|_| Box::new(redundant_async_block::RedundantAsyncBlock));
    store.register_late_pass(|_| Box::new(let_with_type_underscore::UnderscoreTyped));
    store.register_late_pass(move |_| Box::new(manual_main_separator_str::ManualMainSeparatorStr::new(conf)));
    store.register_late_pass(|_| Box::new(unnecessary_struct_initialization::UnnecessaryStruct));
    store.register_late_pass(move |_| Box::new(unnecessary_box_returns::UnnecessaryBoxReturns::new(conf)));
    store.register_late_pass(move |_| Box::new(lines_filter_map_ok::LinesFilterMapOk::new(conf)));
    store.register_late_pass(|_| Box::new(tests_outside_test_module::TestsOutsideTestModule));
    store.register_late_pass(|_| Box::new(manual_slice_size_calculation::ManualSliceSizeCalculation::new(conf)));
    store.register_early_pass(move || Box::new(excessive_nesting::ExcessiveNesting::new(conf)));
    store.register_late_pass(|_| Box::new(items_after_test_module::ItemsAfterTestModule));
    store.register_early_pass(|| Box::new(ref_patterns::RefPatterns));
    store.register_late_pass(|_| Box::new(default_constructed_unit_structs::DefaultConstructedUnitStructs));
    store.register_early_pass(|| Box::new(needless_else::NeedlessElse));
    store.register_late_pass(|_| Box::new(missing_fields_in_debug::MissingFieldsInDebug));
    store.register_late_pass(|_| Box::new(endian_bytes::EndianBytes));
    store.register_late_pass(|_| Box::new(redundant_type_annotations::RedundantTypeAnnotations));
    store.register_late_pass(|_| Box::new(arc_with_non_send_sync::ArcWithNonSendSync));
    store.register_late_pass(|_| Box::new(needless_if::NeedlessIf));
    store.register_late_pass(move |_| Box::new(min_ident_chars::MinIdentChars::new(conf)));
    store.register_late_pass(move |_| Box::new(large_stack_frames::LargeStackFrames::new(conf)));
    store.register_late_pass(|_| Box::new(single_range_in_vec_init::SingleRangeInVecInit));
    store.register_late_pass(move |_| Box::new(needless_pass_by_ref_mut::NeedlessPassByRefMut::new(conf)));
    store.register_late_pass(|_| Box::new(non_canonical_impls::NonCanonicalImpls));
    store.register_late_pass(move |_| Box::new(single_call_fn::SingleCallFn::new(conf)));
    store.register_early_pass(move || Box::new(raw_strings::RawStrings::new(conf)));
    store.register_late_pass(move |_| Box::new(legacy_numeric_constants::LegacyNumericConstants::new(conf)));
    store.register_late_pass(|_| Box::new(manual_range_patterns::ManualRangePatterns));
    store.register_early_pass(|| Box::new(visibility::Visibility));
    store.register_late_pass(move |_| Box::new(tuple_array_conversions::TupleArrayConversions::new(conf)));
    store.register_late_pass(move |_| Box::new(manual_float_methods::ManualFloatMethods::new(conf)));
    store.register_late_pass(|_| Box::new(four_forward_slashes::FourForwardSlashes));
    store.register_late_pass(|_| Box::new(error_impl_error::ErrorImplError));
    store.register_late_pass(move |_| Box::new(absolute_paths::AbsolutePaths::new(conf)));
    store.register_late_pass(|_| Box::new(redundant_locals::RedundantLocals));
    store.register_late_pass(|_| Box::new(ignored_unit_patterns::IgnoredUnitPatterns));
    store.register_late_pass(|_| Box::<reserve_after_initialization::ReserveAfterInitialization>::default());
    store.register_late_pass(|_| Box::new(implied_bounds_in_impls::ImpliedBoundsInImpls));
    store.register_late_pass(|_| Box::new(missing_asserts_for_indexing::MissingAssertsForIndexing));
    store.register_late_pass(|_| Box::new(unnecessary_map_on_constructor::UnnecessaryMapOnConstructor));
    store.register_late_pass(move |_| {
        Box::new(needless_borrows_for_generic_args::NeedlessBorrowsForGenericArgs::new(
            conf,
        ))
    });
    store.register_late_pass(move |_| Box::new(manual_hash_one::ManualHashOne::new(conf)));
    store.register_late_pass(|_| Box::new(iter_without_into_iter::IterWithoutIntoIter));
    store.register_late_pass(|_| Box::<pathbuf_init_then_push::PathbufThenPush<'_>>::default());
    store.register_late_pass(|_| Box::new(iter_over_hash_type::IterOverHashType));
    store.register_late_pass(|_| Box::new(impl_hash_with_borrow_str_and_bytes::ImplHashWithBorrowStrBytes));
    store.register_late_pass(move |_| Box::new(repeat_vec_with_capacity::RepeatVecWithCapacity::new(conf)));
    store.register_late_pass(|_| Box::new(uninhabited_references::UninhabitedReferences));
    store.register_late_pass(|_| Box::new(ineffective_open_options::IneffectiveOpenOptions));
    store.register_late_pass(|_| Box::<unconditional_recursion::UnconditionalRecursion>::default());
    store.register_late_pass(move |_| Box::new(pub_underscore_fields::PubUnderscoreFields::new(conf)));
    store.register_late_pass(move |_| Box::new(missing_const_for_thread_local::MissingConstForThreadLocal::new(conf)));
    store.register_late_pass(move |_| Box::new(incompatible_msrv::IncompatibleMsrv::new(conf)));
    store.register_late_pass(|_| Box::new(to_string_trait_impl::ToStringTraitImpl));
    store.register_early_pass(|| Box::new(multiple_bound_locations::MultipleBoundLocations));
    store.register_late_pass(move |_| Box::new(assigning_clones::AssigningClones::new(conf)));
    store.register_late_pass(|_| Box::new(zero_repeat_side_effects::ZeroRepeatSideEffects));
    store.register_late_pass(|_| Box::new(integer_division_remainder_used::IntegerDivisionRemainderUsed));
    store.register_late_pass(move |_| Box::new(macro_metavars_in_unsafe::ExprMetavarsInUnsafe::new(conf)));
    store.register_late_pass(move |_| Box::new(string_patterns::StringPatterns::new(conf)));
    store.register_early_pass(|| Box::new(field_scoped_visibility_modifiers::FieldScopedVisibilityModifiers));
    store.register_late_pass(|_| Box::new(set_contains_or_insert::SetContainsOrInsert));
    store.register_early_pass(|| Box::new(byte_char_slices::ByteCharSlice));
    store.register_early_pass(|| Box::new(cfg_not_test::CfgNotTest));
    store.register_late_pass(|_| Box::new(zombie_processes::ZombieProcesses));
    store.register_late_pass(|_| Box::new(pointers_in_nomem_asm_block::PointersInNomemAsmBlock));
    store.register_late_pass(move |_| Box::new(manual_div_ceil::ManualDivCeil::new(conf)));
    store.register_late_pass(move |_| Box::new(manual_is_power_of_two::ManualIsPowerOfTwo::new(conf)));
    store.register_late_pass(|_| Box::new(non_zero_suggestions::NonZeroSuggestions));
    store.register_late_pass(|_| Box::new(literal_string_with_formatting_args::LiteralStringWithFormattingArg));
    store.register_late_pass(move |_| Box::new(unused_trait_names::UnusedTraitNames::new(conf)));
    store.register_late_pass(|_| Box::new(manual_ignore_case_cmp::ManualIgnoreCaseCmp));
    store.register_late_pass(|_| Box::new(unnecessary_literal_bound::UnnecessaryLiteralBound));
    store.register_early_pass(|| Box::new(empty_line_after::EmptyLineAfter::new()));
    store.register_late_pass(move |_| Box::new(arbitrary_source_item_ordering::ArbitrarySourceItemOrdering::new(conf)));
    store.register_late_pass(|_| Box::new(useless_concat::UselessConcat));
    store.register_late_pass(|_| Box::new(unneeded_struct_pattern::UnneededStructPattern));
    store.register_late_pass(|_| Box::<unnecessary_semicolon::UnnecessarySemicolon>::default());
    store.register_late_pass(move |_| Box::new(non_std_lazy_statics::NonStdLazyStatic::new(conf)));
    store.register_late_pass(|_| Box::new(manual_option_as_slice::ManualOptionAsSlice::new(conf)));
    store.register_late_pass(|_| Box::new(single_option_map::SingleOptionMap));
    store.register_late_pass(move |_| Box::new(redundant_test_prefix::RedundantTestPrefix));
    store.register_late_pass(|_| Box::new(cloned_ref_to_slice_refs::ClonedRefToSliceRefs::new(conf)));
    // add lints here, do not remove this comment, it's used in `new_lint`
}
