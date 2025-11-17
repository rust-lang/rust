#![feature(array_windows)]
#![feature(box_patterns)]
#![feature(macro_metavar_expr_concat)]
#![feature(f128)]
#![feature(f16)]
#![feature(if_let_guard)]
#![feature(iter_intersperse)]
#![feature(iter_partition_in_place)]
#![feature(never_type)]
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

extern crate rustc_abi;
extern crate rustc_arena;
extern crate rustc_ast;
extern crate rustc_ast_pretty;
extern crate rustc_data_structures;
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
extern crate rustc_parse_format;
extern crate rustc_resolve;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_target;
extern crate rustc_trait_selection;

#[macro_use]
extern crate clippy_utils;

#[macro_use]
extern crate declare_clippy_lint;

mod utils;

pub mod declared_lints;
pub mod deprecated_lints;

// begin lints modules, do not remove this comment, it's used in `update_lints`
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
mod bool_comparison;
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
mod coerce_container_to_any;
mod cognitive_complexity;
mod collapsible_if;
mod collection_is_never_read;
mod comparison_chain;
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
mod empty_enums;
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
mod ifs;
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
mod infallible_try_from;
mod infinite_iter;
mod inherent_impl;
mod inherent_to_string;
mod init_numbered_fields;
mod inline_fn_without_body;
mod int_plus_one;
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
mod mutable_debug_assertion;
mod mutex_atomic;
mod needless_arbitrary_self_type;
mod needless_bool;
mod needless_borrowed_ref;
mod needless_borrows_for_generic_args;
mod needless_continue;
mod needless_else;
mod needless_for_each;
mod needless_ifs;
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
mod replace_box;
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
mod time_subtraction;
mod to_digit_is_some;
mod to_string_trait_impl;
mod toplevel_ref_arg;
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
mod unnecessary_mut_passed;
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
mod volatile_composites;
mod wildcard_imports;
mod write;
mod zero_div_zero;
mod zero_repeat_side_effects;
mod zero_sized_map_values;
mod zombie_processes;
// end lints modules, do not remove this comment, it's used in `update_lints`

use clippy_config::{Conf, get_configuration_metadata, sanitize_explanation};
use clippy_utils::macros::FormatArgsStorage;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sync;
use rustc_lint::{EarlyLintPass, LateLintPass, Lint};
use rustc_middle::ty::TyCtxt;
use utils::attr_collector::{AttrCollector, AttrStorage};

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

/// Register all lints and lint groups with the rustc lint store
///
/// Used in `./src/driver.rs`.
#[expect(clippy::too_many_lines)]
pub fn register_lint_passes(store: &mut rustc_lint::LintStore, conf: &'static Conf) {
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

    let format_args_storage = FormatArgsStorage::default();
    let attr_storage = AttrStorage::default();

    let early_lints: [Box<dyn Fn() -> Box<dyn EarlyLintPass + 'static> + sync::DynSend + sync::DynSync>; _] = [
        {
            let format_args = format_args_storage.clone();
            Box::new(move || {
                Box::new(utils::format_args_collector::FormatArgsCollector::new(
                    format_args.clone(),
                ))
            })
        },
        {
            let attrs = attr_storage.clone();
            Box::new(move || Box::new(AttrCollector::new(attrs.clone())))
        },
        Box::new(move || Box::new(attrs::PostExpansionEarlyAttributes::new(conf))),
        Box::new(|| Box::new(unnecessary_self_imports::UnnecessarySelfImports)),
        Box::new(move || Box::new(redundant_static_lifetimes::RedundantStaticLifetimes::new(conf))),
        Box::new(move || Box::new(redundant_field_names::RedundantFieldNames::new(conf))),
        Box::new(move || Box::new(unnested_or_patterns::UnnestedOrPatterns::new(conf))),
        Box::new(|| Box::new(functions::EarlyFunctions)),
        Box::new(move || Box::new(doc::Documentation::new(conf))),
        Box::new(|| Box::new(suspicious_operation_groupings::SuspiciousOperationGroupings)),
        Box::new(|| Box::new(double_parens::DoubleParens)),
        Box::new(|| Box::new(unsafe_removed_from_name::UnsafeNameRemoval)),
        Box::new(|| Box::new(else_if_without_else::ElseIfWithoutElse)),
        Box::new(|| Box::new(int_plus_one::IntPlusOne)),
        Box::new(|| Box::new(formatting::Formatting)),
        Box::new(|| Box::new(misc_early::MiscEarlyLints)),
        Box::new(|| Box::new(unused_unit::UnusedUnit)),
        Box::new(|| Box::new(precedence::Precedence)),
        Box::new(|| Box::new(redundant_else::RedundantElse)),
        Box::new(|| Box::new(needless_arbitrary_self_type::NeedlessArbitrarySelfType)),
        Box::new(move || Box::new(literal_representation::LiteralDigitGrouping::new(conf))),
        Box::new(move || Box::new(literal_representation::DecimalLiteralRepresentation::new(conf))),
        Box::new(|| Box::new(tabs_in_doc_comments::TabsInDocComments)),
        Box::new(|| Box::<single_component_path_imports::SingleComponentPathImports>::default()),
        Box::new(|| Box::new(option_env_unwrap::OptionEnvUnwrap)),
        Box::new(move || Box::new(non_expressive_names::NonExpressiveNames::new(conf))),
        Box::new(move || Box::new(nonstandard_macro_braces::MacroBraces::new(conf))),
        Box::new(|| Box::new(asm_syntax::InlineAsmX86AttSyntax)),
        Box::new(|| Box::new(asm_syntax::InlineAsmX86IntelSyntax)),
        Box::new(move || Box::new(module_style::ModStyle::default())),
        Box::new(move || Box::new(disallowed_script_idents::DisallowedScriptIdents::new(conf))),
        Box::new(|| Box::new(octal_escapes::OctalEscapes)),
        Box::new(|| Box::new(single_char_lifetime_names::SingleCharLifetimeNames)),
        Box::new(|| Box::new(crate_in_macro_def::CrateInMacroDef)),
        Box::new(|| Box::new(pub_use::PubUse)),
        Box::new(move || Box::new(large_include_file::LargeIncludeFile::new(conf))),
        Box::new(|| Box::<duplicate_mod::DuplicateMod>::default()),
        Box::new(|| Box::new(unused_rounding::UnusedRounding)),
        Box::new(move || Box::new(almost_complete_range::AlmostCompleteRange::new(conf))),
        Box::new(|| Box::new(multi_assignments::MultiAssignments)),
        Box::new(|| Box::new(partial_pub_fields::PartialPubFields)),
        Box::new(|| Box::new(let_with_type_underscore::UnderscoreTyped)),
        Box::new(move || Box::new(excessive_nesting::ExcessiveNesting::new(conf))),
        Box::new(|| Box::new(ref_patterns::RefPatterns)),
        Box::new(|| Box::new(needless_else::NeedlessElse)),
        Box::new(move || Box::new(raw_strings::RawStrings::new(conf))),
        Box::new(|| Box::new(visibility::Visibility)),
        Box::new(|| Box::new(multiple_bound_locations::MultipleBoundLocations)),
        Box::new(|| Box::new(field_scoped_visibility_modifiers::FieldScopedVisibilityModifiers)),
        Box::new(|| Box::new(byte_char_slices::ByteCharSlice)),
        Box::new(|| Box::new(cfg_not_test::CfgNotTest)),
        Box::new(|| Box::new(empty_line_after::EmptyLineAfter::new())),
        // add early passes here, used by `cargo dev new_lint`
    ];
    store.early_passes.extend(early_lints);

    #[expect(clippy::type_complexity)]
    let late_lints: [Box<
        dyn for<'tcx> Fn(TyCtxt<'tcx>) -> Box<dyn LateLintPass<'tcx> + 'tcx> + sync::DynSend + sync::DynSync,
    >; _] = [
        Box::new(move |_| Box::new(operators::arithmetic_side_effects::ArithmeticSideEffects::new(conf))),
        Box::new(|_| Box::new(utils::dump_hir::DumpHir)),
        Box::new(|_| Box::new(utils::author::Author)),
        Box::new(move |tcx| Box::new(await_holding_invalid::AwaitHolding::new(tcx, conf))),
        Box::new(|_| Box::new(serde_api::SerdeApi)),
        Box::new(move |_| Box::new(types::Types::new(conf))),
        Box::new(move |_| Box::new(booleans::NonminimalBool::new(conf))),
        Box::new(|_| Box::new(enum_clike::UnportableVariant)),
        Box::new(move |_| Box::new(float_literal::FloatLiteral::new(conf))),
        Box::new(|_| Box::new(ptr::Ptr)),
        Box::new(|_| Box::new(needless_bool::NeedlessBool)),
        Box::new(|_| Box::new(bool_comparison::BoolComparison)),
        Box::new(|_| Box::new(needless_for_each::NeedlessForEach)),
        Box::new(|_| Box::new(misc::LintPass)),
        Box::new(|_| Box::new(eta_reduction::EtaReduction)),
        Box::new(|_| Box::new(mut_mut::MutMut::default())),
        Box::new(|_| Box::new(unnecessary_mut_passed::UnnecessaryMutPassed)),
        Box::new(|_| Box::<significant_drop_tightening::SignificantDropTightening<'_>>::default()),
        Box::new(move |_| Box::new(len_zero::LenZero::new(conf))),
        Box::new(move |_| Box::new(attrs::Attributes::new(conf))),
        Box::new(|_| Box::new(blocks_in_conditions::BlocksInConditions)),
        Box::new(|_| Box::new(unicode::Unicode)),
        Box::new(|_| Box::new(uninit_vec::UninitVec)),
        Box::new(|_| Box::new(unit_return_expecting_ord::UnitReturnExpectingOrd)),
        Box::new(|_| Box::new(strings::StringAdd)),
        Box::new(|_| Box::new(implicit_return::ImplicitReturn)),
        Box::new(move |_| Box::new(implicit_saturating_sub::ImplicitSaturatingSub::new(conf))),
        Box::new(|_| Box::new(default_numeric_fallback::DefaultNumericFallback)),
        Box::new(|_| Box::new(non_octal_unix_permissions::NonOctalUnixPermissions)),
        Box::new(move |_| Box::new(approx_const::ApproxConstant::new(conf))),
        Box::new(move |_| Box::new(matches::Matches::new(conf))),
        Box::new(move |_| Box::new(manual_non_exhaustive::ManualNonExhaustive::new(conf))),
        Box::new(move |_| Box::new(manual_strip::ManualStrip::new(conf))),
        Box::new(move |_| Box::new(checked_conversions::CheckedConversions::new(conf))),
        Box::new(move |_| Box::new(mem_replace::MemReplace::new(conf))),
        Box::new(move |_| Box::new(ranges::Ranges::new(conf))),
        Box::new(move |_| Box::new(from_over_into::FromOverInto::new(conf))),
        Box::new(move |_| Box::new(use_self::UseSelf::new(conf))),
        Box::new(move |_| Box::new(missing_const_for_fn::MissingConstForFn::new(conf))),
        Box::new(move |_| Box::new(needless_question_mark::NeedlessQuestionMark)),
        Box::new(move |_| Box::new(casts::Casts::new(conf))),
        Box::new(|_| Box::new(size_of_in_element_count::SizeOfInElementCount)),
        Box::new(|_| Box::new(same_name_method::SameNameMethod)),
        Box::new(move |_| Box::new(index_refutable_slice::IndexRefutableSlice::new(conf))),
        Box::new(|_| Box::<shadow::Shadow>::default()),
        Box::new(move |_| {
            Box::new(inconsistent_struct_constructor::InconsistentStructConstructor::new(
                conf,
            ))
        }),
        {
            let format_args = format_args_storage.clone();
            Box::new(move |_| Box::new(methods::Methods::new(conf, format_args.clone())))
        },
        {
            let format_args = format_args_storage.clone();
            Box::new(move |_| Box::new(unit_types::UnitTypes::new(format_args.clone())))
        },
        Box::new(move |_| Box::new(loops::Loops::new(conf))),
        Box::new(|_| Box::<main_recursion::MainRecursion>::default()),
        Box::new(move |_| Box::new(lifetimes::Lifetimes::new(conf))),
        Box::new(|_| Box::new(entry::HashMapPass)),
        Box::new(|_| Box::new(minmax::MinMaxPass)),
        Box::new(|_| Box::new(zero_div_zero::ZeroDiv)),
        Box::new(|_| Box::new(mutex_atomic::Mutex)),
        Box::new(|_| Box::new(needless_update::NeedlessUpdate)),
        Box::new(|_| Box::new(needless_borrowed_ref::NeedlessBorrowedRef)),
        Box::new(|_| Box::new(borrow_deref_ref::BorrowDerefRef)),
        Box::new(|_| Box::<no_effect::NoEffect>::default()),
        Box::new(|_| Box::new(temporary_assignment::TemporaryAssignment)),
        Box::new(move |_| Box::new(transmute::Transmute::new(conf))),
        Box::new(move |_| Box::new(cognitive_complexity::CognitiveComplexity::new(conf))),
        Box::new(move |_| Box::new(escape::BoxedLocal::new(conf))),
        Box::new(move |_| Box::new(vec::UselessVec::new(conf))),
        Box::new(move |_| Box::new(panic_unimplemented::PanicUnimplemented::new(conf))),
        Box::new(|_| Box::new(strings::StringLitAsBytes)),
        Box::new(|_| Box::new(derive::Derive)),
        Box::new(move |_| Box::new(derivable_impls::DerivableImpls::new(conf))),
        Box::new(|_| Box::new(drop_forget_ref::DropForgetRef)),
        Box::new(|_| Box::new(empty_enums::EmptyEnums)),
        Box::new(|_| Box::<regex::Regex>::default()),
        Box::new(move |tcx| Box::new(ifs::CopyAndPaste::new(tcx, conf))),
        Box::new(|_| Box::new(copy_iterator::CopyIterator)),
        {
            let format_args = format_args_storage.clone();
            Box::new(move |_| Box::new(format::UselessFormat::new(format_args.clone())))
        },
        Box::new(|_| Box::new(swap::Swap)),
        Box::new(|_| Box::new(panicking_overflow_checks::PanickingOverflowChecks)),
        Box::new(|_| Box::<new_without_default::NewWithoutDefault>::default()),
        Box::new(move |_| Box::new(disallowed_names::DisallowedNames::new(conf))),
        Box::new(move |tcx| Box::new(functions::Functions::new(tcx, conf))),
        Box::new(move |_| Box::new(doc::Documentation::new(conf))),
        Box::new(|_| Box::new(neg_multiply::NegMultiply)),
        Box::new(|_| Box::new(let_if_seq::LetIfSeq)),
        Box::new(|_| Box::new(mixed_read_write_in_expression::EvalOrderDependence)),
        Box::new(move |_| Box::new(missing_doc::MissingDoc::new(conf))),
        Box::new(|_| Box::new(missing_inline::MissingInline)),
        Box::new(move |_| Box::new(exhaustive_items::ExhaustiveItems)),
        Box::new(|_| Box::new(unused_result_ok::UnusedResultOk)),
        Box::new(|_| Box::new(match_result_ok::MatchResultOk)),
        Box::new(|_| Box::new(partialeq_ne_impl::PartialEqNeImpl)),
        Box::new(|_| Box::new(unused_io_amount::UnusedIoAmount)),
        Box::new(move |_| Box::new(large_enum_variant::LargeEnumVariant::new(conf))),
        {
            let format_args = format_args_storage.clone();
            Box::new(move |_| Box::new(explicit_write::ExplicitWrite::new(format_args.clone())))
        },
        Box::new(|_| Box::new(needless_pass_by_value::NeedlessPassByValue)),
        Box::new(move |tcx| Box::new(pass_by_ref_or_value::PassByRefOrValue::new(tcx, conf))),
        Box::new(|_| Box::new(ref_option_ref::RefOptionRef)),
        Box::new(|_| Box::new(infinite_iter::InfiniteIter)),
        Box::new(|_| Box::new(inline_fn_without_body::InlineFnWithoutBody)),
        Box::new(|_| Box::<useless_conversion::UselessConversion>::default()),
        Box::new(|_| Box::new(implicit_hasher::ImplicitHasher)),
        Box::new(|_| Box::new(fallible_impl_from::FallibleImplFrom)),
        Box::new(move |_| Box::new(question_mark::QuestionMark::new(conf))),
        Box::new(|_| Box::new(question_mark_used::QuestionMarkUsed)),
        Box::new(|_| Box::new(suspicious_trait_impl::SuspiciousImpl)),
        Box::new(|_| Box::new(map_unit_fn::MapUnit)),
        Box::new(move |_| Box::new(inherent_impl::MultipleInherentImpl::new(conf))),
        Box::new(|_| Box::new(neg_cmp_op_on_partial_ord::NoNegCompOpForPartialOrd)),
        Box::new(move |_| Box::new(unwrap::Unwrap::new(conf))),
        Box::new(move |_| Box::new(indexing_slicing::IndexingSlicing::new(conf))),
        Box::new(move |tcx| Box::new(non_copy_const::NonCopyConst::new(tcx, conf))),
        Box::new(|_| Box::new(redundant_clone::RedundantClone)),
        Box::new(|_| Box::new(slow_vector_initialization::SlowVectorInit)),
        Box::new(move |_| Box::new(unnecessary_wraps::UnnecessaryWraps::new(conf))),
        Box::new(|_| Box::new(assertions_on_constants::AssertionsOnConstants::new(conf))),
        Box::new(|_| Box::new(assertions_on_result_states::AssertionsOnResultStates)),
        Box::new(|_| Box::new(inherent_to_string::InherentToString)),
        Box::new(move |_| Box::new(trait_bounds::TraitBounds::new(conf))),
        Box::new(|_| Box::new(comparison_chain::ComparisonChain)),
        Box::new(move |tcx| Box::new(mut_key::MutableKeyType::new(tcx, conf))),
        Box::new(|_| Box::new(reference::DerefAddrOf)),
        {
            let format_args = format_args_storage.clone();
            Box::new(move |_| Box::new(format_impl::FormatImpl::new(format_args.clone())))
        },
        Box::new(|_| Box::new(redundant_closure_call::RedundantClosureCall)),
        Box::new(|_| Box::new(unused_unit::UnusedUnit)),
        Box::new(|_| Box::new(returns::Return)),
        Box::new(move |_| Box::new(collapsible_if::CollapsibleIf::new(conf))),
        Box::new(|_| Box::new(items_after_statements::ItemsAfterStatements)),
        Box::new(|_| Box::new(needless_parens_on_range_literals::NeedlessParensOnRangeLiterals)),
        Box::new(|_| Box::new(needless_continue::NeedlessContinue)),
        Box::new(|_| Box::new(create_dir::CreateDir)),
        Box::new(move |_| Box::new(item_name_repetitions::ItemNameRepetitions::new(conf))),
        Box::new(move |_| Box::new(upper_case_acronyms::UpperCaseAcronyms::new(conf))),
        Box::new(|_| Box::<default::Default>::default()),
        Box::new(move |_| Box::new(unused_self::UnusedSelf::new(conf))),
        Box::new(|_| Box::new(mutable_debug_assertion::DebugAssertWithMutCall)),
        Box::new(|_| Box::new(exit::Exit)),
        Box::new(move |_| Box::new(to_digit_is_some::ToDigitIsSome::new(conf))),
        Box::new(move |_| Box::new(large_stack_arrays::LargeStackArrays::new(conf))),
        Box::new(move |_| Box::new(large_const_arrays::LargeConstArrays::new(conf))),
        Box::new(|_| Box::new(floating_point_arithmetic::FloatingPointArithmetic)),
        Box::new(|_| Box::new(as_conversions::AsConversions)),
        Box::new(|_| Box::new(let_underscore::LetUnderscore)),
        Box::new(move |_| Box::new(excessive_bools::ExcessiveBools::new(conf))),
        Box::new(move |_| Box::new(wildcard_imports::WildcardImports::new(conf))),
        Box::new(|_| Box::<redundant_pub_crate::RedundantPubCrate>::default()),
        Box::new(|_| Box::<dereference::Dereferencing<'_>>::default()),
        Box::new(|_| Box::new(option_if_let_else::OptionIfLetElse)),
        Box::new(|_| Box::new(future_not_send::FutureNotSend)),
        Box::new(move |_| Box::new(large_futures::LargeFuture::new(conf))),
        Box::new(|_| Box::new(if_let_mutex::IfLetMutex)),
        Box::new(|_| Box::new(if_not_else::IfNotElse)),
        Box::new(|_| Box::new(equatable_if_let::PatternEquality)),
        Box::new(|_| Box::new(manual_async_fn::ManualAsyncFn)),
        Box::new(|_| Box::new(panic_in_result_fn::PanicInResultFn)),
        Box::new(|_| Box::<macro_use::MacroUseImports>::default()),
        Box::new(|_| Box::new(pattern_type_mismatch::PatternTypeMismatch)),
        Box::new(|_| Box::<unwrap_in_result::UnwrapInResult>::default()),
        Box::new(|_| Box::new(semicolon_if_nothing_returned::SemicolonIfNothingReturned)),
        Box::new(|_| Box::new(async_yields_async::AsyncYieldsAsync)),
        {
            let attrs = attr_storage.clone();
            Box::new(move |tcx| Box::new(disallowed_macros::DisallowedMacros::new(tcx, conf, attrs.clone())))
        },
        Box::new(move |tcx| Box::new(disallowed_methods::DisallowedMethods::new(tcx, conf))),
        Box::new(|_| Box::new(empty_drop::EmptyDrop)),
        Box::new(|_| Box::new(strings::StrToString)),
        Box::new(|_| Box::new(zero_sized_map_values::ZeroSizedMapValues)),
        Box::new(|_| Box::<vec_init_then_push::VecInitThenPush>::default()),
        Box::new(|_| Box::new(redundant_slicing::RedundantSlicing)),
        Box::new(|_| Box::new(from_str_radix_10::FromStrRadix10)),
        Box::new(move |_| Box::new(if_then_some_else_none::IfThenSomeElseNone::new(conf))),
        Box::new(|_| Box::new(bool_assert_comparison::BoolAssertComparison)),
        Box::new(|_| Box::<unused_async::UnusedAsync>::default()),
        Box::new(move |tcx| Box::new(disallowed_types::DisallowedTypes::new(tcx, conf))),
        Box::new(move |tcx| Box::new(missing_enforced_import_rename::ImportRename::new(tcx, conf))),
        Box::new(|_| Box::new(strlen_on_c_strings::StrlenOnCStrings)),
        Box::new(move |_| Box::new(self_named_constructors::SelfNamedConstructors)),
        Box::new(move |_| Box::new(iter_not_returning_iterator::IterNotReturningIterator)),
        Box::new(move |_| Box::new(manual_assert::ManualAssert)),
        Box::new(move |_| Box::new(non_send_fields_in_send_ty::NonSendFieldInSendTy::new(conf))),
        Box::new(move |_| Box::new(undocumented_unsafe_blocks::UndocumentedUnsafeBlocks::new(conf))),
        {
            let format_args = format_args_storage.clone();
            Box::new(move |tcx| Box::new(format_args::FormatArgs::new(tcx, conf, format_args.clone())))
        },
        Box::new(|_| Box::new(trailing_empty_array::TrailingEmptyArray)),
        Box::new(|_| Box::new(needless_late_init::NeedlessLateInit)),
        Box::new(|_| Box::new(return_self_not_must_use::ReturnSelfNotMustUse)),
        Box::new(|_| Box::new(init_numbered_fields::NumberedFields)),
        Box::new(move |_| Box::new(manual_bits::ManualBits::new(conf))),
        Box::new(|_| Box::new(default_union_representation::DefaultUnionRepresentation)),
        Box::new(|_| Box::<only_used_in_recursion::OnlyUsedInRecursion>::default()),
        Box::new(move |_| Box::new(dbg_macro::DbgMacro::new(conf))),
        {
            let format_args = format_args_storage.clone();
            Box::new(move |_| Box::new(write::Write::new(conf, format_args.clone())))
        },
        Box::new(move |_| Box::new(cargo::Cargo::new(conf))),
        Box::new(|_| Box::new(empty_with_brackets::EmptyWithBrackets::default())),
        Box::new(|_| Box::new(unnecessary_owned_empty_strings::UnnecessaryOwnedEmptyStrings)),
        Box::new(|_| Box::new(format_push_string::FormatPushString)),
        Box::new(move |_| Box::new(large_include_file::LargeIncludeFile::new(conf))),
        Box::new(|_| Box::new(strings::TrimSplitWhitespace)),
        Box::new(|_| Box::new(rc_clone_in_vec_init::RcCloneInVecInit)),
        Box::new(|_| Box::new(swap_ptr_to_ref::SwapPtrToRef)),
        Box::new(|_| Box::new(mismatching_type_param_order::TypeParamMismatch)),
        Box::new(|_| Box::new(read_zero_byte_vec::ReadZeroByteVec)),
        Box::new(|_| Box::new(default_instead_of_iter_empty::DefaultIterEmpty)),
        Box::new(move |_| Box::new(manual_rem_euclid::ManualRemEuclid::new(conf))),
        Box::new(move |_| Box::new(manual_retain::ManualRetain::new(conf))),
        Box::new(move |_| Box::new(manual_rotate::ManualRotate)),
        Box::new(move |_| Box::new(operators::Operators::new(conf))),
        Box::new(move |_| Box::new(std_instead_of_core::StdReexports::new(conf))),
        Box::new(move |_| Box::new(time_subtraction::UncheckedTimeSubtraction::new(conf))),
        Box::new(|_| Box::new(partialeq_to_none::PartialeqToNone)),
        Box::new(move |_| Box::new(manual_abs_diff::ManualAbsDiff::new(conf))),
        Box::new(move |_| Box::new(manual_clamp::ManualClamp::new(conf))),
        Box::new(|_| Box::new(manual_string_new::ManualStringNew)),
        Box::new(|_| Box::new(unused_peekable::UnusedPeekable)),
        Box::new(|_| Box::new(bool_to_int_with_if::BoolToIntWithIf)),
        Box::new(|_| Box::new(box_default::BoxDefault)),
        Box::new(|_| Box::new(implicit_saturating_add::ImplicitSaturatingAdd)),
        Box::new(|_| Box::new(missing_trait_methods::MissingTraitMethods)),
        Box::new(|_| Box::new(from_raw_with_void_ptr::FromRawWithVoidPtr)),
        Box::new(|_| Box::new(suspicious_xor_used_as_pow::ConfusingXorAndPow)),
        Box::new(move |_| Box::new(manual_is_ascii_check::ManualIsAsciiCheck::new(conf))),
        Box::new(move |_| Box::new(semicolon_block::SemicolonBlock::new(conf))),
        Box::new(|_| Box::new(permissions_set_readonly_false::PermissionsSetReadonlyFalse)),
        Box::new(|_| Box::new(size_of_ref::SizeOfRef)),
        Box::new(|_| Box::new(multiple_unsafe_ops_per_block::MultipleUnsafeOpsPerBlock)),
        Box::new(move |_| Box::new(extra_unused_type_parameters::ExtraUnusedTypeParameters::new(conf))),
        Box::new(|_| Box::new(no_mangle_with_rust_abi::NoMangleWithRustAbi)),
        Box::new(|_| Box::new(collection_is_never_read::CollectionIsNeverRead)),
        Box::new(|_| Box::new(missing_assert_message::MissingAssertMessage)),
        Box::new(|_| Box::new(needless_maybe_sized::NeedlessMaybeSized)),
        Box::new(|_| Box::new(redundant_async_block::RedundantAsyncBlock)),
        Box::new(move |_| Box::new(manual_main_separator_str::ManualMainSeparatorStr::new(conf))),
        Box::new(|_| Box::new(unnecessary_struct_initialization::UnnecessaryStruct)),
        Box::new(move |_| Box::new(unnecessary_box_returns::UnnecessaryBoxReturns::new(conf))),
        Box::new(|_| Box::new(tests_outside_test_module::TestsOutsideTestModule)),
        Box::new(|_| Box::new(manual_slice_size_calculation::ManualSliceSizeCalculation::new(conf))),
        Box::new(|_| Box::new(items_after_test_module::ItemsAfterTestModule)),
        Box::new(|_| Box::new(default_constructed_unit_structs::DefaultConstructedUnitStructs)),
        Box::new(|_| Box::new(missing_fields_in_debug::MissingFieldsInDebug)),
        Box::new(|_| Box::new(endian_bytes::EndianBytes)),
        Box::new(|_| Box::new(redundant_type_annotations::RedundantTypeAnnotations)),
        Box::new(|_| Box::new(arc_with_non_send_sync::ArcWithNonSendSync)),
        Box::new(|_| Box::new(needless_ifs::NeedlessIfs)),
        Box::new(move |_| Box::new(min_ident_chars::MinIdentChars::new(conf))),
        Box::new(move |_| Box::new(large_stack_frames::LargeStackFrames::new(conf))),
        Box::new(|_| Box::new(single_range_in_vec_init::SingleRangeInVecInit)),
        Box::new(move |_| Box::new(needless_pass_by_ref_mut::NeedlessPassByRefMut::new(conf))),
        Box::new(|tcx| Box::new(non_canonical_impls::NonCanonicalImpls::new(tcx))),
        Box::new(move |_| Box::new(single_call_fn::SingleCallFn::new(conf))),
        Box::new(move |_| Box::new(legacy_numeric_constants::LegacyNumericConstants::new(conf))),
        Box::new(|_| Box::new(manual_range_patterns::ManualRangePatterns)),
        Box::new(move |_| Box::new(tuple_array_conversions::TupleArrayConversions::new(conf))),
        Box::new(move |_| Box::new(manual_float_methods::ManualFloatMethods::new(conf))),
        Box::new(|_| Box::new(four_forward_slashes::FourForwardSlashes)),
        Box::new(|_| Box::new(error_impl_error::ErrorImplError)),
        Box::new(move |_| Box::new(absolute_paths::AbsolutePaths::new(conf))),
        Box::new(|_| Box::new(redundant_locals::RedundantLocals)),
        Box::new(|_| Box::new(ignored_unit_patterns::IgnoredUnitPatterns)),
        Box::new(|_| Box::<reserve_after_initialization::ReserveAfterInitialization>::default()),
        Box::new(|_| Box::new(implied_bounds_in_impls::ImpliedBoundsInImpls)),
        Box::new(|_| Box::new(missing_asserts_for_indexing::MissingAssertsForIndexing)),
        Box::new(|_| Box::new(unnecessary_map_on_constructor::UnnecessaryMapOnConstructor)),
        Box::new(move |_| {
            Box::new(needless_borrows_for_generic_args::NeedlessBorrowsForGenericArgs::new(
                conf,
            ))
        }),
        Box::new(move |_| Box::new(manual_hash_one::ManualHashOne::new(conf))),
        Box::new(|_| Box::new(iter_without_into_iter::IterWithoutIntoIter)),
        Box::new(|_| Box::<pathbuf_init_then_push::PathbufThenPush<'_>>::default()),
        Box::new(|_| Box::new(iter_over_hash_type::IterOverHashType)),
        Box::new(|_| Box::new(impl_hash_with_borrow_str_and_bytes::ImplHashWithBorrowStrBytes)),
        Box::new(move |_| Box::new(repeat_vec_with_capacity::RepeatVecWithCapacity::new(conf))),
        Box::new(|_| Box::new(uninhabited_references::UninhabitedReferences)),
        Box::new(|_| Box::new(ineffective_open_options::IneffectiveOpenOptions)),
        Box::new(|_| Box::<unconditional_recursion::UnconditionalRecursion>::default()),
        Box::new(move |_| Box::new(pub_underscore_fields::PubUnderscoreFields::new(conf))),
        Box::new(move |_| Box::new(missing_const_for_thread_local::MissingConstForThreadLocal::new(conf))),
        Box::new(move |tcx| Box::new(incompatible_msrv::IncompatibleMsrv::new(tcx, conf))),
        Box::new(|_| Box::new(to_string_trait_impl::ToStringTraitImpl)),
        Box::new(move |_| Box::new(assigning_clones::AssigningClones::new(conf))),
        Box::new(|_| Box::new(zero_repeat_side_effects::ZeroRepeatSideEffects)),
        Box::new(move |_| Box::new(macro_metavars_in_unsafe::ExprMetavarsInUnsafe::new(conf))),
        Box::new(move |_| Box::new(string_patterns::StringPatterns::new(conf))),
        Box::new(|_| Box::new(set_contains_or_insert::SetContainsOrInsert)),
        Box::new(|_| Box::new(zombie_processes::ZombieProcesses)),
        Box::new(|_| Box::new(pointers_in_nomem_asm_block::PointersInNomemAsmBlock)),
        Box::new(move |_| Box::new(manual_is_power_of_two::ManualIsPowerOfTwo::new(conf))),
        Box::new(|_| Box::new(non_zero_suggestions::NonZeroSuggestions)),
        Box::new(|_| Box::new(literal_string_with_formatting_args::LiteralStringWithFormattingArg)),
        Box::new(move |_| Box::new(unused_trait_names::UnusedTraitNames::new(conf))),
        Box::new(|_| Box::new(manual_ignore_case_cmp::ManualIgnoreCaseCmp)),
        Box::new(|_| Box::new(unnecessary_literal_bound::UnnecessaryLiteralBound)),
        Box::new(move |_| Box::new(arbitrary_source_item_ordering::ArbitrarySourceItemOrdering::new(conf))),
        Box::new(|_| Box::new(useless_concat::UselessConcat)),
        Box::new(|_| Box::new(unneeded_struct_pattern::UnneededStructPattern)),
        Box::new(|_| Box::<unnecessary_semicolon::UnnecessarySemicolon>::default()),
        Box::new(move |_| Box::new(non_std_lazy_statics::NonStdLazyStatic::new(conf))),
        Box::new(|_| Box::new(manual_option_as_slice::ManualOptionAsSlice::new(conf))),
        Box::new(|_| Box::new(single_option_map::SingleOptionMap)),
        Box::new(move |_| Box::new(redundant_test_prefix::RedundantTestPrefix)),
        Box::new(|_| Box::new(cloned_ref_to_slice_refs::ClonedRefToSliceRefs::new(conf))),
        Box::new(|_| Box::new(infallible_try_from::InfallibleTryFrom)),
        Box::new(|_| Box::new(coerce_container_to_any::CoerceContainerToAny)),
        Box::new(|_| Box::new(toplevel_ref_arg::ToplevelRefArg)),
        Box::new(|_| Box::new(volatile_composites::VolatileComposites)),
        Box::new(|_| Box::<replace_box::ReplaceBox>::default()),
        // add late passes here, used by `cargo dev new_lint`
    ];
    store.late_passes.extend(late_lints);
}
