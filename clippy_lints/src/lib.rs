#![feature(box_patterns)]
#![feature(control_flow_into_value)]
#![feature(exact_div)]
#![feature(f128)]
#![feature(f16)]
#![feature(iter_intersperse)]
#![feature(iter_partition_in_place)]
#![feature(macro_metavar_expr_concat)]
#![feature(never_type)]
#![feature(rustc_private)]
#![feature(stmt_expr_attributes)]
#![feature(unwrap_infallible)]
#![recursion_limit = "512"]
#![allow(
    clippy::missing_docs_in_private_items,
    clippy::must_use_candidate,
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

mod combined_early_pass;
mod combined_late_pass;

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
mod disallowed_fields;
mod disallowed_macros;
mod disallowed_methods;
mod disallowed_names;
mod disallowed_script_idents;
mod disallowed_types;
mod doc;
mod double_parens;
mod drop_forget_ref;
mod duplicate_mod;
mod duration_suboptimal_units;
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
mod inline_trait_bounds;
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
mod len_without_is_empty;
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
mod manual_assert_eq;
mod manual_async_fn;
mod manual_bits;
mod manual_checked_ops;
mod manual_clamp;
mod manual_float_methods;
mod manual_hash_one;
mod manual_ignore_case_cmp;
mod manual_ilog2;
mod manual_is_ascii_check;
mod manual_is_power_of_two;
mod manual_let_else;
mod manual_main_separator_str;
mod manual_non_exhaustive;
mod manual_noop_waker;
mod manual_option_as_slice;
mod manual_pop_if;
mod manual_range_patterns;
mod manual_rem_euclid;
mod manual_retain;
mod manual_rotate;
mod manual_slice_size_calculation;
mod manual_string_new;
mod manual_strip;
mod manual_take;
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
mod same_length_and_capacity;
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
mod useless_vec;
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
use rustc_lint::Lint;
use rustc_middle::ty::TyCtxt;
use utils::attr_collector::AttrStorage;

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

    {
        let format_args = format_args_storage.clone();
        let attrs = attr_storage.clone();
        store.early_passes.push(Box::new(move || {
            Box::new(CombinedEarlyLintPass::new(conf, format_args.clone(), attrs.clone()))
        }));
    }

    store.late_passes.push(Box::new(move |tcx: TyCtxt<'_>| {
        let dont_need = tcx.lints_that_dont_need_to_run(());
        let is_active = |lints: &rustc_lint::LintVec| {
            lints.is_empty()
                || !lints
                    .iter()
                    .all(|lint| dont_need.contains(&rustc_lint::LintId::of(lint)))
        };
        Box::new(CombinedLateLintPass::new(
            tcx,
            conf,
            format_args_storage.clone(),
            attr_storage.clone(),
            &is_active,
        ))
    }));
}

// Fold every early pass into one statically-combined struct (see
// `combined_early_pass`); the method list comes from `early_lint_methods!`.
#[rustfmt::skip]
rustc_lint::early_lint_methods!(
    crate::combined_early_lint_pass,
    [CombinedEarlyLintPass, (conf: &'static Conf, format_args: FormatArgsStorage, attrs: AttrStorage), [
        FormatArgsCollector: utils::format_args_collector::FormatArgsCollector = utils::format_args_collector::FormatArgsCollector::new(format_args.clone()),
        AttrCollector: utils::attr_collector::AttrCollector = utils::attr_collector::AttrCollector::new(attrs.clone()),
        PostExpansionEarlyAttributes: attrs::PostExpansionEarlyAttributes = attrs::PostExpansionEarlyAttributes::new(conf),
        UnnecessarySelfImports: unnecessary_self_imports::UnnecessarySelfImports = unnecessary_self_imports::UnnecessarySelfImports,
        RedundantStaticLifetimes: redundant_static_lifetimes::RedundantStaticLifetimes = redundant_static_lifetimes::RedundantStaticLifetimes::new(conf),
        RedundantFieldNames: redundant_field_names::RedundantFieldNames = redundant_field_names::RedundantFieldNames::new(conf),
        UnnestedOrPatterns: unnested_or_patterns::UnnestedOrPatterns = unnested_or_patterns::UnnestedOrPatterns::new(conf),
        EarlyFunctions: functions::EarlyFunctions = functions::EarlyFunctions,
        Documentation: doc::Documentation = doc::Documentation::new(conf),
        SuspiciousOperationGroupings: suspicious_operation_groupings::SuspiciousOperationGroupings = suspicious_operation_groupings::SuspiciousOperationGroupings,
        DoubleParens: double_parens::DoubleParens = double_parens::DoubleParens,
        UnsafeNameRemoval: unsafe_removed_from_name::UnsafeNameRemoval = unsafe_removed_from_name::UnsafeNameRemoval,
        ElseIfWithoutElse: else_if_without_else::ElseIfWithoutElse = else_if_without_else::ElseIfWithoutElse,
        IntPlusOne: int_plus_one::IntPlusOne = int_plus_one::IntPlusOne,
        Formatting: formatting::Formatting = formatting::Formatting,
        MiscEarlyLints: misc_early::MiscEarlyLints = misc_early::MiscEarlyLints,
        UnusedUnit: unused_unit::UnusedUnit = unused_unit::UnusedUnit,
        Precedence: precedence::Precedence = precedence::Precedence,
        RedundantElse: redundant_else::RedundantElse = redundant_else::RedundantElse,
        NeedlessArbitrarySelfType: needless_arbitrary_self_type::NeedlessArbitrarySelfType = needless_arbitrary_self_type::NeedlessArbitrarySelfType,
        LiteralDigitGrouping: literal_representation::LiteralDigitGrouping = literal_representation::LiteralDigitGrouping::new(conf),
        DecimalLiteralRepresentation: literal_representation::DecimalLiteralRepresentation = literal_representation::DecimalLiteralRepresentation::new(conf),
        TabsInDocComments: tabs_in_doc_comments::TabsInDocComments = tabs_in_doc_comments::TabsInDocComments,
        SingleComponentPathImports: single_component_path_imports::SingleComponentPathImports = single_component_path_imports::SingleComponentPathImports::default(),
        OptionEnvUnwrap: option_env_unwrap::OptionEnvUnwrap = option_env_unwrap::OptionEnvUnwrap,
        NonExpressiveNames: non_expressive_names::NonExpressiveNames = non_expressive_names::NonExpressiveNames::new(conf),
        MacroBraces: nonstandard_macro_braces::MacroBraces = nonstandard_macro_braces::MacroBraces::new(conf),
        InlineAsmX86AttSyntax: asm_syntax::InlineAsmX86AttSyntax = asm_syntax::InlineAsmX86AttSyntax,
        InlineAsmX86IntelSyntax: asm_syntax::InlineAsmX86IntelSyntax = asm_syntax::InlineAsmX86IntelSyntax,
        ModStyle: module_style::ModStyle = module_style::ModStyle::default(),
        DisallowedScriptIdents: disallowed_script_idents::DisallowedScriptIdents = disallowed_script_idents::DisallowedScriptIdents::new(conf),
        OctalEscapes: octal_escapes::OctalEscapes = octal_escapes::OctalEscapes,
        SingleCharLifetimeNames: single_char_lifetime_names::SingleCharLifetimeNames = single_char_lifetime_names::SingleCharLifetimeNames,
        CrateInMacroDef: crate_in_macro_def::CrateInMacroDef = crate_in_macro_def::CrateInMacroDef,
        PubUse: pub_use::PubUse = pub_use::PubUse,
        LargeIncludeFile: large_include_file::LargeIncludeFile = large_include_file::LargeIncludeFile::new(conf),
        DuplicateMod: duplicate_mod::DuplicateMod = duplicate_mod::DuplicateMod::default(),
        UnusedRounding: unused_rounding::UnusedRounding = unused_rounding::UnusedRounding,
        AlmostCompleteRange: almost_complete_range::AlmostCompleteRange = almost_complete_range::AlmostCompleteRange::new(conf),
        MultiAssignments: multi_assignments::MultiAssignments = multi_assignments::MultiAssignments,
        PartialPubFields: partial_pub_fields::PartialPubFields = partial_pub_fields::PartialPubFields,
        UnderscoreTyped: let_with_type_underscore::UnderscoreTyped = let_with_type_underscore::UnderscoreTyped,
        ExcessiveNesting: excessive_nesting::ExcessiveNesting = excessive_nesting::ExcessiveNesting::new(conf),
        RefPatterns: ref_patterns::RefPatterns = ref_patterns::RefPatterns,
        NeedlessElse: needless_else::NeedlessElse = needless_else::NeedlessElse,
        RawStrings: raw_strings::RawStrings = raw_strings::RawStrings::new(conf),
        Visibility: visibility::Visibility = visibility::Visibility,
        MultipleBoundLocations: multiple_bound_locations::MultipleBoundLocations = multiple_bound_locations::MultipleBoundLocations,
        FieldScopedVisibilityModifiers: field_scoped_visibility_modifiers::FieldScopedVisibilityModifiers = field_scoped_visibility_modifiers::FieldScopedVisibilityModifiers,
        CfgNotTest: cfg_not_test::CfgNotTest = cfg_not_test::CfgNotTest,
        EmptyLineAfter: empty_line_after::EmptyLineAfter = empty_line_after::EmptyLineAfter::new(),
        InlineTraitBounds: inline_trait_bounds::InlineTraitBounds = inline_trait_bounds::InlineTraitBounds,
        // add early passes here, used by `cargo dev new_lint`
    ]]
);

// Fold every late pass into one statically-combined struct (see
// `combined_late_pass`); the method list comes from `late_lint_methods!`.
#[rustfmt::skip]
rustc_lint::late_lint_methods!(
    crate::combined_late_lint_pass,
    [CombinedLateLintPass, (tcx: TyCtxt<'tcx>, conf: &'static Conf, format_args: FormatArgsStorage, attrs: AttrStorage), [
        ArithmeticSideEffects: operators::arithmetic_side_effects::ArithmeticSideEffects = operators::arithmetic_side_effects::ArithmeticSideEffects::new(conf),
        DumpHir: utils::dump_hir::DumpHir = utils::dump_hir::DumpHir,
        Author: utils::author::Author = utils::author::Author,
        AwaitHolding: await_holding_invalid::AwaitHolding = await_holding_invalid::AwaitHolding::new(tcx, conf),
        SerdeApi: serde_api::SerdeApi = serde_api::SerdeApi,
        Types: types::Types = types::Types::new(conf),
        NonminimalBool: booleans::NonminimalBool = booleans::NonminimalBool::new(conf),
        UnportableVariant: enum_clike::UnportableVariant = enum_clike::UnportableVariant,
        FloatLiteral: float_literal::FloatLiteral = float_literal::FloatLiteral::new(conf),
        Ptr: ptr::Ptr = ptr::Ptr,
        NeedlessBool: needless_bool::NeedlessBool = needless_bool::NeedlessBool,
        BoolComparison: bool_comparison::BoolComparison = bool_comparison::BoolComparison,
        NeedlessForEach: needless_for_each::NeedlessForEach = needless_for_each::NeedlessForEach,
        LintPass: misc::LintPass = misc::LintPass,
        EtaReduction: eta_reduction::EtaReduction = eta_reduction::EtaReduction,
        MutMut: mut_mut::MutMut = mut_mut::MutMut::default(),
        UnnecessaryMutPassed: unnecessary_mut_passed::UnnecessaryMutPassed = unnecessary_mut_passed::UnnecessaryMutPassed,
        SignificantDropTightening: significant_drop_tightening::SignificantDropTightening<'tcx> = <significant_drop_tightening::SignificantDropTightening<'_>>::default(),
        LenZero: len_zero::LenZero = len_zero::LenZero::new(conf),
        LenWithoutIsEmpty: len_without_is_empty::LenWithoutIsEmpty = len_without_is_empty::LenWithoutIsEmpty,
        Attributes: attrs::Attributes = attrs::Attributes::new(conf),
        BlocksInConditions: blocks_in_conditions::BlocksInConditions = blocks_in_conditions::BlocksInConditions,
        Unicode: unicode::Unicode = unicode::Unicode,
        UninitVec: uninit_vec::UninitVec = uninit_vec::UninitVec,
        UnitReturnExpectingOrd: unit_return_expecting_ord::UnitReturnExpectingOrd = unit_return_expecting_ord::UnitReturnExpectingOrd,
        StringAdd: strings::StringAdd = strings::StringAdd,
        ImplicitReturn: implicit_return::ImplicitReturn = implicit_return::ImplicitReturn,
        ImplicitSaturatingSub: implicit_saturating_sub::ImplicitSaturatingSub = implicit_saturating_sub::ImplicitSaturatingSub::new(conf),
        DefaultNumericFallback: default_numeric_fallback::DefaultNumericFallback = default_numeric_fallback::DefaultNumericFallback,
        NonOctalUnixPermissions: non_octal_unix_permissions::NonOctalUnixPermissions = non_octal_unix_permissions::NonOctalUnixPermissions,
        ApproxConstant: approx_const::ApproxConstant = approx_const::ApproxConstant::new(conf),
        Matches: matches::Matches = matches::Matches::new(conf),
        ManualNonExhaustive: manual_non_exhaustive::ManualNonExhaustive = manual_non_exhaustive::ManualNonExhaustive::new(conf),
        ManualStrip: manual_strip::ManualStrip = manual_strip::ManualStrip::new(conf),
        CheckedConversions: checked_conversions::CheckedConversions = checked_conversions::CheckedConversions::new(conf),
        MemReplace: mem_replace::MemReplace = mem_replace::MemReplace::new(conf),
        Ranges: ranges::Ranges = ranges::Ranges::new(conf),
        FromOverInto: from_over_into::FromOverInto = from_over_into::FromOverInto::new(conf),
        UseSelf: use_self::UseSelf = use_self::UseSelf::new(conf),
        MissingConstForFn: missing_const_for_fn::MissingConstForFn = missing_const_for_fn::MissingConstForFn::new(conf),
        NeedlessQuestionMark: needless_question_mark::NeedlessQuestionMark = needless_question_mark::NeedlessQuestionMark,
        Casts: casts::Casts = casts::Casts::new(conf),
        SizeOfInElementCount: size_of_in_element_count::SizeOfInElementCount = size_of_in_element_count::SizeOfInElementCount,
        SameNameMethod: same_name_method::SameNameMethod = same_name_method::SameNameMethod,
        IndexRefutableSlice: index_refutable_slice::IndexRefutableSlice = index_refutable_slice::IndexRefutableSlice::new(conf),
        Shadow: shadow::Shadow = <shadow::Shadow>::default(),
        InconsistentStructConstructor: inconsistent_struct_constructor::InconsistentStructConstructor = inconsistent_struct_constructor::InconsistentStructConstructor::new( conf, ),
        Methods: methods::Methods = methods::Methods::new(conf, format_args.clone()),
        UnitTypes: unit_types::UnitTypes = unit_types::UnitTypes::new(format_args.clone()),
        Loops: loops::Loops = loops::Loops::new(conf),
        MainRecursion: main_recursion::MainRecursion = <main_recursion::MainRecursion>::default(),
        Lifetimes: lifetimes::Lifetimes = lifetimes::Lifetimes::new(conf),
        HashMapPass: entry::HashMapPass = entry::HashMapPass,
        MinMaxPass: minmax::MinMaxPass = minmax::MinMaxPass,
        ZeroDiv: zero_div_zero::ZeroDiv = zero_div_zero::ZeroDiv,
        Mutex: mutex_atomic::Mutex = mutex_atomic::Mutex,
        NeedlessUpdate: needless_update::NeedlessUpdate = needless_update::NeedlessUpdate,
        NeedlessBorrowedRef: needless_borrowed_ref::NeedlessBorrowedRef = needless_borrowed_ref::NeedlessBorrowedRef,
        BorrowDerefRef: borrow_deref_ref::BorrowDerefRef = borrow_deref_ref::BorrowDerefRef,
        NoEffect: no_effect::NoEffect = <no_effect::NoEffect>::default(),
        TemporaryAssignment: temporary_assignment::TemporaryAssignment = temporary_assignment::TemporaryAssignment,
        Transmute: transmute::Transmute = transmute::Transmute::new(conf),
        CognitiveComplexity: cognitive_complexity::CognitiveComplexity = cognitive_complexity::CognitiveComplexity::new(conf),
        BoxedLocal: escape::BoxedLocal = escape::BoxedLocal::new(conf),
        UselessVec: useless_vec::UselessVec = useless_vec::UselessVec::new(conf),
        PanicUnimplemented: panic_unimplemented::PanicUnimplemented = panic_unimplemented::PanicUnimplemented::new(conf),
        StringLitAsBytes: strings::StringLitAsBytes = strings::StringLitAsBytes,
        Derive: derive::Derive = derive::Derive,
        DerivableImpls: derivable_impls::DerivableImpls = derivable_impls::DerivableImpls::new(conf),
        DropForgetRef: drop_forget_ref::DropForgetRef = drop_forget_ref::DropForgetRef,
        EmptyEnums: empty_enums::EmptyEnums = empty_enums::EmptyEnums,
        Regex: regex::Regex = <regex::Regex>::default(),
        CopyAndPaste: ifs::CopyAndPaste<'tcx> = ifs::CopyAndPaste::new(tcx, conf),
        CopyIterator: copy_iterator::CopyIterator = copy_iterator::CopyIterator,
        UselessFormat: format::UselessFormat = format::UselessFormat::new(format_args.clone()),
        Swap: swap::Swap = swap::Swap,
        PanickingOverflowChecks: panicking_overflow_checks::PanickingOverflowChecks = panicking_overflow_checks::PanickingOverflowChecks,
        NewWithoutDefault: new_without_default::NewWithoutDefault = <new_without_default::NewWithoutDefault>::default(),
        DisallowedNames: disallowed_names::DisallowedNames = disallowed_names::DisallowedNames::new(conf),
        Functions: functions::Functions = functions::Functions::new(tcx, conf),
        Documentation: doc::Documentation = doc::Documentation::new(conf),
        NegMultiply: neg_multiply::NegMultiply = neg_multiply::NegMultiply,
        LetIfSeq: let_if_seq::LetIfSeq = let_if_seq::LetIfSeq,
        EvalOrderDependence: mixed_read_write_in_expression::EvalOrderDependence = mixed_read_write_in_expression::EvalOrderDependence,
        MissingDoc: missing_doc::MissingDoc = missing_doc::MissingDoc::new(conf),
        MissingInline: missing_inline::MissingInline = missing_inline::MissingInline,
        ExhaustiveItems: exhaustive_items::ExhaustiveItems = exhaustive_items::ExhaustiveItems,
        UnusedResultOk: unused_result_ok::UnusedResultOk = unused_result_ok::UnusedResultOk,
        MatchResultOk: match_result_ok::MatchResultOk = match_result_ok::MatchResultOk,
        PartialEqNeImpl: partialeq_ne_impl::PartialEqNeImpl = partialeq_ne_impl::PartialEqNeImpl,
        UnusedIoAmount: unused_io_amount::UnusedIoAmount = unused_io_amount::UnusedIoAmount,
        LargeEnumVariant: large_enum_variant::LargeEnumVariant = large_enum_variant::LargeEnumVariant::new(conf),
        ExplicitWrite: explicit_write::ExplicitWrite = explicit_write::ExplicitWrite::new(format_args.clone()),
        NeedlessPassByValue: needless_pass_by_value::NeedlessPassByValue = needless_pass_by_value::NeedlessPassByValue,
        PassByRefOrValue: pass_by_ref_or_value::PassByRefOrValue = pass_by_ref_or_value::PassByRefOrValue::new(tcx, conf),
        RefOptionRef: ref_option_ref::RefOptionRef = ref_option_ref::RefOptionRef,
        InfiniteIter: infinite_iter::InfiniteIter = infinite_iter::InfiniteIter,
        InlineFnWithoutBody: inline_fn_without_body::InlineFnWithoutBody = inline_fn_without_body::InlineFnWithoutBody,
        UselessConversion: useless_conversion::UselessConversion = <useless_conversion::UselessConversion>::default(),
        ImplicitHasher: implicit_hasher::ImplicitHasher = implicit_hasher::ImplicitHasher,
        FallibleImplFrom: fallible_impl_from::FallibleImplFrom = fallible_impl_from::FallibleImplFrom,
        QuestionMark: question_mark::QuestionMark = question_mark::QuestionMark::new(conf),
        QuestionMarkUsed: question_mark_used::QuestionMarkUsed = question_mark_used::QuestionMarkUsed,
        SuspiciousImpl: suspicious_trait_impl::SuspiciousImpl = suspicious_trait_impl::SuspiciousImpl,
        MapUnit: map_unit_fn::MapUnit = map_unit_fn::MapUnit,
        MultipleInherentImpl: inherent_impl::MultipleInherentImpl = inherent_impl::MultipleInherentImpl::new(conf),
        NoNegCompOpForPartialOrd: neg_cmp_op_on_partial_ord::NoNegCompOpForPartialOrd = neg_cmp_op_on_partial_ord::NoNegCompOpForPartialOrd,
        Unwrap: unwrap::Unwrap = unwrap::Unwrap::new(conf),
        IndexingSlicing: indexing_slicing::IndexingSlicing = indexing_slicing::IndexingSlicing::new(conf),
        NonCopyConst: non_copy_const::NonCopyConst<'tcx> = non_copy_const::NonCopyConst::new(tcx, conf),
        RedundantClone: redundant_clone::RedundantClone = redundant_clone::RedundantClone,
        SlowVectorInit: slow_vector_initialization::SlowVectorInit = slow_vector_initialization::SlowVectorInit,
        UnnecessaryWraps: unnecessary_wraps::UnnecessaryWraps = unnecessary_wraps::UnnecessaryWraps::new(conf),
        AssertionsOnConstants: assertions_on_constants::AssertionsOnConstants = assertions_on_constants::AssertionsOnConstants::new(conf),
        AssertionsOnResultStates: assertions_on_result_states::AssertionsOnResultStates = assertions_on_result_states::AssertionsOnResultStates,
        InherentToString: inherent_to_string::InherentToString = inherent_to_string::InherentToString,
        TraitBounds: trait_bounds::TraitBounds = trait_bounds::TraitBounds::new(conf),
        ComparisonChain: comparison_chain::ComparisonChain = comparison_chain::ComparisonChain,
        MutableKeyType: mut_key::MutableKeyType<'tcx> = mut_key::MutableKeyType::new(tcx, conf),
        DerefAddrOf: reference::DerefAddrOf = reference::DerefAddrOf,
        FormatImpl: format_impl::FormatImpl = format_impl::FormatImpl::new(format_args.clone()),
        RedundantClosureCall: redundant_closure_call::RedundantClosureCall = redundant_closure_call::RedundantClosureCall,
        UnusedUnit: unused_unit::UnusedUnit = unused_unit::UnusedUnit,
        Return: returns::Return = returns::Return,
        CollapsibleIf: collapsible_if::CollapsibleIf = collapsible_if::CollapsibleIf::new(conf),
        ItemsAfterStatements: items_after_statements::ItemsAfterStatements = items_after_statements::ItemsAfterStatements,
        NeedlessParensOnRangeLiterals: needless_parens_on_range_literals::NeedlessParensOnRangeLiterals = needless_parens_on_range_literals::NeedlessParensOnRangeLiterals,
        NeedlessContinue: needless_continue::NeedlessContinue = needless_continue::NeedlessContinue,
        CreateDir: create_dir::CreateDir = create_dir::CreateDir,
        ItemNameRepetitions: item_name_repetitions::ItemNameRepetitions = item_name_repetitions::ItemNameRepetitions::new(conf),
        UpperCaseAcronyms: upper_case_acronyms::UpperCaseAcronyms = upper_case_acronyms::UpperCaseAcronyms::new(conf),
        Default: default::Default = <default::Default>::default(),
        UnusedSelf: unused_self::UnusedSelf = unused_self::UnusedSelf::new(conf),
        DebugAssertWithMutCall: mutable_debug_assertion::DebugAssertWithMutCall = mutable_debug_assertion::DebugAssertWithMutCall,
        Exit: exit::Exit = exit::Exit,
        ToDigitIsSome: to_digit_is_some::ToDigitIsSome = to_digit_is_some::ToDigitIsSome::new(conf),
        LargeStackArrays: large_stack_arrays::LargeStackArrays = large_stack_arrays::LargeStackArrays::new(conf),
        LargeConstArrays: large_const_arrays::LargeConstArrays = large_const_arrays::LargeConstArrays::new(conf),
        FloatingPointArithmetic: floating_point_arithmetic::FloatingPointArithmetic = floating_point_arithmetic::FloatingPointArithmetic,
        AsConversions: as_conversions::AsConversions = as_conversions::AsConversions,
        LetUnderscore: let_underscore::LetUnderscore = let_underscore::LetUnderscore,
        ExcessiveBools: excessive_bools::ExcessiveBools = excessive_bools::ExcessiveBools::new(conf),
        WildcardImports: wildcard_imports::WildcardImports = wildcard_imports::WildcardImports::new(conf),
        RedundantPubCrate: redundant_pub_crate::RedundantPubCrate = <redundant_pub_crate::RedundantPubCrate>::default(),
        Dereferencing: dereference::Dereferencing<'tcx> = <dereference::Dereferencing<'_>>::default(),
        OptionIfLetElse: option_if_let_else::OptionIfLetElse = option_if_let_else::OptionIfLetElse,
        FutureNotSend: future_not_send::FutureNotSend = future_not_send::FutureNotSend,
        LargeFuture: large_futures::LargeFuture = large_futures::LargeFuture::new(conf),
        IfLetMutex: if_let_mutex::IfLetMutex = if_let_mutex::IfLetMutex,
        IfNotElse: if_not_else::IfNotElse = if_not_else::IfNotElse,
        PatternEquality: equatable_if_let::PatternEquality = equatable_if_let::PatternEquality,
        ManualAsyncFn: manual_async_fn::ManualAsyncFn = manual_async_fn::ManualAsyncFn,
        PanicInResultFn: panic_in_result_fn::PanicInResultFn = panic_in_result_fn::PanicInResultFn,
        MacroUseImports: macro_use::MacroUseImports = <macro_use::MacroUseImports>::default(),
        PatternTypeMismatch: pattern_type_mismatch::PatternTypeMismatch = pattern_type_mismatch::PatternTypeMismatch,
        UnwrapInResult: unwrap_in_result::UnwrapInResult = <unwrap_in_result::UnwrapInResult>::default(),
        SemicolonIfNothingReturned: semicolon_if_nothing_returned::SemicolonIfNothingReturned = semicolon_if_nothing_returned::SemicolonIfNothingReturned,
        AsyncYieldsAsync: async_yields_async::AsyncYieldsAsync = async_yields_async::AsyncYieldsAsync,
        DisallowedMacros: disallowed_macros::DisallowedMacros = disallowed_macros::DisallowedMacros::new(tcx, conf, attrs.clone()),
        DisallowedMethods: disallowed_methods::DisallowedMethods = disallowed_methods::DisallowedMethods::new(tcx, conf),
        EmptyDrop: empty_drop::EmptyDrop = empty_drop::EmptyDrop,
        StrToString: strings::StrToString = strings::StrToString,
        ZeroSizedMapValues: zero_sized_map_values::ZeroSizedMapValues = zero_sized_map_values::ZeroSizedMapValues,
        VecInitThenPush: vec_init_then_push::VecInitThenPush = <vec_init_then_push::VecInitThenPush>::default(),
        RedundantSlicing: redundant_slicing::RedundantSlicing = redundant_slicing::RedundantSlicing,
        FromStrRadix10: from_str_radix_10::FromStrRadix10 = from_str_radix_10::FromStrRadix10,
        IfThenSomeElseNone: if_then_some_else_none::IfThenSomeElseNone = if_then_some_else_none::IfThenSomeElseNone::new(conf),
        BoolAssertComparison: bool_assert_comparison::BoolAssertComparison = bool_assert_comparison::BoolAssertComparison,
        UnusedAsync: unused_async::UnusedAsync = <unused_async::UnusedAsync>::default(),
        DisallowedTypes: disallowed_types::DisallowedTypes = disallowed_types::DisallowedTypes::new(tcx, conf),
        ImportRename: missing_enforced_import_rename::ImportRename = missing_enforced_import_rename::ImportRename::new(tcx, conf),
        StrlenOnCStrings: strlen_on_c_strings::StrlenOnCStrings = strlen_on_c_strings::StrlenOnCStrings::new(conf),
        SelfNamedConstructors: self_named_constructors::SelfNamedConstructors = self_named_constructors::SelfNamedConstructors,
        IterNotReturningIterator: iter_not_returning_iterator::IterNotReturningIterator = iter_not_returning_iterator::IterNotReturningIterator,
        ManualAssert: manual_assert::ManualAssert = manual_assert::ManualAssert,
        NonSendFieldInSendTy: non_send_fields_in_send_ty::NonSendFieldInSendTy = non_send_fields_in_send_ty::NonSendFieldInSendTy::new(conf),
        UndocumentedUnsafeBlocks: undocumented_unsafe_blocks::UndocumentedUnsafeBlocks = undocumented_unsafe_blocks::UndocumentedUnsafeBlocks::new(conf),
        FormatArgs: format_args::FormatArgs<'tcx> = format_args::FormatArgs::new(tcx, conf, format_args.clone()),
        TrailingEmptyArray: trailing_empty_array::TrailingEmptyArray = trailing_empty_array::TrailingEmptyArray,
        NeedlessLateInit: needless_late_init::NeedlessLateInit = needless_late_init::NeedlessLateInit,
        ReturnSelfNotMustUse: return_self_not_must_use::ReturnSelfNotMustUse = return_self_not_must_use::ReturnSelfNotMustUse,
        NumberedFields: init_numbered_fields::NumberedFields = init_numbered_fields::NumberedFields,
        ManualBits: manual_bits::ManualBits = manual_bits::ManualBits::new(conf),
        DefaultUnionRepresentation: default_union_representation::DefaultUnionRepresentation = default_union_representation::DefaultUnionRepresentation,
        OnlyUsedInRecursion: only_used_in_recursion::OnlyUsedInRecursion = <only_used_in_recursion::OnlyUsedInRecursion>::default(),
        DbgMacro: dbg_macro::DbgMacro = dbg_macro::DbgMacro::new(conf),
        Write: write::Write = write::Write::new(conf, format_args.clone()),
        Cargo: cargo::Cargo = cargo::Cargo::new(conf),
        EmptyWithBrackets: empty_with_brackets::EmptyWithBrackets = empty_with_brackets::EmptyWithBrackets::default(),
        UnnecessaryOwnedEmptyStrings: unnecessary_owned_empty_strings::UnnecessaryOwnedEmptyStrings = unnecessary_owned_empty_strings::UnnecessaryOwnedEmptyStrings,
        FormatPushString: format_push_string::FormatPushString = format_push_string::FormatPushString::new(format_args.clone()),
        LargeIncludeFile: large_include_file::LargeIncludeFile = large_include_file::LargeIncludeFile::new(conf),
        TrimSplitWhitespace: strings::TrimSplitWhitespace = strings::TrimSplitWhitespace,
        RcCloneInVecInit: rc_clone_in_vec_init::RcCloneInVecInit = rc_clone_in_vec_init::RcCloneInVecInit,
        SwapPtrToRef: swap_ptr_to_ref::SwapPtrToRef = swap_ptr_to_ref::SwapPtrToRef,
        TypeParamMismatch: mismatching_type_param_order::TypeParamMismatch = mismatching_type_param_order::TypeParamMismatch,
        ReadZeroByteVec: read_zero_byte_vec::ReadZeroByteVec = read_zero_byte_vec::ReadZeroByteVec,
        DefaultIterEmpty: default_instead_of_iter_empty::DefaultIterEmpty = default_instead_of_iter_empty::DefaultIterEmpty,
        ManualRemEuclid: manual_rem_euclid::ManualRemEuclid = manual_rem_euclid::ManualRemEuclid::new(conf),
        ManualRetain: manual_retain::ManualRetain = manual_retain::ManualRetain::new(conf),
        ManualRotate: manual_rotate::ManualRotate = manual_rotate::ManualRotate,
        Operators: operators::Operators = operators::Operators::new(conf),
        StdReexports: std_instead_of_core::StdReexports = std_instead_of_core::StdReexports::new(conf),
        UncheckedTimeSubtraction: time_subtraction::UncheckedTimeSubtraction = time_subtraction::UncheckedTimeSubtraction::new(conf),
        PartialeqToNone: partialeq_to_none::PartialeqToNone = partialeq_to_none::PartialeqToNone,
        ManualAbsDiff: manual_abs_diff::ManualAbsDiff = manual_abs_diff::ManualAbsDiff::new(conf),
        ManualClamp: manual_clamp::ManualClamp = manual_clamp::ManualClamp::new(conf),
        ManualStringNew: manual_string_new::ManualStringNew = manual_string_new::ManualStringNew,
        UnusedPeekable: unused_peekable::UnusedPeekable = unused_peekable::UnusedPeekable,
        BoolToIntWithIf: bool_to_int_with_if::BoolToIntWithIf = bool_to_int_with_if::BoolToIntWithIf,
        BoxDefault: box_default::BoxDefault = box_default::BoxDefault,
        ImplicitSaturatingAdd: implicit_saturating_add::ImplicitSaturatingAdd = implicit_saturating_add::ImplicitSaturatingAdd,
        MissingTraitMethods: missing_trait_methods::MissingTraitMethods = missing_trait_methods::MissingTraitMethods,
        FromRawWithVoidPtr: from_raw_with_void_ptr::FromRawWithVoidPtr = from_raw_with_void_ptr::FromRawWithVoidPtr,
        ConfusingXorAndPow: suspicious_xor_used_as_pow::ConfusingXorAndPow = suspicious_xor_used_as_pow::ConfusingXorAndPow,
        ManualIsAsciiCheck: manual_is_ascii_check::ManualIsAsciiCheck = manual_is_ascii_check::ManualIsAsciiCheck::new(conf),
        SemicolonBlock: semicolon_block::SemicolonBlock = semicolon_block::SemicolonBlock::new(conf),
        PermissionsSetReadonlyFalse: permissions_set_readonly_false::PermissionsSetReadonlyFalse = permissions_set_readonly_false::PermissionsSetReadonlyFalse,
        SizeOfRef: size_of_ref::SizeOfRef = size_of_ref::SizeOfRef,
        MultipleUnsafeOpsPerBlock: multiple_unsafe_ops_per_block::MultipleUnsafeOpsPerBlock = multiple_unsafe_ops_per_block::MultipleUnsafeOpsPerBlock,
        ExtraUnusedTypeParameters: extra_unused_type_parameters::ExtraUnusedTypeParameters = extra_unused_type_parameters::ExtraUnusedTypeParameters::new(conf),
        NoMangleWithRustAbi: no_mangle_with_rust_abi::NoMangleWithRustAbi = no_mangle_with_rust_abi::NoMangleWithRustAbi,
        CollectionIsNeverRead: collection_is_never_read::CollectionIsNeverRead = collection_is_never_read::CollectionIsNeverRead,
        MissingAssertMessage: missing_assert_message::MissingAssertMessage = missing_assert_message::MissingAssertMessage,
        NeedlessMaybeSized: needless_maybe_sized::NeedlessMaybeSized = needless_maybe_sized::NeedlessMaybeSized,
        RedundantAsyncBlock: redundant_async_block::RedundantAsyncBlock = redundant_async_block::RedundantAsyncBlock,
        ManualMainSeparatorStr: manual_main_separator_str::ManualMainSeparatorStr = manual_main_separator_str::ManualMainSeparatorStr::new(conf),
        UnnecessaryStruct: unnecessary_struct_initialization::UnnecessaryStruct = unnecessary_struct_initialization::UnnecessaryStruct,
        UnnecessaryBoxReturns: unnecessary_box_returns::UnnecessaryBoxReturns = unnecessary_box_returns::UnnecessaryBoxReturns::new(conf),
        TestsOutsideTestModule: tests_outside_test_module::TestsOutsideTestModule = tests_outside_test_module::TestsOutsideTestModule,
        ManualSliceSizeCalculation: manual_slice_size_calculation::ManualSliceSizeCalculation = manual_slice_size_calculation::ManualSliceSizeCalculation::new(conf),
        ItemsAfterTestModule: items_after_test_module::ItemsAfterTestModule = items_after_test_module::ItemsAfterTestModule,
        DefaultConstructedUnitStructs: default_constructed_unit_structs::DefaultConstructedUnitStructs = default_constructed_unit_structs::DefaultConstructedUnitStructs,
        MissingFieldsInDebug: missing_fields_in_debug::MissingFieldsInDebug = missing_fields_in_debug::MissingFieldsInDebug,
        EndianBytes: endian_bytes::EndianBytes = endian_bytes::EndianBytes,
        RedundantTypeAnnotations: redundant_type_annotations::RedundantTypeAnnotations = redundant_type_annotations::RedundantTypeAnnotations,
        ArcWithNonSendSync: arc_with_non_send_sync::ArcWithNonSendSync = arc_with_non_send_sync::ArcWithNonSendSync,
        NeedlessIfs: needless_ifs::NeedlessIfs = needless_ifs::NeedlessIfs,
        MinIdentChars: min_ident_chars::MinIdentChars = min_ident_chars::MinIdentChars::new(conf),
        LargeStackFrames: large_stack_frames::LargeStackFrames = large_stack_frames::LargeStackFrames::new(conf),
        SingleRangeInVecInit: single_range_in_vec_init::SingleRangeInVecInit = single_range_in_vec_init::SingleRangeInVecInit,
        NeedlessPassByRefMut: needless_pass_by_ref_mut::NeedlessPassByRefMut<'tcx> = needless_pass_by_ref_mut::NeedlessPassByRefMut::new(conf),
        NonCanonicalImpls: non_canonical_impls::NonCanonicalImpls = non_canonical_impls::NonCanonicalImpls::new(tcx),
        SingleCallFn: single_call_fn::SingleCallFn = single_call_fn::SingleCallFn::new(conf),
        LegacyNumericConstants: legacy_numeric_constants::LegacyNumericConstants = legacy_numeric_constants::LegacyNumericConstants::new(conf),
        ManualRangePatterns: manual_range_patterns::ManualRangePatterns = manual_range_patterns::ManualRangePatterns,
        TupleArrayConversions: tuple_array_conversions::TupleArrayConversions = tuple_array_conversions::TupleArrayConversions::new(conf),
        ManualFloatMethods: manual_float_methods::ManualFloatMethods = manual_float_methods::ManualFloatMethods::new(conf),
        FourForwardSlashes: four_forward_slashes::FourForwardSlashes = four_forward_slashes::FourForwardSlashes,
        ErrorImplError: error_impl_error::ErrorImplError = error_impl_error::ErrorImplError,
        AbsolutePaths: absolute_paths::AbsolutePaths = absolute_paths::AbsolutePaths::new(conf),
        RedundantLocals: redundant_locals::RedundantLocals = redundant_locals::RedundantLocals,
        IgnoredUnitPatterns: ignored_unit_patterns::IgnoredUnitPatterns = ignored_unit_patterns::IgnoredUnitPatterns,
        ReserveAfterInitialization: reserve_after_initialization::ReserveAfterInitialization = <reserve_after_initialization::ReserveAfterInitialization>::default(),
        ImpliedBoundsInImpls: implied_bounds_in_impls::ImpliedBoundsInImpls = implied_bounds_in_impls::ImpliedBoundsInImpls,
        MissingAssertsForIndexing: missing_asserts_for_indexing::MissingAssertsForIndexing = missing_asserts_for_indexing::MissingAssertsForIndexing,
        UnnecessaryMapOnConstructor: unnecessary_map_on_constructor::UnnecessaryMapOnConstructor = unnecessary_map_on_constructor::UnnecessaryMapOnConstructor,
        NeedlessBorrowsForGenericArgs: needless_borrows_for_generic_args::NeedlessBorrowsForGenericArgs<'tcx> = needless_borrows_for_generic_args::NeedlessBorrowsForGenericArgs::new( conf, ),
        ManualHashOne: manual_hash_one::ManualHashOne = manual_hash_one::ManualHashOne::new(conf),
        IterWithoutIntoIter: iter_without_into_iter::IterWithoutIntoIter = iter_without_into_iter::IterWithoutIntoIter,
        PathbufThenPush: pathbuf_init_then_push::PathbufThenPush<'tcx> = <pathbuf_init_then_push::PathbufThenPush<'_>>::default(),
        IterOverHashType: iter_over_hash_type::IterOverHashType = iter_over_hash_type::IterOverHashType,
        ImplHashWithBorrowStrBytes: impl_hash_with_borrow_str_and_bytes::ImplHashWithBorrowStrBytes = impl_hash_with_borrow_str_and_bytes::ImplHashWithBorrowStrBytes,
        RepeatVecWithCapacity: repeat_vec_with_capacity::RepeatVecWithCapacity = repeat_vec_with_capacity::RepeatVecWithCapacity::new(conf),
        UninhabitedReferences: uninhabited_references::UninhabitedReferences = uninhabited_references::UninhabitedReferences,
        IneffectiveOpenOptions: ineffective_open_options::IneffectiveOpenOptions = ineffective_open_options::IneffectiveOpenOptions,
        UnconditionalRecursion: unconditional_recursion::UnconditionalRecursion = <unconditional_recursion::UnconditionalRecursion>::default(),
        PubUnderscoreFields: pub_underscore_fields::PubUnderscoreFields = pub_underscore_fields::PubUnderscoreFields::new(conf),
        MissingConstForThreadLocal: missing_const_for_thread_local::MissingConstForThreadLocal = missing_const_for_thread_local::MissingConstForThreadLocal::new(conf),
        IncompatibleMsrv: incompatible_msrv::IncompatibleMsrv = incompatible_msrv::IncompatibleMsrv::new(tcx, conf),
        ToStringTraitImpl: to_string_trait_impl::ToStringTraitImpl = to_string_trait_impl::ToStringTraitImpl,
        AssigningClones: assigning_clones::AssigningClones = assigning_clones::AssigningClones::new(conf),
        ZeroRepeatSideEffects: zero_repeat_side_effects::ZeroRepeatSideEffects = zero_repeat_side_effects::ZeroRepeatSideEffects,
        ExprMetavarsInUnsafe: macro_metavars_in_unsafe::ExprMetavarsInUnsafe = macro_metavars_in_unsafe::ExprMetavarsInUnsafe::new(conf),
        StringPatterns: string_patterns::StringPatterns = string_patterns::StringPatterns::new(conf),
        SetContainsOrInsert: set_contains_or_insert::SetContainsOrInsert = set_contains_or_insert::SetContainsOrInsert,
        ZombieProcesses: zombie_processes::ZombieProcesses = zombie_processes::ZombieProcesses,
        PointersInNomemAsmBlock: pointers_in_nomem_asm_block::PointersInNomemAsmBlock = pointers_in_nomem_asm_block::PointersInNomemAsmBlock,
        ManualIsPowerOfTwo: manual_is_power_of_two::ManualIsPowerOfTwo = manual_is_power_of_two::ManualIsPowerOfTwo::new(conf),
        NonZeroSuggestions: non_zero_suggestions::NonZeroSuggestions = non_zero_suggestions::NonZeroSuggestions,
        LiteralStringWithFormattingArg: literal_string_with_formatting_args::LiteralStringWithFormattingArg = literal_string_with_formatting_args::LiteralStringWithFormattingArg,
        UnusedTraitNames: unused_trait_names::UnusedTraitNames = unused_trait_names::UnusedTraitNames::new(conf),
        ManualIgnoreCaseCmp: manual_ignore_case_cmp::ManualIgnoreCaseCmp = manual_ignore_case_cmp::ManualIgnoreCaseCmp,
        UnnecessaryLiteralBound: unnecessary_literal_bound::UnnecessaryLiteralBound = unnecessary_literal_bound::UnnecessaryLiteralBound,
        ArbitrarySourceItemOrdering: arbitrary_source_item_ordering::ArbitrarySourceItemOrdering = arbitrary_source_item_ordering::ArbitrarySourceItemOrdering::new(conf),
        UselessConcat: useless_concat::UselessConcat = useless_concat::UselessConcat,
        UnneededStructPattern: unneeded_struct_pattern::UnneededStructPattern = unneeded_struct_pattern::UnneededStructPattern,
        UnnecessarySemicolon: unnecessary_semicolon::UnnecessarySemicolon = <unnecessary_semicolon::UnnecessarySemicolon>::default(),
        NonStdLazyStatic: non_std_lazy_statics::NonStdLazyStatic = non_std_lazy_statics::NonStdLazyStatic::new(conf),
        ManualOptionAsSlice: manual_option_as_slice::ManualOptionAsSlice = manual_option_as_slice::ManualOptionAsSlice::new(conf),
        SingleOptionMap: single_option_map::SingleOptionMap = single_option_map::SingleOptionMap,
        RedundantTestPrefix: redundant_test_prefix::RedundantTestPrefix = redundant_test_prefix::RedundantTestPrefix,
        ClonedRefToSliceRefs: cloned_ref_to_slice_refs::ClonedRefToSliceRefs<'tcx> = cloned_ref_to_slice_refs::ClonedRefToSliceRefs::new(conf),
        InfallibleTryFrom: infallible_try_from::InfallibleTryFrom = infallible_try_from::InfallibleTryFrom,
        CoerceContainerToAny: coerce_container_to_any::CoerceContainerToAny = coerce_container_to_any::CoerceContainerToAny,
        ToplevelRefArg: toplevel_ref_arg::ToplevelRefArg = toplevel_ref_arg::ToplevelRefArg,
        VolatileComposites: volatile_composites::VolatileComposites = volatile_composites::VolatileComposites,
        ReplaceBox: replace_box::ReplaceBox = <replace_box::ReplaceBox>::default(),
        DisallowedFields: disallowed_fields::DisallowedFields = disallowed_fields::DisallowedFields::new(tcx, conf),
        ManualIlog2: manual_ilog2::ManualIlog2 = manual_ilog2::ManualIlog2::new(conf),
        SameLengthAndCapacity: same_length_and_capacity::SameLengthAndCapacity = same_length_and_capacity::SameLengthAndCapacity,
        DurationSuboptimalUnits: duration_suboptimal_units::DurationSuboptimalUnits = duration_suboptimal_units::DurationSuboptimalUnits::new(tcx, conf),
        ManualTake: manual_take::ManualTake = manual_take::ManualTake::new(conf),
        ManualCheckedOps: manual_checked_ops::ManualCheckedOps = manual_checked_ops::ManualCheckedOps,
        ManualPopIf: manual_pop_if::ManualPopIf = manual_pop_if::ManualPopIf::new(tcx, conf),
        ManualNoopWaker: manual_noop_waker::ManualNoopWaker = manual_noop_waker::ManualNoopWaker::new(conf),
        ByteCharSlice: byte_char_slices::ByteCharSlice = byte_char_slices::ByteCharSlice,
        ManualAssertEq: manual_assert_eq::ManualAssertEq = manual_assert_eq::ManualAssertEq,
        // add late passes here, used by `cargo dev new_lint`
    ]]
);
