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
extern crate rustc_macros;
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
use rustc_lint::{EarlyLintPassFactory, LateLintPassFactory, Lint};
use std::rc::Rc;
use std::cell::RefCell;
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
    store.register_pre_expansion_pass(
        Box::new(move || Rc::new(RefCell::new(attrs::EarlyAttributes::new(conf))))
    );

    let format_args_storage = FormatArgsStorage::default();
    let attr_storage = AttrStorage::default();

    macro_rules! b { ($e:expr) => { Box::new(|| Rc::new(RefCell::new($e))) } }
    macro_rules! bm { ($e:expr) => { Box::new(move || Rc::new(RefCell::new($e))) } }

    let early_lints: [EarlyLintPassFactory; _] = [
        {
            let format_args = format_args_storage.clone();
            bm!(utils::format_args_collector::FormatArgsCollector::new(format_args.clone()))
        },
        {
            let attrs = attr_storage.clone();
            bm!(AttrCollector::new(attrs.clone()))
        },
        bm!(attrs::PostExpansionEarlyAttributes::new(conf)),
        b!(unnecessary_self_imports::UnnecessarySelfImports),
        bm!(redundant_static_lifetimes::RedundantStaticLifetimes::new(conf)),
        bm!(redundant_field_names::RedundantFieldNames::new(conf)),
        bm!(unnested_or_patterns::UnnestedOrPatterns::new(conf)),
        b!(functions::EarlyFunctions),
        bm!(doc::Documentation::new(conf)),
        b!(suspicious_operation_groupings::SuspiciousOperationGroupings),
        b!(double_parens::DoubleParens),
        b!(unsafe_removed_from_name::UnsafeNameRemoval),
        b!(else_if_without_else::ElseIfWithoutElse),
        b!(int_plus_one::IntPlusOne),
        b!(formatting::Formatting),
        b!(misc_early::MiscEarlyLints),
        b!(unused_unit::UnusedUnit),
        b!(precedence::Precedence),
        b!(redundant_else::RedundantElse),
        b!(needless_arbitrary_self_type::NeedlessArbitrarySelfType),
        bm!(literal_representation::LiteralDigitGrouping::new(conf)),
        bm!(literal_representation::DecimalLiteralRepresentation::new(conf)),
        b!(tabs_in_doc_comments::TabsInDocComments),
        b!(single_component_path_imports::SingleComponentPathImports::default()),
        b!(option_env_unwrap::OptionEnvUnwrap),
        bm!(non_expressive_names::NonExpressiveNames::new(conf)),
        bm!(nonstandard_macro_braces::MacroBraces::new(conf)),
        b!(asm_syntax::InlineAsmX86AttSyntax),
        b!(asm_syntax::InlineAsmX86IntelSyntax),
        b!(module_style::ModStyle::default()),
        bm!(disallowed_script_idents::DisallowedScriptIdents::new(conf)),
        b!(octal_escapes::OctalEscapes),
        b!(single_char_lifetime_names::SingleCharLifetimeNames),
        b!(crate_in_macro_def::CrateInMacroDef),
        b!(pub_use::PubUse),
        bm!(large_include_file::LargeIncludeFile::new(conf)),
        b!(duplicate_mod::DuplicateMod::default()),
        b!(unused_rounding::UnusedRounding),
        bm!(almost_complete_range::AlmostCompleteRange::new(conf)),
        b!(multi_assignments::MultiAssignments),
        b!(partial_pub_fields::PartialPubFields),
        b!(let_with_type_underscore::UnderscoreTyped),
        bm!(excessive_nesting::ExcessiveNesting::new(conf)),
        b!(ref_patterns::RefPatterns),
        b!(needless_else::NeedlessElse),
        bm!(raw_strings::RawStrings::new(conf)),
        b!(visibility::Visibility),
        b!(multiple_bound_locations::MultipleBoundLocations),
        b!(field_scoped_visibility_modifiers::FieldScopedVisibilityModifiers),
        b!(cfg_not_test::CfgNotTest),
        b!(empty_line_after::EmptyLineAfter::new()),
        b!(inline_trait_bounds::InlineTraitBounds),
        // add early passes here, used by `cargo dev new_lint`
    ];
    store.early_passes.extend(early_lints);

    macro_rules! b { ($e:expr) => { Box::new(|_| Rc::new(RefCell::new($e))) } }
    macro_rules! bm { ($e:expr) => { Box::new(move |_| Rc::new(RefCell::new($e))) } }
    macro_rules! bmt { ($e:expr) => { Box::new(move |tcx| Rc::new(RefCell::new($e(tcx)))) } }

    #[expect(clippy::type_complexity)]
    let late_lints: [LateLintPassFactory; _] = [
        bm!(operators::arithmetic_side_effects::ArithmeticSideEffects::new(conf)),
        b!(utils::dump_hir::DumpHir),
        b!(utils::author::Author),
        bmt!(|tcx| await_holding_invalid::AwaitHolding::new(tcx, conf)),
        b!(serde_api::SerdeApi),
        bm!(types::Types::new(conf)),
        bm!(booleans::NonminimalBool::new(conf)),
        b!(enum_clike::UnportableVariant),
        bm!(float_literal::FloatLiteral::new(conf)),
        b!(ptr::Ptr),
        b!(needless_bool::NeedlessBool),
        b!(bool_comparison::BoolComparison),
        b!(needless_for_each::NeedlessForEach),
        b!(misc::LintPass),
        b!(eta_reduction::EtaReduction),
        b!(mut_mut::MutMut::default()),
        b!(unnecessary_mut_passed::UnnecessaryMutPassed),
        b!(significant_drop_tightening::SignificantDropTightening::default()),
        bm!(len_zero::LenZero::new(conf)),
        b!(len_without_is_empty::LenWithoutIsEmpty),
        bm!(attrs::Attributes::new(conf)),
        b!(blocks_in_conditions::BlocksInConditions),
        b!(unicode::Unicode),
        b!(uninit_vec::UninitVec),
        b!(unit_return_expecting_ord::UnitReturnExpectingOrd),
        b!(strings::StringAdd),
        b!(implicit_return::ImplicitReturn),
        bm!(implicit_saturating_sub::ImplicitSaturatingSub::new(conf)),
        b!(default_numeric_fallback::DefaultNumericFallback),
        b!(non_octal_unix_permissions::NonOctalUnixPermissions),
        bm!(approx_const::ApproxConstant::new(conf)),
        bm!(matches::Matches::new(conf)),
        bm!(manual_non_exhaustive::ManualNonExhaustive::new(conf)),
        bm!(manual_strip::ManualStrip::new(conf)),
        bm!(checked_conversions::CheckedConversions::new(conf)),
        bm!(mem_replace::MemReplace::new(conf)),
        bm!(ranges::Ranges::new(conf)),
        bm!(from_over_into::FromOverInto::new(conf)),
        bm!(use_self::UseSelf::new(conf)),
        bm!(missing_const_for_fn::MissingConstForFn::new(conf)),
        b!(needless_question_mark::NeedlessQuestionMark),
        bm!(casts::Casts::new(conf)),
        b!(size_of_in_element_count::SizeOfInElementCount),
        b!(same_name_method::SameNameMethod),
        bm!(index_refutable_slice::IndexRefutableSlice::new(conf)),
        b!(shadow::Shadow::default()),
        bm!(inconsistent_struct_constructor::InconsistentStructConstructor::new(conf)),
        {
            let format_args = format_args_storage.clone();
            bm!(methods::Methods::new(conf, format_args.clone()))
        },
        {
            let format_args = format_args_storage.clone();
            bm!(unit_types::UnitTypes::new(format_args.clone()))
        },
        bm!(loops::Loops::new(conf)),
        b!(main_recursion::MainRecursion::default()),
        bm!(lifetimes::Lifetimes::new(conf)),
        b!(entry::HashMapPass),
        b!(minmax::MinMaxPass),
        b!(zero_div_zero::ZeroDiv),
        b!(mutex_atomic::Mutex),
        b!(needless_update::NeedlessUpdate),
        b!(needless_borrowed_ref::NeedlessBorrowedRef),
        b!(borrow_deref_ref::BorrowDerefRef),
        b!(no_effect::NoEffect::default()),
        b!(temporary_assignment::TemporaryAssignment),
        bm!(transmute::Transmute::new(conf)),
        bm!(cognitive_complexity::CognitiveComplexity::new(conf)),
        bm!(escape::BoxedLocal::new(conf)),
        bm!(useless_vec::UselessVec::new(conf)),
        bm!(panic_unimplemented::PanicUnimplemented::new(conf)),
        b!(strings::StringLitAsBytes),
        b!(derive::Derive),
        bm!(derivable_impls::DerivableImpls::new(conf)),
        b!(drop_forget_ref::DropForgetRef),
        b!(empty_enums::EmptyEnums),
        b!(regex::Regex::default()),
        bmt!(|tcx| ifs::CopyAndPaste::new(tcx, conf)),
        b!(copy_iterator::CopyIterator),
        {
            let format_args = format_args_storage.clone();
            bm!(format::UselessFormat::new(format_args.clone()))
        },
        b!(swap::Swap),
        b!(panicking_overflow_checks::PanickingOverflowChecks),
        b!(new_without_default::NewWithoutDefault::default()),
        bm!(disallowed_names::DisallowedNames::new(conf)),
        bmt!(|tcx| functions::Functions::new(tcx, conf)),
        bm!(doc::Documentation::new(conf)),
        b!(neg_multiply::NegMultiply),
        b!(let_if_seq::LetIfSeq),
        b!(mixed_read_write_in_expression::EvalOrderDependence),
        bm!(missing_doc::MissingDoc::new(conf)),
        b!(missing_inline::MissingInline),
        b!(exhaustive_items::ExhaustiveItems),
        b!(unused_result_ok::UnusedResultOk),
        b!(match_result_ok::MatchResultOk),
        b!(partialeq_ne_impl::PartialEqNeImpl),
        b!(unused_io_amount::UnusedIoAmount),
        bm!(large_enum_variant::LargeEnumVariant::new(conf)),
        {
            let format_args = format_args_storage.clone();
            bm!(explicit_write::ExplicitWrite::new(format_args.clone()))
        },
        b!(needless_pass_by_value::NeedlessPassByValue),
        bmt!(|tcx| pass_by_ref_or_value::PassByRefOrValue::new(tcx, conf)),
        b!(ref_option_ref::RefOptionRef),
        b!(infinite_iter::InfiniteIter),
        b!(inline_fn_without_body::InlineFnWithoutBody),
        b!(useless_conversion::UselessConversion::default()),
        b!(implicit_hasher::ImplicitHasher),
        b!(fallible_impl_from::FallibleImplFrom),
        bm!(question_mark::QuestionMark::new(conf)),
        b!(question_mark_used::QuestionMarkUsed),
        b!(suspicious_trait_impl::SuspiciousImpl),
        b!(map_unit_fn::MapUnit),
        bm!(inherent_impl::MultipleInherentImpl::new(conf)),
        b!(neg_cmp_op_on_partial_ord::NoNegCompOpForPartialOrd),
        bm!(unwrap::Unwrap::new(conf)),
        bm!(indexing_slicing::IndexingSlicing::new(conf)),
        bmt!(|tcx| non_copy_const::NonCopyConst::new(tcx, conf)),
        b!(redundant_clone::RedundantClone),
        b!(slow_vector_initialization::SlowVectorInit),
        bm!(unnecessary_wraps::UnnecessaryWraps::new(conf)),
        bm!(assertions_on_constants::AssertionsOnConstants::new(conf)),
        b!(assertions_on_result_states::AssertionsOnResultStates),
        b!(inherent_to_string::InherentToString),
        bm!(trait_bounds::TraitBounds::new(conf)),
        b!(comparison_chain::ComparisonChain),
        bmt!(|tcx| mut_key::MutableKeyType::new(tcx, conf)),
        b!(reference::DerefAddrOf),
        {                                                                  
            let format_args = format_args_storage.clone();                 
            bm!(format_impl::FormatImpl::new(format_args.clone()))
        },    
        b!(redundant_closure_call::RedundantClosureCall),
        b!(unused_unit::UnusedUnit),
        b!(returns::Return),
        bm!(collapsible_if::CollapsibleIf::new(conf)),
        b!(items_after_statements::ItemsAfterStatements),
        b!(needless_parens_on_range_literals::NeedlessParensOnRangeLiterals),
        b!(needless_continue::NeedlessContinue),
        b!(create_dir::CreateDir),
        bm!(item_name_repetitions::ItemNameRepetitions::new(conf)),
        bm!(upper_case_acronyms::UpperCaseAcronyms::new(conf)),
        b!(default::Default::default()),
        bm!(unused_self::UnusedSelf::new(conf)),
        b!(mutable_debug_assertion::DebugAssertWithMutCall),
        b!(exit::Exit),
        bm!(to_digit_is_some::ToDigitIsSome::new(conf)),
        bm!(large_stack_arrays::LargeStackArrays::new(conf)),
        bm!(large_const_arrays::LargeConstArrays::new(conf)),
        b!(floating_point_arithmetic::FloatingPointArithmetic),
        b!(as_conversions::AsConversions),
        b!(let_underscore::LetUnderscore),
        bm!(excessive_bools::ExcessiveBools::new(conf)),
        bm!(wildcard_imports::WildcardImports::new(conf)),
        b!(redundant_pub_crate::RedundantPubCrate::default()),
        b!(dereference::Dereferencing::default()),
        b!(option_if_let_else::OptionIfLetElse),
        b!(future_not_send::FutureNotSend),
        bm!(large_futures::LargeFuture::new(conf)),
        b!(if_let_mutex::IfLetMutex),
        b!(if_not_else::IfNotElse),
        b!(equatable_if_let::PatternEquality),
        b!(manual_async_fn::ManualAsyncFn),
        b!(panic_in_result_fn::PanicInResultFn),
        b!(macro_use::MacroUseImports::default()),
        b!(pattern_type_mismatch::PatternTypeMismatch),
        b!(unwrap_in_result::UnwrapInResult::default()),
        b!(semicolon_if_nothing_returned::SemicolonIfNothingReturned),
        b!(async_yields_async::AsyncYieldsAsync),
        {                                                                  
            let attrs = attr_storage.clone();                              
            bmt!(|tcx| disallowed_macros::DisallowedMacros::new(tcx, conf, attrs.clone()))
        },
        bmt!(|tcx| disallowed_methods::DisallowedMethods::new(tcx, conf)),
        b!(empty_drop::EmptyDrop),
        b!(strings::StrToString),
        b!(zero_sized_map_values::ZeroSizedMapValues),
        b!(vec_init_then_push::VecInitThenPush::default()),
        b!(redundant_slicing::RedundantSlicing),
        b!(from_str_radix_10::FromStrRadix10),
        bm!(if_then_some_else_none::IfThenSomeElseNone::new(conf)),
        b!(bool_assert_comparison::BoolAssertComparison),
        b!(unused_async::UnusedAsync::default()),
        bmt!(|tcx| disallowed_types::DisallowedTypes::new(tcx, conf)),
        bmt!(|tcx| missing_enforced_import_rename::ImportRename::new(tcx, conf)),
        bm!(strlen_on_c_strings::StrlenOnCStrings::new(conf)),
        b!(self_named_constructors::SelfNamedConstructors),
        b!(iter_not_returning_iterator::IterNotReturningIterator),
        b!(manual_assert::ManualAssert),
        bm!(non_send_fields_in_send_ty::NonSendFieldInSendTy::new(conf)),
        bm!(undocumented_unsafe_blocks::UndocumentedUnsafeBlocks::new(conf)),
        {
            let format_args = format_args_storage.clone();
            bmt!(|tcx| format_args::FormatArgs::new(tcx, conf, format_args.clone()))
        },
        b!(trailing_empty_array::TrailingEmptyArray),
        b!(needless_late_init::NeedlessLateInit),
        b!(return_self_not_must_use::ReturnSelfNotMustUse),
        b!(init_numbered_fields::NumberedFields),
        bm!(manual_bits::ManualBits::new(conf)),
        b!(default_union_representation::DefaultUnionRepresentation),
        b!(only_used_in_recursion::OnlyUsedInRecursion::default()),
        bm!(dbg_macro::DbgMacro::new(conf)),
        {
            let format_args = format_args_storage.clone();
            bm!(write::Write::new(conf, format_args.clone()))
        },
        bm!(cargo::Cargo::new(conf)),
        b!(empty_with_brackets::EmptyWithBrackets::default()),
        b!(unnecessary_owned_empty_strings::UnnecessaryOwnedEmptyStrings),
        {
            let format_args = format_args_storage.clone();
            bm!(format_push_string::FormatPushString::new(format_args.clone()))
        },
        bm!(large_include_file::LargeIncludeFile::new(conf)),
        b!(strings::TrimSplitWhitespace),
        b!(rc_clone_in_vec_init::RcCloneInVecInit),
        b!(swap_ptr_to_ref::SwapPtrToRef),
        b!(mismatching_type_param_order::TypeParamMismatch),
        b!(read_zero_byte_vec::ReadZeroByteVec),
        b!(default_instead_of_iter_empty::DefaultIterEmpty),
        bm!(manual_rem_euclid::ManualRemEuclid::new(conf)),
        bm!(manual_retain::ManualRetain::new(conf)),
        b!(manual_rotate::ManualRotate),
        bm!(operators::Operators::new(conf)),
        bm!(std_instead_of_core::StdReexports::new(conf)),
        bm!(time_subtraction::UncheckedTimeSubtraction::new(conf)),
        b!(partialeq_to_none::PartialeqToNone),
        bm!(manual_abs_diff::ManualAbsDiff::new(conf)),
        bm!(manual_clamp::ManualClamp::new(conf)),
        b!(manual_string_new::ManualStringNew),
        b!(unused_peekable::UnusedPeekable),
        b!(bool_to_int_with_if::BoolToIntWithIf),
        b!(box_default::BoxDefault),
        b!(implicit_saturating_add::ImplicitSaturatingAdd),
        b!(missing_trait_methods::MissingTraitMethods),
        b!(from_raw_with_void_ptr::FromRawWithVoidPtr),
        b!(suspicious_xor_used_as_pow::ConfusingXorAndPow),
        bm!(manual_is_ascii_check::ManualIsAsciiCheck::new(conf)),
        bm!(semicolon_block::SemicolonBlock::new(conf)),
        b!(permissions_set_readonly_false::PermissionsSetReadonlyFalse),
        b!(size_of_ref::SizeOfRef),
        b!(multiple_unsafe_ops_per_block::MultipleUnsafeOpsPerBlock),
        bm!(extra_unused_type_parameters::ExtraUnusedTypeParameters::new(conf)),
        b!(no_mangle_with_rust_abi::NoMangleWithRustAbi),
        b!(collection_is_never_read::CollectionIsNeverRead),
        b!(missing_assert_message::MissingAssertMessage),
        b!(needless_maybe_sized::NeedlessMaybeSized),
        b!(redundant_async_block::RedundantAsyncBlock),
        bm!(manual_main_separator_str::ManualMainSeparatorStr::new(conf)),
        b!(unnecessary_struct_initialization::UnnecessaryStruct),
        bm!(unnecessary_box_returns::UnnecessaryBoxReturns::new(conf)),
        b!(tests_outside_test_module::TestsOutsideTestModule),
        bm!(manual_slice_size_calculation::ManualSliceSizeCalculation::new(conf)),
        b!(items_after_test_module::ItemsAfterTestModule),
        b!(default_constructed_unit_structs::DefaultConstructedUnitStructs),
        b!(missing_fields_in_debug::MissingFieldsInDebug),
        b!(endian_bytes::EndianBytes),
        b!(redundant_type_annotations::RedundantTypeAnnotations),
        b!(arc_with_non_send_sync::ArcWithNonSendSync),
        b!(needless_ifs::NeedlessIfs),
        bm!(min_ident_chars::MinIdentChars::new(conf)),
        bm!(large_stack_frames::LargeStackFrames::new(conf)),
        b!(single_range_in_vec_init::SingleRangeInVecInit),
        bm!(needless_pass_by_ref_mut::NeedlessPassByRefMut::new(conf)),
        bmt!(|tcx| non_canonical_impls::NonCanonicalImpls::new(tcx)),
        bm!(single_call_fn::SingleCallFn::new(conf)),
        bm!(legacy_numeric_constants::LegacyNumericConstants::new(conf)),
        b!(manual_range_patterns::ManualRangePatterns),
        bm!(tuple_array_conversions::TupleArrayConversions::new(conf)),
        bm!(manual_float_methods::ManualFloatMethods::new(conf)),
        b!(four_forward_slashes::FourForwardSlashes),
        b!(error_impl_error::ErrorImplError),
        bm!(absolute_paths::AbsolutePaths::new(conf)),
        b!(redundant_locals::RedundantLocals),
        b!(ignored_unit_patterns::IgnoredUnitPatterns),
        b!(reserve_after_initialization::ReserveAfterInitialization::default()),
        b!(implied_bounds_in_impls::ImpliedBoundsInImpls),
        b!(missing_asserts_for_indexing::MissingAssertsForIndexing),
        b!(unnecessary_map_on_constructor::UnnecessaryMapOnConstructor),
        bm!(needless_borrows_for_generic_args::NeedlessBorrowsForGenericArgs::new(conf)),
        bm!(manual_hash_one::ManualHashOne::new(conf)),
        b!(iter_without_into_iter::IterWithoutIntoIter),
        b!(pathbuf_init_then_push::PathbufThenPush::default()),
        b!(iter_over_hash_type::IterOverHashType),
        b!(impl_hash_with_borrow_str_and_bytes::ImplHashWithBorrowStrBytes),
        bm!(repeat_vec_with_capacity::RepeatVecWithCapacity::new(conf)),
        b!(uninhabited_references::UninhabitedReferences),
        b!(ineffective_open_options::IneffectiveOpenOptions),
        b!(unconditional_recursion::UnconditionalRecursion::default()),
        bm!(pub_underscore_fields::PubUnderscoreFields::new(conf)),
        bm!(missing_const_for_thread_local::MissingConstForThreadLocal::new(conf)),
        bmt!(|tcx| incompatible_msrv::IncompatibleMsrv::new(tcx, conf)),
        b!(to_string_trait_impl::ToStringTraitImpl),
        bm!(assigning_clones::AssigningClones::new(conf)),
        b!(zero_repeat_side_effects::ZeroRepeatSideEffects),
        bm!(macro_metavars_in_unsafe::ExprMetavarsInUnsafe::new(conf)),
        bm!(string_patterns::StringPatterns::new(conf)),
        b!(set_contains_or_insert::SetContainsOrInsert),
        b!(zombie_processes::ZombieProcesses),
        b!(pointers_in_nomem_asm_block::PointersInNomemAsmBlock),
        bm!(manual_is_power_of_two::ManualIsPowerOfTwo::new(conf)),
        b!(non_zero_suggestions::NonZeroSuggestions),
        b!(literal_string_with_formatting_args::LiteralStringWithFormattingArg),
        bm!(unused_trait_names::UnusedTraitNames::new(conf)),
        b!(manual_ignore_case_cmp::ManualIgnoreCaseCmp),
        b!(unnecessary_literal_bound::UnnecessaryLiteralBound),
        bm!(arbitrary_source_item_ordering::ArbitrarySourceItemOrdering::new(conf)),
        b!(useless_concat::UselessConcat),
        b!(unneeded_struct_pattern::UnneededStructPattern),
        b!(unnecessary_semicolon::UnnecessarySemicolon::default()),
        bm!(non_std_lazy_statics::NonStdLazyStatic::new(conf)),
        bm!(manual_option_as_slice::ManualOptionAsSlice::new(conf)),
        b!(single_option_map::SingleOptionMap),
        b!(redundant_test_prefix::RedundantTestPrefix),
        bm!(cloned_ref_to_slice_refs::ClonedRefToSliceRefs::new(conf)),
        b!(infallible_try_from::InfallibleTryFrom),
        b!(coerce_container_to_any::CoerceContainerToAny),
        b!(toplevel_ref_arg::ToplevelRefArg),
        b!(volatile_composites::VolatileComposites),
        b!(replace_box::ReplaceBox::default()),
        bmt!(|tcx| disallowed_fields::DisallowedFields::new(tcx, conf)),
        bm!(manual_ilog2::ManualIlog2::new(conf)),
        b!(same_length_and_capacity::SameLengthAndCapacity),
        bmt!(|tcx| duration_suboptimal_units::DurationSuboptimalUnits::new(tcx, conf)),
        bm!(manual_take::ManualTake::new(conf)),
        b!(manual_checked_ops::ManualCheckedOps),
        bmt!(|tcx| manual_pop_if::ManualPopIf::new(tcx, conf)),
        bm!(manual_noop_waker::ManualNoopWaker::new(conf)),
        b!(byte_char_slices::ByteCharSlice),
        b!(manual_assert_eq::ManualAssertEq),
        // add late passes here, used by `cargo dev new_lint`
    ];
    store.late_passes.extend(late_lints);
}
