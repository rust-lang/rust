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
mod if_let_some_result;
mod if_not_else;
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
mod matches;
mod mem_discriminant;
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

    // begin deprecated lints, do not remove this comment, it’s used in `update_lints`
    store.register_removed(
        "clippy::should_assert_eq",
        "`assert!()` will be more flexible with RFC 2011",
    );
    store.register_removed(
        "clippy::extend_from_slice",
        "`.extend_from_slice(_)` is a faster way to extend a Vec by a slice",
    );
    store.register_removed(
        "clippy::range_step_by_zero",
        "`iterator.step_by(0)` panics nowadays",
    );
    store.register_removed(
        "clippy::unstable_as_slice",
        "`Vec::as_slice` has been stabilized in 1.7",
    );
    store.register_removed(
        "clippy::unstable_as_mut_slice",
        "`Vec::as_mut_slice` has been stabilized in 1.7",
    );
    store.register_removed(
        "clippy::misaligned_transmute",
        "this lint has been split into cast_ptr_alignment and transmute_ptr_to_ptr",
    );
    store.register_removed(
        "clippy::assign_ops",
        "using compound assignment operators (e.g., `+=`) is harmless",
    );
    store.register_removed(
        "clippy::if_let_redundant_pattern_matching",
        "this lint has been changed to redundant_pattern_matching",
    );
    store.register_removed(
        "clippy::unsafe_vector_initialization",
        "the replacement suggested by this lint had substantially different behavior",
    );
    store.register_removed(
        "clippy::unused_collect",
        "`collect` has been marked as #[must_use] in rustc and that covers all cases of this lint",
    );
    store.register_removed(
        "clippy::replace_consts",
        "associated-constants `MIN`/`MAX` of integers are preferred to `{min,max}_value()` and module constants",
    );
    store.register_removed(
        "clippy::regex_macro",
        "the regex! macro has been removed from the regex crate in 2018",
    );
    store.register_removed(
        "clippy::find_map",
        "this lint has been replaced by `manual_find_map`, a more specific lint",
    );
    store.register_removed(
        "clippy::filter_map",
        "this lint has been replaced by `manual_filter_map`, a more specific lint",
    );
    store.register_removed(
        "clippy::pub_enum_variant_names",
        "set the `avoid-breaking-exported-api` config option to `false` to enable the `enum_variant_names` lint for public items",
    );
    store.register_removed(
        "clippy::wrong_pub_self_convention",
        "set the `avoid-breaking-exported-api` config option to `false` to enable the `wrong_self_convention` lint for public items",
    );
    // end deprecated lints, do not remove this comment, it’s used in `update_lints`

    // begin register lints, do not remove this comment, it’s used in `update_lints`
    store.register_lints(&[
        #[cfg(feature = "internal-lints")]
        utils::internal_lints::CLIPPY_LINTS_INTERNAL,
        #[cfg(feature = "internal-lints")]
        utils::internal_lints::COLLAPSIBLE_SPAN_LINT_CALLS,
        #[cfg(feature = "internal-lints")]
        utils::internal_lints::COMPILER_LINT_FUNCTIONS,
        #[cfg(feature = "internal-lints")]
        utils::internal_lints::DEFAULT_LINT,
        #[cfg(feature = "internal-lints")]
        utils::internal_lints::IF_CHAIN_STYLE,
        #[cfg(feature = "internal-lints")]
        utils::internal_lints::INTERNING_DEFINED_SYMBOL,
        #[cfg(feature = "internal-lints")]
        utils::internal_lints::INVALID_PATHS,
        #[cfg(feature = "internal-lints")]
        utils::internal_lints::LINT_WITHOUT_LINT_PASS,
        #[cfg(feature = "internal-lints")]
        utils::internal_lints::MATCH_TYPE_ON_DIAGNOSTIC_ITEM,
        #[cfg(feature = "internal-lints")]
        utils::internal_lints::OUTER_EXPN_EXPN_DATA,
        #[cfg(feature = "internal-lints")]
        utils::internal_lints::PRODUCE_ICE,
        #[cfg(feature = "internal-lints")]
        utils::internal_lints::UNNECESSARY_SYMBOL_STR,
        absurd_extreme_comparisons::ABSURD_EXTREME_COMPARISONS,
        approx_const::APPROX_CONSTANT,
        arithmetic::FLOAT_ARITHMETIC,
        arithmetic::INTEGER_ARITHMETIC,
        as_conversions::AS_CONVERSIONS,
        asm_syntax::INLINE_ASM_X86_ATT_SYNTAX,
        asm_syntax::INLINE_ASM_X86_INTEL_SYNTAX,
        assertions_on_constants::ASSERTIONS_ON_CONSTANTS,
        assign_ops::ASSIGN_OP_PATTERN,
        assign_ops::MISREFACTORED_ASSIGN_OP,
        async_yields_async::ASYNC_YIELDS_ASYNC,
        attrs::BLANKET_CLIPPY_RESTRICTION_LINTS,
        attrs::DEPRECATED_CFG_ATTR,
        attrs::DEPRECATED_SEMVER,
        attrs::EMPTY_LINE_AFTER_OUTER_ATTR,
        attrs::INLINE_ALWAYS,
        attrs::MISMATCHED_TARGET_OS,
        attrs::USELESS_ATTRIBUTE,
        await_holding_invalid::AWAIT_HOLDING_LOCK,
        await_holding_invalid::AWAIT_HOLDING_REFCELL_REF,
        bit_mask::BAD_BIT_MASK,
        bit_mask::INEFFECTIVE_BIT_MASK,
        bit_mask::VERBOSE_BIT_MASK,
        blacklisted_name::BLACKLISTED_NAME,
        blocks_in_if_conditions::BLOCKS_IN_IF_CONDITIONS,
        bool_assert_comparison::BOOL_ASSERT_COMPARISON,
        booleans::LOGIC_BUG,
        booleans::NONMINIMAL_BOOL,
        bytecount::NAIVE_BYTECOUNT,
        cargo_common_metadata::CARGO_COMMON_METADATA,
        case_sensitive_file_extension_comparisons::CASE_SENSITIVE_FILE_EXTENSION_COMPARISONS,
        casts::CAST_LOSSLESS,
        casts::CAST_POSSIBLE_TRUNCATION,
        casts::CAST_POSSIBLE_WRAP,
        casts::CAST_PRECISION_LOSS,
        casts::CAST_PTR_ALIGNMENT,
        casts::CAST_REF_TO_MUT,
        casts::CAST_SIGN_LOSS,
        casts::CHAR_LIT_AS_U8,
        casts::FN_TO_NUMERIC_CAST,
        casts::FN_TO_NUMERIC_CAST_WITH_TRUNCATION,
        casts::PTR_AS_PTR,
        casts::UNNECESSARY_CAST,
        checked_conversions::CHECKED_CONVERSIONS,
        cognitive_complexity::COGNITIVE_COMPLEXITY,
        collapsible_if::COLLAPSIBLE_ELSE_IF,
        collapsible_if::COLLAPSIBLE_IF,
        collapsible_match::COLLAPSIBLE_MATCH,
        comparison_chain::COMPARISON_CHAIN,
        copies::BRANCHES_SHARING_CODE,
        copies::IFS_SAME_COND,
        copies::IF_SAME_THEN_ELSE,
        copies::SAME_FUNCTIONS_IN_IF_CONDITION,
        copy_iterator::COPY_ITERATOR,
        create_dir::CREATE_DIR,
        dbg_macro::DBG_MACRO,
        default::DEFAULT_TRAIT_ACCESS,
        default::FIELD_REASSIGN_WITH_DEFAULT,
        default_numeric_fallback::DEFAULT_NUMERIC_FALLBACK,
        dereference::EXPLICIT_DEREF_METHODS,
        derivable_impls::DERIVABLE_IMPLS,
        derive::DERIVE_HASH_XOR_EQ,
        derive::DERIVE_ORD_XOR_PARTIAL_ORD,
        derive::EXPL_IMPL_CLONE_ON_COPY,
        derive::UNSAFE_DERIVE_DESERIALIZE,
        disallowed_method::DISALLOWED_METHOD,
        disallowed_script_idents::DISALLOWED_SCRIPT_IDENTS,
        disallowed_type::DISALLOWED_TYPE,
        doc::DOC_MARKDOWN,
        doc::MISSING_ERRORS_DOC,
        doc::MISSING_PANICS_DOC,
        doc::MISSING_SAFETY_DOC,
        doc::NEEDLESS_DOCTEST_MAIN,
        double_comparison::DOUBLE_COMPARISONS,
        double_parens::DOUBLE_PARENS,
        drop_forget_ref::DROP_COPY,
        drop_forget_ref::DROP_REF,
        drop_forget_ref::FORGET_COPY,
        drop_forget_ref::FORGET_REF,
        duration_subsec::DURATION_SUBSEC,
        else_if_without_else::ELSE_IF_WITHOUT_ELSE,
        empty_enum::EMPTY_ENUM,
        entry::MAP_ENTRY,
        enum_clike::ENUM_CLIKE_UNPORTABLE_VARIANT,
        enum_variants::ENUM_VARIANT_NAMES,
        enum_variants::MODULE_INCEPTION,
        enum_variants::MODULE_NAME_REPETITIONS,
        eq_op::EQ_OP,
        eq_op::OP_REF,
        erasing_op::ERASING_OP,
        escape::BOXED_LOCAL,
        eta_reduction::REDUNDANT_CLOSURE,
        eta_reduction::REDUNDANT_CLOSURE_FOR_METHOD_CALLS,
        eval_order_dependence::DIVERGING_SUB_EXPRESSION,
        eval_order_dependence::EVAL_ORDER_DEPENDENCE,
        excessive_bools::FN_PARAMS_EXCESSIVE_BOOLS,
        excessive_bools::STRUCT_EXCESSIVE_BOOLS,
        exhaustive_items::EXHAUSTIVE_ENUMS,
        exhaustive_items::EXHAUSTIVE_STRUCTS,
        exit::EXIT,
        explicit_write::EXPLICIT_WRITE,
        fallible_impl_from::FALLIBLE_IMPL_FROM,
        feature_name::NEGATIVE_FEATURE_NAMES,
        feature_name::REDUNDANT_FEATURE_NAMES,
        float_equality_without_abs::FLOAT_EQUALITY_WITHOUT_ABS,
        float_literal::EXCESSIVE_PRECISION,
        float_literal::LOSSY_FLOAT_LITERAL,
        floating_point_arithmetic::IMPRECISE_FLOPS,
        floating_point_arithmetic::SUBOPTIMAL_FLOPS,
        format::USELESS_FORMAT,
        formatting::POSSIBLE_MISSING_COMMA,
        formatting::SUSPICIOUS_ASSIGNMENT_FORMATTING,
        formatting::SUSPICIOUS_ELSE_FORMATTING,
        formatting::SUSPICIOUS_UNARY_OP_FORMATTING,
        from_over_into::FROM_OVER_INTO,
        from_str_radix_10::FROM_STR_RADIX_10,
        functions::DOUBLE_MUST_USE,
        functions::MUST_USE_CANDIDATE,
        functions::MUST_USE_UNIT,
        functions::NOT_UNSAFE_PTR_ARG_DEREF,
        functions::RESULT_UNIT_ERR,
        functions::TOO_MANY_ARGUMENTS,
        functions::TOO_MANY_LINES,
        future_not_send::FUTURE_NOT_SEND,
        get_last_with_len::GET_LAST_WITH_LEN,
        identity_op::IDENTITY_OP,
        if_let_mutex::IF_LET_MUTEX,
        if_let_some_result::IF_LET_SOME_RESULT,
        if_not_else::IF_NOT_ELSE,
        if_then_some_else_none::IF_THEN_SOME_ELSE_NONE,
        implicit_hasher::IMPLICIT_HASHER,
        implicit_return::IMPLICIT_RETURN,
        implicit_saturating_sub::IMPLICIT_SATURATING_SUB,
        inconsistent_struct_constructor::INCONSISTENT_STRUCT_CONSTRUCTOR,
        indexing_slicing::INDEXING_SLICING,
        indexing_slicing::OUT_OF_BOUNDS_INDEXING,
        infinite_iter::INFINITE_ITER,
        infinite_iter::MAYBE_INFINITE_ITER,
        inherent_impl::MULTIPLE_INHERENT_IMPL,
        inherent_to_string::INHERENT_TO_STRING,
        inherent_to_string::INHERENT_TO_STRING_SHADOW_DISPLAY,
        inline_fn_without_body::INLINE_FN_WITHOUT_BODY,
        int_plus_one::INT_PLUS_ONE,
        integer_division::INTEGER_DIVISION,
        invalid_upcast_comparisons::INVALID_UPCAST_COMPARISONS,
        items_after_statements::ITEMS_AFTER_STATEMENTS,
        large_const_arrays::LARGE_CONST_ARRAYS,
        large_enum_variant::LARGE_ENUM_VARIANT,
        large_stack_arrays::LARGE_STACK_ARRAYS,
        len_zero::COMPARISON_TO_EMPTY,
        len_zero::LEN_WITHOUT_IS_EMPTY,
        len_zero::LEN_ZERO,
        let_if_seq::USELESS_LET_IF_SEQ,
        let_underscore::LET_UNDERSCORE_DROP,
        let_underscore::LET_UNDERSCORE_LOCK,
        let_underscore::LET_UNDERSCORE_MUST_USE,
        lifetimes::EXTRA_UNUSED_LIFETIMES,
        lifetimes::NEEDLESS_LIFETIMES,
        literal_representation::DECIMAL_LITERAL_REPRESENTATION,
        literal_representation::INCONSISTENT_DIGIT_GROUPING,
        literal_representation::LARGE_DIGIT_GROUPS,
        literal_representation::MISTYPED_LITERAL_SUFFIXES,
        literal_representation::UNREADABLE_LITERAL,
        literal_representation::UNUSUAL_BYTE_GROUPINGS,
        loops::EMPTY_LOOP,
        loops::EXPLICIT_COUNTER_LOOP,
        loops::EXPLICIT_INTO_ITER_LOOP,
        loops::EXPLICIT_ITER_LOOP,
        loops::FOR_KV_MAP,
        loops::FOR_LOOPS_OVER_FALLIBLES,
        loops::ITER_NEXT_LOOP,
        loops::MANUAL_FLATTEN,
        loops::MANUAL_MEMCPY,
        loops::MUT_RANGE_BOUND,
        loops::NEEDLESS_COLLECT,
        loops::NEEDLESS_RANGE_LOOP,
        loops::NEVER_LOOP,
        loops::SAME_ITEM_PUSH,
        loops::SINGLE_ELEMENT_LOOP,
        loops::WHILE_IMMUTABLE_CONDITION,
        loops::WHILE_LET_LOOP,
        loops::WHILE_LET_ON_ITERATOR,
        macro_use::MACRO_USE_IMPORTS,
        main_recursion::MAIN_RECURSION,
        manual_async_fn::MANUAL_ASYNC_FN,
        manual_map::MANUAL_MAP,
        manual_non_exhaustive::MANUAL_NON_EXHAUSTIVE,
        manual_ok_or::MANUAL_OK_OR,
        manual_strip::MANUAL_STRIP,
        manual_unwrap_or::MANUAL_UNWRAP_OR,
        map_clone::MAP_CLONE,
        map_err_ignore::MAP_ERR_IGNORE,
        map_unit_fn::OPTION_MAP_UNIT_FN,
        map_unit_fn::RESULT_MAP_UNIT_FN,
        match_on_vec_items::MATCH_ON_VEC_ITEMS,
        matches::INFALLIBLE_DESTRUCTURING_MATCH,
        matches::MATCH_AS_REF,
        matches::MATCH_BOOL,
        matches::MATCH_LIKE_MATCHES_MACRO,
        matches::MATCH_OVERLAPPING_ARM,
        matches::MATCH_REF_PATS,
        matches::MATCH_SAME_ARMS,
        matches::MATCH_SINGLE_BINDING,
        matches::MATCH_WILDCARD_FOR_SINGLE_VARIANTS,
        matches::MATCH_WILD_ERR_ARM,
        matches::REDUNDANT_PATTERN_MATCHING,
        matches::REST_PAT_IN_FULLY_BOUND_STRUCTS,
        matches::SINGLE_MATCH,
        matches::SINGLE_MATCH_ELSE,
        matches::WILDCARD_ENUM_MATCH_ARM,
        matches::WILDCARD_IN_OR_PATTERNS,
        mem_discriminant::MEM_DISCRIMINANT_NON_ENUM,
        mem_forget::MEM_FORGET,
        mem_replace::MEM_REPLACE_OPTION_WITH_NONE,
        mem_replace::MEM_REPLACE_WITH_DEFAULT,
        mem_replace::MEM_REPLACE_WITH_UNINIT,
        methods::BIND_INSTEAD_OF_MAP,
        methods::BYTES_NTH,
        methods::CHARS_LAST_CMP,
        methods::CHARS_NEXT_CMP,
        methods::CLONED_INSTEAD_OF_COPIED,
        methods::CLONE_DOUBLE_REF,
        methods::CLONE_ON_COPY,
        methods::CLONE_ON_REF_PTR,
        methods::EXPECT_FUN_CALL,
        methods::EXPECT_USED,
        methods::EXTEND_WITH_DRAIN,
        methods::FILETYPE_IS_FILE,
        methods::FILTER_MAP_IDENTITY,
        methods::FILTER_MAP_NEXT,
        methods::FILTER_NEXT,
        methods::FLAT_MAP_IDENTITY,
        methods::FLAT_MAP_OPTION,
        methods::FROM_ITER_INSTEAD_OF_COLLECT,
        methods::GET_UNWRAP,
        methods::IMPLICIT_CLONE,
        methods::INEFFICIENT_TO_STRING,
        methods::INSPECT_FOR_EACH,
        methods::INTO_ITER_ON_REF,
        methods::ITERATOR_STEP_BY_ZERO,
        methods::ITER_CLONED_COLLECT,
        methods::ITER_COUNT,
        methods::ITER_NEXT_SLICE,
        methods::ITER_NTH,
        methods::ITER_NTH_ZERO,
        methods::ITER_SKIP_NEXT,
        methods::MANUAL_FILTER_MAP,
        methods::MANUAL_FIND_MAP,
        methods::MANUAL_SATURATING_ARITHMETIC,
        methods::MANUAL_SPLIT_ONCE,
        methods::MANUAL_STR_REPEAT,
        methods::MAP_COLLECT_RESULT_UNIT,
        methods::MAP_FLATTEN,
        methods::MAP_IDENTITY,
        methods::MAP_UNWRAP_OR,
        methods::NEW_RET_NO_SELF,
        methods::OK_EXPECT,
        methods::OPTION_AS_REF_DEREF,
        methods::OPTION_FILTER_MAP,
        methods::OPTION_MAP_OR_NONE,
        methods::OR_FUN_CALL,
        methods::RESULT_MAP_OR_INTO_OPTION,
        methods::SEARCH_IS_SOME,
        methods::SHOULD_IMPLEMENT_TRAIT,
        methods::SINGLE_CHAR_ADD_STR,
        methods::SINGLE_CHAR_PATTERN,
        methods::SKIP_WHILE_NEXT,
        methods::STRING_EXTEND_CHARS,
        methods::SUSPICIOUS_MAP,
        methods::SUSPICIOUS_SPLITN,
        methods::UNINIT_ASSUMED_INIT,
        methods::UNNECESSARY_FILTER_MAP,
        methods::UNNECESSARY_FOLD,
        methods::UNNECESSARY_LAZY_EVALUATIONS,
        methods::UNWRAP_OR_ELSE_DEFAULT,
        methods::UNWRAP_USED,
        methods::USELESS_ASREF,
        methods::WRONG_SELF_CONVENTION,
        methods::ZST_OFFSET,
        minmax::MIN_MAX,
        misc::CMP_NAN,
        misc::CMP_OWNED,
        misc::FLOAT_CMP,
        misc::FLOAT_CMP_CONST,
        misc::MODULO_ONE,
        misc::SHORT_CIRCUIT_STATEMENT,
        misc::TOPLEVEL_REF_ARG,
        misc::USED_UNDERSCORE_BINDING,
        misc::ZERO_PTR,
        misc_early::BUILTIN_TYPE_SHADOW,
        misc_early::DOUBLE_NEG,
        misc_early::DUPLICATE_UNDERSCORE_ARGUMENT,
        misc_early::MIXED_CASE_HEX_LITERALS,
        misc_early::REDUNDANT_PATTERN,
        misc_early::UNNEEDED_FIELD_PATTERN,
        misc_early::UNNEEDED_WILDCARD_PATTERN,
        misc_early::UNSEPARATED_LITERAL_SUFFIX,
        misc_early::ZERO_PREFIXED_LITERAL,
        missing_const_for_fn::MISSING_CONST_FOR_FN,
        missing_doc::MISSING_DOCS_IN_PRIVATE_ITEMS,
        missing_enforced_import_rename::MISSING_ENFORCED_IMPORT_RENAMES,
        missing_inline::MISSING_INLINE_IN_PUBLIC_ITEMS,
        module_style::MOD_MODULE_FILES,
        module_style::SELF_NAMED_MODULE_FILES,
        modulo_arithmetic::MODULO_ARITHMETIC,
        multiple_crate_versions::MULTIPLE_CRATE_VERSIONS,
        mut_key::MUTABLE_KEY_TYPE,
        mut_mut::MUT_MUT,
        mut_mutex_lock::MUT_MUTEX_LOCK,
        mut_reference::UNNECESSARY_MUT_PASSED,
        mutable_debug_assertion::DEBUG_ASSERT_WITH_MUT_CALL,
        mutex_atomic::MUTEX_ATOMIC,
        mutex_atomic::MUTEX_INTEGER,
        needless_arbitrary_self_type::NEEDLESS_ARBITRARY_SELF_TYPE,
        needless_bitwise_bool::NEEDLESS_BITWISE_BOOL,
        needless_bool::BOOL_COMPARISON,
        needless_bool::NEEDLESS_BOOL,
        needless_borrow::NEEDLESS_BORROW,
        needless_borrow::REF_BINDING_TO_REFERENCE,
        needless_borrowed_ref::NEEDLESS_BORROWED_REFERENCE,
        needless_continue::NEEDLESS_CONTINUE,
        needless_for_each::NEEDLESS_FOR_EACH,
        needless_option_as_deref::NEEDLESS_OPTION_AS_DEREF,
        needless_pass_by_value::NEEDLESS_PASS_BY_VALUE,
        needless_question_mark::NEEDLESS_QUESTION_MARK,
        needless_update::NEEDLESS_UPDATE,
        neg_cmp_op_on_partial_ord::NEG_CMP_OP_ON_PARTIAL_ORD,
        neg_multiply::NEG_MULTIPLY,
        new_without_default::NEW_WITHOUT_DEFAULT,
        no_effect::NO_EFFECT,
        no_effect::UNNECESSARY_OPERATION,
        non_copy_const::BORROW_INTERIOR_MUTABLE_CONST,
        non_copy_const::DECLARE_INTERIOR_MUTABLE_CONST,
        non_expressive_names::JUST_UNDERSCORES_AND_DIGITS,
        non_expressive_names::MANY_SINGLE_CHAR_NAMES,
        non_expressive_names::SIMILAR_NAMES,
        non_octal_unix_permissions::NON_OCTAL_UNIX_PERMISSIONS,
        nonstandard_macro_braces::NONSTANDARD_MACRO_BRACES,
        open_options::NONSENSICAL_OPEN_OPTIONS,
        option_env_unwrap::OPTION_ENV_UNWRAP,
        option_if_let_else::OPTION_IF_LET_ELSE,
        overflow_check_conditional::OVERFLOW_CHECK_CONDITIONAL,
        panic_in_result_fn::PANIC_IN_RESULT_FN,
        panic_unimplemented::PANIC,
        panic_unimplemented::TODO,
        panic_unimplemented::UNIMPLEMENTED,
        panic_unimplemented::UNREACHABLE,
        partialeq_ne_impl::PARTIALEQ_NE_IMPL,
        pass_by_ref_or_value::LARGE_TYPES_PASSED_BY_VALUE,
        pass_by_ref_or_value::TRIVIALLY_COPY_PASS_BY_REF,
        path_buf_push_overwrite::PATH_BUF_PUSH_OVERWRITE,
        pattern_type_mismatch::PATTERN_TYPE_MISMATCH,
        precedence::PRECEDENCE,
        ptr::CMP_NULL,
        ptr::INVALID_NULL_PTR_USAGE,
        ptr::MUT_FROM_REF,
        ptr::PTR_ARG,
        ptr_eq::PTR_EQ,
        ptr_offset_with_cast::PTR_OFFSET_WITH_CAST,
        question_mark::QUESTION_MARK,
        ranges::MANUAL_RANGE_CONTAINS,
        ranges::RANGE_MINUS_ONE,
        ranges::RANGE_PLUS_ONE,
        ranges::RANGE_ZIP_WITH_LEN,
        ranges::REVERSED_EMPTY_RANGES,
        redundant_clone::REDUNDANT_CLONE,
        redundant_closure_call::REDUNDANT_CLOSURE_CALL,
        redundant_else::REDUNDANT_ELSE,
        redundant_field_names::REDUNDANT_FIELD_NAMES,
        redundant_pub_crate::REDUNDANT_PUB_CRATE,
        redundant_slicing::REDUNDANT_SLICING,
        redundant_static_lifetimes::REDUNDANT_STATIC_LIFETIMES,
        ref_option_ref::REF_OPTION_REF,
        reference::DEREF_ADDROF,
        reference::REF_IN_DEREF,
        regex::INVALID_REGEX,
        regex::TRIVIAL_REGEX,
        repeat_once::REPEAT_ONCE,
        returns::LET_AND_RETURN,
        returns::NEEDLESS_RETURN,
        self_assignment::SELF_ASSIGNMENT,
        self_named_constructors::SELF_NAMED_CONSTRUCTORS,
        semicolon_if_nothing_returned::SEMICOLON_IF_NOTHING_RETURNED,
        serde_api::SERDE_API_MISUSE,
        shadow::SHADOW_REUSE,
        shadow::SHADOW_SAME,
        shadow::SHADOW_UNRELATED,
        single_component_path_imports::SINGLE_COMPONENT_PATH_IMPORTS,
        size_of_in_element_count::SIZE_OF_IN_ELEMENT_COUNT,
        slow_vector_initialization::SLOW_VECTOR_INITIALIZATION,
        stable_sort_primitive::STABLE_SORT_PRIMITIVE,
        strings::STRING_ADD,
        strings::STRING_ADD_ASSIGN,
        strings::STRING_FROM_UTF8_AS_BYTES,
        strings::STRING_LIT_AS_BYTES,
        strings::STRING_TO_STRING,
        strings::STR_TO_STRING,
        strlen_on_c_strings::STRLEN_ON_C_STRINGS,
        suspicious_operation_groupings::SUSPICIOUS_OPERATION_GROUPINGS,
        suspicious_trait_impl::SUSPICIOUS_ARITHMETIC_IMPL,
        suspicious_trait_impl::SUSPICIOUS_OP_ASSIGN_IMPL,
        swap::ALMOST_SWAPPED,
        swap::MANUAL_SWAP,
        tabs_in_doc_comments::TABS_IN_DOC_COMMENTS,
        temporary_assignment::TEMPORARY_ASSIGNMENT,
        to_digit_is_some::TO_DIGIT_IS_SOME,
        to_string_in_display::TO_STRING_IN_DISPLAY,
        trait_bounds::TRAIT_DUPLICATION_IN_BOUNDS,
        trait_bounds::TYPE_REPETITION_IN_BOUNDS,
        transmute::CROSSPOINTER_TRANSMUTE,
        transmute::TRANSMUTES_EXPRESSIBLE_AS_PTR_CASTS,
        transmute::TRANSMUTE_BYTES_TO_STR,
        transmute::TRANSMUTE_FLOAT_TO_INT,
        transmute::TRANSMUTE_INT_TO_BOOL,
        transmute::TRANSMUTE_INT_TO_CHAR,
        transmute::TRANSMUTE_INT_TO_FLOAT,
        transmute::TRANSMUTE_PTR_TO_PTR,
        transmute::TRANSMUTE_PTR_TO_REF,
        transmute::UNSOUND_COLLECTION_TRANSMUTE,
        transmute::USELESS_TRANSMUTE,
        transmute::WRONG_TRANSMUTE,
        transmuting_null::TRANSMUTING_NULL,
        try_err::TRY_ERR,
        types::BORROWED_BOX,
        types::BOX_VEC,
        types::LINKEDLIST,
        types::OPTION_OPTION,
        types::RC_BUFFER,
        types::RC_MUTEX,
        types::REDUNDANT_ALLOCATION,
        types::TYPE_COMPLEXITY,
        types::VEC_BOX,
        undropped_manually_drops::UNDROPPED_MANUALLY_DROPS,
        unicode::INVISIBLE_CHARACTERS,
        unicode::NON_ASCII_LITERAL,
        unicode::UNICODE_NOT_NFC,
        unit_return_expecting_ord::UNIT_RETURN_EXPECTING_ORD,
        unit_types::LET_UNIT_VALUE,
        unit_types::UNIT_ARG,
        unit_types::UNIT_CMP,
        unnamed_address::FN_ADDRESS_COMPARISONS,
        unnamed_address::VTABLE_ADDRESS_COMPARISONS,
        unnecessary_self_imports::UNNECESSARY_SELF_IMPORTS,
        unnecessary_sort_by::UNNECESSARY_SORT_BY,
        unnecessary_wraps::UNNECESSARY_WRAPS,
        unnested_or_patterns::UNNESTED_OR_PATTERNS,
        unsafe_removed_from_name::UNSAFE_REMOVED_FROM_NAME,
        unused_async::UNUSED_ASYNC,
        unused_io_amount::UNUSED_IO_AMOUNT,
        unused_self::UNUSED_SELF,
        unused_unit::UNUSED_UNIT,
        unwrap::PANICKING_UNWRAP,
        unwrap::UNNECESSARY_UNWRAP,
        unwrap_in_result::UNWRAP_IN_RESULT,
        upper_case_acronyms::UPPER_CASE_ACRONYMS,
        use_self::USE_SELF,
        useless_conversion::USELESS_CONVERSION,
        vec::USELESS_VEC,
        vec_init_then_push::VEC_INIT_THEN_PUSH,
        vec_resize_to_zero::VEC_RESIZE_TO_ZERO,
        verbose_file_reads::VERBOSE_FILE_READS,
        wildcard_dependencies::WILDCARD_DEPENDENCIES,
        wildcard_imports::ENUM_GLOB_USE,
        wildcard_imports::WILDCARD_IMPORTS,
        write::PRINTLN_EMPTY_STRING,
        write::PRINT_LITERAL,
        write::PRINT_STDERR,
        write::PRINT_STDOUT,
        write::PRINT_WITH_NEWLINE,
        write::USE_DEBUG,
        write::WRITELN_EMPTY_STRING,
        write::WRITE_LITERAL,
        write::WRITE_WITH_NEWLINE,
        zero_div_zero::ZERO_DIVIDED_BY_ZERO,
        zero_sized_map_values::ZERO_SIZED_MAP_VALUES,
    ]);
    // end register lints, do not remove this comment, it’s used in `update_lints`

    store.register_group(true, "clippy::restriction", Some("clippy_restriction"), vec![
        LintId::of(arithmetic::FLOAT_ARITHMETIC),
        LintId::of(arithmetic::INTEGER_ARITHMETIC),
        LintId::of(as_conversions::AS_CONVERSIONS),
        LintId::of(asm_syntax::INLINE_ASM_X86_ATT_SYNTAX),
        LintId::of(asm_syntax::INLINE_ASM_X86_INTEL_SYNTAX),
        LintId::of(create_dir::CREATE_DIR),
        LintId::of(dbg_macro::DBG_MACRO),
        LintId::of(default_numeric_fallback::DEFAULT_NUMERIC_FALLBACK),
        LintId::of(disallowed_script_idents::DISALLOWED_SCRIPT_IDENTS),
        LintId::of(else_if_without_else::ELSE_IF_WITHOUT_ELSE),
        LintId::of(exhaustive_items::EXHAUSTIVE_ENUMS),
        LintId::of(exhaustive_items::EXHAUSTIVE_STRUCTS),
        LintId::of(exit::EXIT),
        LintId::of(float_literal::LOSSY_FLOAT_LITERAL),
        LintId::of(if_then_some_else_none::IF_THEN_SOME_ELSE_NONE),
        LintId::of(implicit_return::IMPLICIT_RETURN),
        LintId::of(indexing_slicing::INDEXING_SLICING),
        LintId::of(inherent_impl::MULTIPLE_INHERENT_IMPL),
        LintId::of(integer_division::INTEGER_DIVISION),
        LintId::of(let_underscore::LET_UNDERSCORE_MUST_USE),
        LintId::of(literal_representation::DECIMAL_LITERAL_REPRESENTATION),
        LintId::of(map_err_ignore::MAP_ERR_IGNORE),
        LintId::of(matches::REST_PAT_IN_FULLY_BOUND_STRUCTS),
        LintId::of(matches::WILDCARD_ENUM_MATCH_ARM),
        LintId::of(mem_forget::MEM_FORGET),
        LintId::of(methods::CLONE_ON_REF_PTR),
        LintId::of(methods::EXPECT_USED),
        LintId::of(methods::FILETYPE_IS_FILE),
        LintId::of(methods::GET_UNWRAP),
        LintId::of(methods::UNWRAP_USED),
        LintId::of(misc::FLOAT_CMP_CONST),
        LintId::of(misc_early::UNNEEDED_FIELD_PATTERN),
        LintId::of(missing_doc::MISSING_DOCS_IN_PRIVATE_ITEMS),
        LintId::of(missing_enforced_import_rename::MISSING_ENFORCED_IMPORT_RENAMES),
        LintId::of(missing_inline::MISSING_INLINE_IN_PUBLIC_ITEMS),
        LintId::of(module_style::MOD_MODULE_FILES),
        LintId::of(module_style::SELF_NAMED_MODULE_FILES),
        LintId::of(modulo_arithmetic::MODULO_ARITHMETIC),
        LintId::of(panic_in_result_fn::PANIC_IN_RESULT_FN),
        LintId::of(panic_unimplemented::PANIC),
        LintId::of(panic_unimplemented::TODO),
        LintId::of(panic_unimplemented::UNIMPLEMENTED),
        LintId::of(panic_unimplemented::UNREACHABLE),
        LintId::of(pattern_type_mismatch::PATTERN_TYPE_MISMATCH),
        LintId::of(shadow::SHADOW_REUSE),
        LintId::of(shadow::SHADOW_SAME),
        LintId::of(strings::STRING_ADD),
        LintId::of(strings::STRING_TO_STRING),
        LintId::of(strings::STR_TO_STRING),
        LintId::of(types::RC_BUFFER),
        LintId::of(types::RC_MUTEX),
        LintId::of(unnecessary_self_imports::UNNECESSARY_SELF_IMPORTS),
        LintId::of(unwrap_in_result::UNWRAP_IN_RESULT),
        LintId::of(verbose_file_reads::VERBOSE_FILE_READS),
        LintId::of(write::PRINT_STDERR),
        LintId::of(write::PRINT_STDOUT),
        LintId::of(write::USE_DEBUG),
    ]);

    store.register_group(true, "clippy::pedantic", Some("clippy_pedantic"), vec![
        LintId::of(attrs::INLINE_ALWAYS),
        LintId::of(await_holding_invalid::AWAIT_HOLDING_LOCK),
        LintId::of(await_holding_invalid::AWAIT_HOLDING_REFCELL_REF),
        LintId::of(bit_mask::VERBOSE_BIT_MASK),
        LintId::of(bytecount::NAIVE_BYTECOUNT),
        LintId::of(case_sensitive_file_extension_comparisons::CASE_SENSITIVE_FILE_EXTENSION_COMPARISONS),
        LintId::of(casts::CAST_LOSSLESS),
        LintId::of(casts::CAST_POSSIBLE_TRUNCATION),
        LintId::of(casts::CAST_POSSIBLE_WRAP),
        LintId::of(casts::CAST_PRECISION_LOSS),
        LintId::of(casts::CAST_PTR_ALIGNMENT),
        LintId::of(casts::CAST_SIGN_LOSS),
        LintId::of(casts::PTR_AS_PTR),
        LintId::of(checked_conversions::CHECKED_CONVERSIONS),
        LintId::of(copies::SAME_FUNCTIONS_IN_IF_CONDITION),
        LintId::of(copy_iterator::COPY_ITERATOR),
        LintId::of(default::DEFAULT_TRAIT_ACCESS),
        LintId::of(dereference::EXPLICIT_DEREF_METHODS),
        LintId::of(derive::EXPL_IMPL_CLONE_ON_COPY),
        LintId::of(derive::UNSAFE_DERIVE_DESERIALIZE),
        LintId::of(doc::DOC_MARKDOWN),
        LintId::of(doc::MISSING_ERRORS_DOC),
        LintId::of(doc::MISSING_PANICS_DOC),
        LintId::of(empty_enum::EMPTY_ENUM),
        LintId::of(enum_variants::MODULE_NAME_REPETITIONS),
        LintId::of(eta_reduction::REDUNDANT_CLOSURE_FOR_METHOD_CALLS),
        LintId::of(excessive_bools::FN_PARAMS_EXCESSIVE_BOOLS),
        LintId::of(excessive_bools::STRUCT_EXCESSIVE_BOOLS),
        LintId::of(functions::MUST_USE_CANDIDATE),
        LintId::of(functions::TOO_MANY_LINES),
        LintId::of(if_not_else::IF_NOT_ELSE),
        LintId::of(implicit_hasher::IMPLICIT_HASHER),
        LintId::of(implicit_saturating_sub::IMPLICIT_SATURATING_SUB),
        LintId::of(inconsistent_struct_constructor::INCONSISTENT_STRUCT_CONSTRUCTOR),
        LintId::of(infinite_iter::MAYBE_INFINITE_ITER),
        LintId::of(invalid_upcast_comparisons::INVALID_UPCAST_COMPARISONS),
        LintId::of(items_after_statements::ITEMS_AFTER_STATEMENTS),
        LintId::of(large_stack_arrays::LARGE_STACK_ARRAYS),
        LintId::of(let_underscore::LET_UNDERSCORE_DROP),
        LintId::of(literal_representation::LARGE_DIGIT_GROUPS),
        LintId::of(literal_representation::UNREADABLE_LITERAL),
        LintId::of(loops::EXPLICIT_INTO_ITER_LOOP),
        LintId::of(loops::EXPLICIT_ITER_LOOP),
        LintId::of(macro_use::MACRO_USE_IMPORTS),
        LintId::of(manual_ok_or::MANUAL_OK_OR),
        LintId::of(match_on_vec_items::MATCH_ON_VEC_ITEMS),
        LintId::of(matches::MATCH_BOOL),
        LintId::of(matches::MATCH_SAME_ARMS),
        LintId::of(matches::MATCH_WILDCARD_FOR_SINGLE_VARIANTS),
        LintId::of(matches::MATCH_WILD_ERR_ARM),
        LintId::of(matches::SINGLE_MATCH_ELSE),
        LintId::of(methods::CLONED_INSTEAD_OF_COPIED),
        LintId::of(methods::FILTER_MAP_NEXT),
        LintId::of(methods::FLAT_MAP_OPTION),
        LintId::of(methods::FROM_ITER_INSTEAD_OF_COLLECT),
        LintId::of(methods::IMPLICIT_CLONE),
        LintId::of(methods::INEFFICIENT_TO_STRING),
        LintId::of(methods::MAP_FLATTEN),
        LintId::of(methods::MAP_UNWRAP_OR),
        LintId::of(misc::USED_UNDERSCORE_BINDING),
        LintId::of(misc_early::UNSEPARATED_LITERAL_SUFFIX),
        LintId::of(mut_mut::MUT_MUT),
        LintId::of(needless_bitwise_bool::NEEDLESS_BITWISE_BOOL),
        LintId::of(needless_borrow::REF_BINDING_TO_REFERENCE),
        LintId::of(needless_continue::NEEDLESS_CONTINUE),
        LintId::of(needless_for_each::NEEDLESS_FOR_EACH),
        LintId::of(needless_pass_by_value::NEEDLESS_PASS_BY_VALUE),
        LintId::of(non_expressive_names::SIMILAR_NAMES),
        LintId::of(pass_by_ref_or_value::LARGE_TYPES_PASSED_BY_VALUE),
        LintId::of(pass_by_ref_or_value::TRIVIALLY_COPY_PASS_BY_REF),
        LintId::of(ranges::RANGE_MINUS_ONE),
        LintId::of(ranges::RANGE_PLUS_ONE),
        LintId::of(redundant_else::REDUNDANT_ELSE),
        LintId::of(ref_option_ref::REF_OPTION_REF),
        LintId::of(semicolon_if_nothing_returned::SEMICOLON_IF_NOTHING_RETURNED),
        LintId::of(shadow::SHADOW_UNRELATED),
        LintId::of(strings::STRING_ADD_ASSIGN),
        LintId::of(trait_bounds::TRAIT_DUPLICATION_IN_BOUNDS),
        LintId::of(trait_bounds::TYPE_REPETITION_IN_BOUNDS),
        LintId::of(transmute::TRANSMUTE_PTR_TO_PTR),
        LintId::of(types::LINKEDLIST),
        LintId::of(types::OPTION_OPTION),
        LintId::of(unicode::NON_ASCII_LITERAL),
        LintId::of(unicode::UNICODE_NOT_NFC),
        LintId::of(unit_types::LET_UNIT_VALUE),
        LintId::of(unnecessary_wraps::UNNECESSARY_WRAPS),
        LintId::of(unnested_or_patterns::UNNESTED_OR_PATTERNS),
        LintId::of(unused_async::UNUSED_ASYNC),
        LintId::of(unused_self::UNUSED_SELF),
        LintId::of(wildcard_imports::ENUM_GLOB_USE),
        LintId::of(wildcard_imports::WILDCARD_IMPORTS),
        LintId::of(zero_sized_map_values::ZERO_SIZED_MAP_VALUES),
    ]);

    #[cfg(feature = "internal-lints")]
    store.register_group(true, "clippy::internal", Some("clippy_internal"), vec![
        LintId::of(utils::internal_lints::CLIPPY_LINTS_INTERNAL),
        LintId::of(utils::internal_lints::COLLAPSIBLE_SPAN_LINT_CALLS),
        LintId::of(utils::internal_lints::COMPILER_LINT_FUNCTIONS),
        LintId::of(utils::internal_lints::DEFAULT_LINT),
        LintId::of(utils::internal_lints::IF_CHAIN_STYLE),
        LintId::of(utils::internal_lints::INTERNING_DEFINED_SYMBOL),
        LintId::of(utils::internal_lints::INVALID_PATHS),
        LintId::of(utils::internal_lints::LINT_WITHOUT_LINT_PASS),
        LintId::of(utils::internal_lints::MATCH_TYPE_ON_DIAGNOSTIC_ITEM),
        LintId::of(utils::internal_lints::OUTER_EXPN_EXPN_DATA),
        LintId::of(utils::internal_lints::PRODUCE_ICE),
        LintId::of(utils::internal_lints::UNNECESSARY_SYMBOL_STR),
    ]);

    store.register_group(true, "clippy::all", Some("clippy"), vec![
        LintId::of(absurd_extreme_comparisons::ABSURD_EXTREME_COMPARISONS),
        LintId::of(approx_const::APPROX_CONSTANT),
        LintId::of(assertions_on_constants::ASSERTIONS_ON_CONSTANTS),
        LintId::of(assign_ops::ASSIGN_OP_PATTERN),
        LintId::of(assign_ops::MISREFACTORED_ASSIGN_OP),
        LintId::of(async_yields_async::ASYNC_YIELDS_ASYNC),
        LintId::of(attrs::BLANKET_CLIPPY_RESTRICTION_LINTS),
        LintId::of(attrs::DEPRECATED_CFG_ATTR),
        LintId::of(attrs::DEPRECATED_SEMVER),
        LintId::of(attrs::MISMATCHED_TARGET_OS),
        LintId::of(attrs::USELESS_ATTRIBUTE),
        LintId::of(bit_mask::BAD_BIT_MASK),
        LintId::of(bit_mask::INEFFECTIVE_BIT_MASK),
        LintId::of(blacklisted_name::BLACKLISTED_NAME),
        LintId::of(blocks_in_if_conditions::BLOCKS_IN_IF_CONDITIONS),
        LintId::of(bool_assert_comparison::BOOL_ASSERT_COMPARISON),
        LintId::of(booleans::LOGIC_BUG),
        LintId::of(booleans::NONMINIMAL_BOOL),
        LintId::of(casts::CAST_REF_TO_MUT),
        LintId::of(casts::CHAR_LIT_AS_U8),
        LintId::of(casts::FN_TO_NUMERIC_CAST),
        LintId::of(casts::FN_TO_NUMERIC_CAST_WITH_TRUNCATION),
        LintId::of(casts::UNNECESSARY_CAST),
        LintId::of(collapsible_if::COLLAPSIBLE_ELSE_IF),
        LintId::of(collapsible_if::COLLAPSIBLE_IF),
        LintId::of(collapsible_match::COLLAPSIBLE_MATCH),
        LintId::of(comparison_chain::COMPARISON_CHAIN),
        LintId::of(copies::IFS_SAME_COND),
        LintId::of(copies::IF_SAME_THEN_ELSE),
        LintId::of(default::FIELD_REASSIGN_WITH_DEFAULT),
        LintId::of(derivable_impls::DERIVABLE_IMPLS),
        LintId::of(derive::DERIVE_HASH_XOR_EQ),
        LintId::of(derive::DERIVE_ORD_XOR_PARTIAL_ORD),
        LintId::of(doc::MISSING_SAFETY_DOC),
        LintId::of(doc::NEEDLESS_DOCTEST_MAIN),
        LintId::of(double_comparison::DOUBLE_COMPARISONS),
        LintId::of(double_parens::DOUBLE_PARENS),
        LintId::of(drop_forget_ref::DROP_COPY),
        LintId::of(drop_forget_ref::DROP_REF),
        LintId::of(drop_forget_ref::FORGET_COPY),
        LintId::of(drop_forget_ref::FORGET_REF),
        LintId::of(duration_subsec::DURATION_SUBSEC),
        LintId::of(entry::MAP_ENTRY),
        LintId::of(enum_clike::ENUM_CLIKE_UNPORTABLE_VARIANT),
        LintId::of(enum_variants::ENUM_VARIANT_NAMES),
        LintId::of(enum_variants::MODULE_INCEPTION),
        LintId::of(eq_op::EQ_OP),
        LintId::of(eq_op::OP_REF),
        LintId::of(erasing_op::ERASING_OP),
        LintId::of(escape::BOXED_LOCAL),
        LintId::of(eta_reduction::REDUNDANT_CLOSURE),
        LintId::of(eval_order_dependence::DIVERGING_SUB_EXPRESSION),
        LintId::of(eval_order_dependence::EVAL_ORDER_DEPENDENCE),
        LintId::of(explicit_write::EXPLICIT_WRITE),
        LintId::of(float_equality_without_abs::FLOAT_EQUALITY_WITHOUT_ABS),
        LintId::of(float_literal::EXCESSIVE_PRECISION),
        LintId::of(format::USELESS_FORMAT),
        LintId::of(formatting::POSSIBLE_MISSING_COMMA),
        LintId::of(formatting::SUSPICIOUS_ASSIGNMENT_FORMATTING),
        LintId::of(formatting::SUSPICIOUS_ELSE_FORMATTING),
        LintId::of(formatting::SUSPICIOUS_UNARY_OP_FORMATTING),
        LintId::of(from_over_into::FROM_OVER_INTO),
        LintId::of(from_str_radix_10::FROM_STR_RADIX_10),
        LintId::of(functions::DOUBLE_MUST_USE),
        LintId::of(functions::MUST_USE_UNIT),
        LintId::of(functions::NOT_UNSAFE_PTR_ARG_DEREF),
        LintId::of(functions::RESULT_UNIT_ERR),
        LintId::of(functions::TOO_MANY_ARGUMENTS),
        LintId::of(get_last_with_len::GET_LAST_WITH_LEN),
        LintId::of(identity_op::IDENTITY_OP),
        LintId::of(if_let_mutex::IF_LET_MUTEX),
        LintId::of(if_let_some_result::IF_LET_SOME_RESULT),
        LintId::of(indexing_slicing::OUT_OF_BOUNDS_INDEXING),
        LintId::of(infinite_iter::INFINITE_ITER),
        LintId::of(inherent_to_string::INHERENT_TO_STRING),
        LintId::of(inherent_to_string::INHERENT_TO_STRING_SHADOW_DISPLAY),
        LintId::of(inline_fn_without_body::INLINE_FN_WITHOUT_BODY),
        LintId::of(int_plus_one::INT_PLUS_ONE),
        LintId::of(large_const_arrays::LARGE_CONST_ARRAYS),
        LintId::of(large_enum_variant::LARGE_ENUM_VARIANT),
        LintId::of(len_zero::COMPARISON_TO_EMPTY),
        LintId::of(len_zero::LEN_WITHOUT_IS_EMPTY),
        LintId::of(len_zero::LEN_ZERO),
        LintId::of(let_underscore::LET_UNDERSCORE_LOCK),
        LintId::of(lifetimes::EXTRA_UNUSED_LIFETIMES),
        LintId::of(lifetimes::NEEDLESS_LIFETIMES),
        LintId::of(literal_representation::INCONSISTENT_DIGIT_GROUPING),
        LintId::of(literal_representation::MISTYPED_LITERAL_SUFFIXES),
        LintId::of(literal_representation::UNUSUAL_BYTE_GROUPINGS),
        LintId::of(loops::EMPTY_LOOP),
        LintId::of(loops::EXPLICIT_COUNTER_LOOP),
        LintId::of(loops::FOR_KV_MAP),
        LintId::of(loops::FOR_LOOPS_OVER_FALLIBLES),
        LintId::of(loops::ITER_NEXT_LOOP),
        LintId::of(loops::MANUAL_FLATTEN),
        LintId::of(loops::MANUAL_MEMCPY),
        LintId::of(loops::MUT_RANGE_BOUND),
        LintId::of(loops::NEEDLESS_COLLECT),
        LintId::of(loops::NEEDLESS_RANGE_LOOP),
        LintId::of(loops::NEVER_LOOP),
        LintId::of(loops::SAME_ITEM_PUSH),
        LintId::of(loops::SINGLE_ELEMENT_LOOP),
        LintId::of(loops::WHILE_IMMUTABLE_CONDITION),
        LintId::of(loops::WHILE_LET_LOOP),
        LintId::of(loops::WHILE_LET_ON_ITERATOR),
        LintId::of(main_recursion::MAIN_RECURSION),
        LintId::of(manual_async_fn::MANUAL_ASYNC_FN),
        LintId::of(manual_map::MANUAL_MAP),
        LintId::of(manual_non_exhaustive::MANUAL_NON_EXHAUSTIVE),
        LintId::of(manual_strip::MANUAL_STRIP),
        LintId::of(manual_unwrap_or::MANUAL_UNWRAP_OR),
        LintId::of(map_clone::MAP_CLONE),
        LintId::of(map_unit_fn::OPTION_MAP_UNIT_FN),
        LintId::of(map_unit_fn::RESULT_MAP_UNIT_FN),
        LintId::of(matches::INFALLIBLE_DESTRUCTURING_MATCH),
        LintId::of(matches::MATCH_AS_REF),
        LintId::of(matches::MATCH_LIKE_MATCHES_MACRO),
        LintId::of(matches::MATCH_OVERLAPPING_ARM),
        LintId::of(matches::MATCH_REF_PATS),
        LintId::of(matches::MATCH_SINGLE_BINDING),
        LintId::of(matches::REDUNDANT_PATTERN_MATCHING),
        LintId::of(matches::SINGLE_MATCH),
        LintId::of(matches::WILDCARD_IN_OR_PATTERNS),
        LintId::of(mem_discriminant::MEM_DISCRIMINANT_NON_ENUM),
        LintId::of(mem_replace::MEM_REPLACE_OPTION_WITH_NONE),
        LintId::of(mem_replace::MEM_REPLACE_WITH_DEFAULT),
        LintId::of(mem_replace::MEM_REPLACE_WITH_UNINIT),
        LintId::of(methods::BIND_INSTEAD_OF_MAP),
        LintId::of(methods::BYTES_NTH),
        LintId::of(methods::CHARS_LAST_CMP),
        LintId::of(methods::CHARS_NEXT_CMP),
        LintId::of(methods::CLONE_DOUBLE_REF),
        LintId::of(methods::CLONE_ON_COPY),
        LintId::of(methods::EXPECT_FUN_CALL),
        LintId::of(methods::EXTEND_WITH_DRAIN),
        LintId::of(methods::FILTER_MAP_IDENTITY),
        LintId::of(methods::FILTER_NEXT),
        LintId::of(methods::FLAT_MAP_IDENTITY),
        LintId::of(methods::INSPECT_FOR_EACH),
        LintId::of(methods::INTO_ITER_ON_REF),
        LintId::of(methods::ITERATOR_STEP_BY_ZERO),
        LintId::of(methods::ITER_CLONED_COLLECT),
        LintId::of(methods::ITER_COUNT),
        LintId::of(methods::ITER_NEXT_SLICE),
        LintId::of(methods::ITER_NTH),
        LintId::of(methods::ITER_NTH_ZERO),
        LintId::of(methods::ITER_SKIP_NEXT),
        LintId::of(methods::MANUAL_FILTER_MAP),
        LintId::of(methods::MANUAL_FIND_MAP),
        LintId::of(methods::MANUAL_SATURATING_ARITHMETIC),
        LintId::of(methods::MANUAL_SPLIT_ONCE),
        LintId::of(methods::MANUAL_STR_REPEAT),
        LintId::of(methods::MAP_COLLECT_RESULT_UNIT),
        LintId::of(methods::MAP_IDENTITY),
        LintId::of(methods::NEW_RET_NO_SELF),
        LintId::of(methods::OK_EXPECT),
        LintId::of(methods::OPTION_AS_REF_DEREF),
        LintId::of(methods::OPTION_FILTER_MAP),
        LintId::of(methods::OPTION_MAP_OR_NONE),
        LintId::of(methods::OR_FUN_CALL),
        LintId::of(methods::RESULT_MAP_OR_INTO_OPTION),
        LintId::of(methods::SEARCH_IS_SOME),
        LintId::of(methods::SHOULD_IMPLEMENT_TRAIT),
        LintId::of(methods::SINGLE_CHAR_ADD_STR),
        LintId::of(methods::SINGLE_CHAR_PATTERN),
        LintId::of(methods::SKIP_WHILE_NEXT),
        LintId::of(methods::STRING_EXTEND_CHARS),
        LintId::of(methods::SUSPICIOUS_MAP),
        LintId::of(methods::SUSPICIOUS_SPLITN),
        LintId::of(methods::UNINIT_ASSUMED_INIT),
        LintId::of(methods::UNNECESSARY_FILTER_MAP),
        LintId::of(methods::UNNECESSARY_FOLD),
        LintId::of(methods::UNNECESSARY_LAZY_EVALUATIONS),
        LintId::of(methods::UNWRAP_OR_ELSE_DEFAULT),
        LintId::of(methods::USELESS_ASREF),
        LintId::of(methods::WRONG_SELF_CONVENTION),
        LintId::of(methods::ZST_OFFSET),
        LintId::of(minmax::MIN_MAX),
        LintId::of(misc::CMP_NAN),
        LintId::of(misc::CMP_OWNED),
        LintId::of(misc::FLOAT_CMP),
        LintId::of(misc::MODULO_ONE),
        LintId::of(misc::SHORT_CIRCUIT_STATEMENT),
        LintId::of(misc::TOPLEVEL_REF_ARG),
        LintId::of(misc::ZERO_PTR),
        LintId::of(misc_early::BUILTIN_TYPE_SHADOW),
        LintId::of(misc_early::DOUBLE_NEG),
        LintId::of(misc_early::DUPLICATE_UNDERSCORE_ARGUMENT),
        LintId::of(misc_early::MIXED_CASE_HEX_LITERALS),
        LintId::of(misc_early::REDUNDANT_PATTERN),
        LintId::of(misc_early::UNNEEDED_WILDCARD_PATTERN),
        LintId::of(misc_early::ZERO_PREFIXED_LITERAL),
        LintId::of(mut_key::MUTABLE_KEY_TYPE),
        LintId::of(mut_mutex_lock::MUT_MUTEX_LOCK),
        LintId::of(mut_reference::UNNECESSARY_MUT_PASSED),
        LintId::of(mutex_atomic::MUTEX_ATOMIC),
        LintId::of(needless_arbitrary_self_type::NEEDLESS_ARBITRARY_SELF_TYPE),
        LintId::of(needless_bool::BOOL_COMPARISON),
        LintId::of(needless_bool::NEEDLESS_BOOL),
        LintId::of(needless_borrow::NEEDLESS_BORROW),
        LintId::of(needless_borrowed_ref::NEEDLESS_BORROWED_REFERENCE),
        LintId::of(needless_option_as_deref::NEEDLESS_OPTION_AS_DEREF),
        LintId::of(needless_question_mark::NEEDLESS_QUESTION_MARK),
        LintId::of(needless_update::NEEDLESS_UPDATE),
        LintId::of(neg_cmp_op_on_partial_ord::NEG_CMP_OP_ON_PARTIAL_ORD),
        LintId::of(neg_multiply::NEG_MULTIPLY),
        LintId::of(new_without_default::NEW_WITHOUT_DEFAULT),
        LintId::of(no_effect::NO_EFFECT),
        LintId::of(no_effect::UNNECESSARY_OPERATION),
        LintId::of(non_copy_const::BORROW_INTERIOR_MUTABLE_CONST),
        LintId::of(non_copy_const::DECLARE_INTERIOR_MUTABLE_CONST),
        LintId::of(non_expressive_names::JUST_UNDERSCORES_AND_DIGITS),
        LintId::of(non_expressive_names::MANY_SINGLE_CHAR_NAMES),
        LintId::of(non_octal_unix_permissions::NON_OCTAL_UNIX_PERMISSIONS),
        LintId::of(open_options::NONSENSICAL_OPEN_OPTIONS),
        LintId::of(option_env_unwrap::OPTION_ENV_UNWRAP),
        LintId::of(overflow_check_conditional::OVERFLOW_CHECK_CONDITIONAL),
        LintId::of(partialeq_ne_impl::PARTIALEQ_NE_IMPL),
        LintId::of(precedence::PRECEDENCE),
        LintId::of(ptr::CMP_NULL),
        LintId::of(ptr::INVALID_NULL_PTR_USAGE),
        LintId::of(ptr::MUT_FROM_REF),
        LintId::of(ptr::PTR_ARG),
        LintId::of(ptr_eq::PTR_EQ),
        LintId::of(ptr_offset_with_cast::PTR_OFFSET_WITH_CAST),
        LintId::of(question_mark::QUESTION_MARK),
        LintId::of(ranges::MANUAL_RANGE_CONTAINS),
        LintId::of(ranges::RANGE_ZIP_WITH_LEN),
        LintId::of(ranges::REVERSED_EMPTY_RANGES),
        LintId::of(redundant_clone::REDUNDANT_CLONE),
        LintId::of(redundant_closure_call::REDUNDANT_CLOSURE_CALL),
        LintId::of(redundant_field_names::REDUNDANT_FIELD_NAMES),
        LintId::of(redundant_slicing::REDUNDANT_SLICING),
        LintId::of(redundant_static_lifetimes::REDUNDANT_STATIC_LIFETIMES),
        LintId::of(reference::DEREF_ADDROF),
        LintId::of(reference::REF_IN_DEREF),
        LintId::of(regex::INVALID_REGEX),
        LintId::of(repeat_once::REPEAT_ONCE),
        LintId::of(returns::LET_AND_RETURN),
        LintId::of(returns::NEEDLESS_RETURN),
        LintId::of(self_assignment::SELF_ASSIGNMENT),
        LintId::of(self_named_constructors::SELF_NAMED_CONSTRUCTORS),
        LintId::of(serde_api::SERDE_API_MISUSE),
        LintId::of(single_component_path_imports::SINGLE_COMPONENT_PATH_IMPORTS),
        LintId::of(size_of_in_element_count::SIZE_OF_IN_ELEMENT_COUNT),
        LintId::of(slow_vector_initialization::SLOW_VECTOR_INITIALIZATION),
        LintId::of(stable_sort_primitive::STABLE_SORT_PRIMITIVE),
        LintId::of(strings::STRING_FROM_UTF8_AS_BYTES),
        LintId::of(strlen_on_c_strings::STRLEN_ON_C_STRINGS),
        LintId::of(suspicious_trait_impl::SUSPICIOUS_ARITHMETIC_IMPL),
        LintId::of(suspicious_trait_impl::SUSPICIOUS_OP_ASSIGN_IMPL),
        LintId::of(swap::ALMOST_SWAPPED),
        LintId::of(swap::MANUAL_SWAP),
        LintId::of(tabs_in_doc_comments::TABS_IN_DOC_COMMENTS),
        LintId::of(temporary_assignment::TEMPORARY_ASSIGNMENT),
        LintId::of(to_digit_is_some::TO_DIGIT_IS_SOME),
        LintId::of(to_string_in_display::TO_STRING_IN_DISPLAY),
        LintId::of(transmute::CROSSPOINTER_TRANSMUTE),
        LintId::of(transmute::TRANSMUTES_EXPRESSIBLE_AS_PTR_CASTS),
        LintId::of(transmute::TRANSMUTE_BYTES_TO_STR),
        LintId::of(transmute::TRANSMUTE_FLOAT_TO_INT),
        LintId::of(transmute::TRANSMUTE_INT_TO_BOOL),
        LintId::of(transmute::TRANSMUTE_INT_TO_CHAR),
        LintId::of(transmute::TRANSMUTE_INT_TO_FLOAT),
        LintId::of(transmute::TRANSMUTE_PTR_TO_REF),
        LintId::of(transmute::UNSOUND_COLLECTION_TRANSMUTE),
        LintId::of(transmute::WRONG_TRANSMUTE),
        LintId::of(transmuting_null::TRANSMUTING_NULL),
        LintId::of(try_err::TRY_ERR),
        LintId::of(types::BORROWED_BOX),
        LintId::of(types::BOX_VEC),
        LintId::of(types::REDUNDANT_ALLOCATION),
        LintId::of(types::TYPE_COMPLEXITY),
        LintId::of(types::VEC_BOX),
        LintId::of(undropped_manually_drops::UNDROPPED_MANUALLY_DROPS),
        LintId::of(unicode::INVISIBLE_CHARACTERS),
        LintId::of(unit_return_expecting_ord::UNIT_RETURN_EXPECTING_ORD),
        LintId::of(unit_types::UNIT_ARG),
        LintId::of(unit_types::UNIT_CMP),
        LintId::of(unnamed_address::FN_ADDRESS_COMPARISONS),
        LintId::of(unnamed_address::VTABLE_ADDRESS_COMPARISONS),
        LintId::of(unnecessary_sort_by::UNNECESSARY_SORT_BY),
        LintId::of(unsafe_removed_from_name::UNSAFE_REMOVED_FROM_NAME),
        LintId::of(unused_io_amount::UNUSED_IO_AMOUNT),
        LintId::of(unused_unit::UNUSED_UNIT),
        LintId::of(unwrap::PANICKING_UNWRAP),
        LintId::of(unwrap::UNNECESSARY_UNWRAP),
        LintId::of(upper_case_acronyms::UPPER_CASE_ACRONYMS),
        LintId::of(useless_conversion::USELESS_CONVERSION),
        LintId::of(vec::USELESS_VEC),
        LintId::of(vec_init_then_push::VEC_INIT_THEN_PUSH),
        LintId::of(vec_resize_to_zero::VEC_RESIZE_TO_ZERO),
        LintId::of(write::PRINTLN_EMPTY_STRING),
        LintId::of(write::PRINT_LITERAL),
        LintId::of(write::PRINT_WITH_NEWLINE),
        LintId::of(write::WRITELN_EMPTY_STRING),
        LintId::of(write::WRITE_LITERAL),
        LintId::of(write::WRITE_WITH_NEWLINE),
        LintId::of(zero_div_zero::ZERO_DIVIDED_BY_ZERO),
    ]);

    store.register_group(true, "clippy::style", Some("clippy_style"), vec![
        LintId::of(assertions_on_constants::ASSERTIONS_ON_CONSTANTS),
        LintId::of(assign_ops::ASSIGN_OP_PATTERN),
        LintId::of(blacklisted_name::BLACKLISTED_NAME),
        LintId::of(blocks_in_if_conditions::BLOCKS_IN_IF_CONDITIONS),
        LintId::of(bool_assert_comparison::BOOL_ASSERT_COMPARISON),
        LintId::of(casts::FN_TO_NUMERIC_CAST),
        LintId::of(casts::FN_TO_NUMERIC_CAST_WITH_TRUNCATION),
        LintId::of(collapsible_if::COLLAPSIBLE_ELSE_IF),
        LintId::of(collapsible_if::COLLAPSIBLE_IF),
        LintId::of(collapsible_match::COLLAPSIBLE_MATCH),
        LintId::of(comparison_chain::COMPARISON_CHAIN),
        LintId::of(default::FIELD_REASSIGN_WITH_DEFAULT),
        LintId::of(doc::MISSING_SAFETY_DOC),
        LintId::of(doc::NEEDLESS_DOCTEST_MAIN),
        LintId::of(enum_variants::ENUM_VARIANT_NAMES),
        LintId::of(enum_variants::MODULE_INCEPTION),
        LintId::of(eq_op::OP_REF),
        LintId::of(eta_reduction::REDUNDANT_CLOSURE),
        LintId::of(float_literal::EXCESSIVE_PRECISION),
        LintId::of(from_over_into::FROM_OVER_INTO),
        LintId::of(from_str_radix_10::FROM_STR_RADIX_10),
        LintId::of(functions::DOUBLE_MUST_USE),
        LintId::of(functions::MUST_USE_UNIT),
        LintId::of(functions::RESULT_UNIT_ERR),
        LintId::of(if_let_some_result::IF_LET_SOME_RESULT),
        LintId::of(inherent_to_string::INHERENT_TO_STRING),
        LintId::of(len_zero::COMPARISON_TO_EMPTY),
        LintId::of(len_zero::LEN_WITHOUT_IS_EMPTY),
        LintId::of(len_zero::LEN_ZERO),
        LintId::of(literal_representation::INCONSISTENT_DIGIT_GROUPING),
        LintId::of(literal_representation::UNUSUAL_BYTE_GROUPINGS),
        LintId::of(loops::FOR_KV_MAP),
        LintId::of(loops::NEEDLESS_RANGE_LOOP),
        LintId::of(loops::SAME_ITEM_PUSH),
        LintId::of(loops::WHILE_LET_ON_ITERATOR),
        LintId::of(main_recursion::MAIN_RECURSION),
        LintId::of(manual_async_fn::MANUAL_ASYNC_FN),
        LintId::of(manual_map::MANUAL_MAP),
        LintId::of(manual_non_exhaustive::MANUAL_NON_EXHAUSTIVE),
        LintId::of(map_clone::MAP_CLONE),
        LintId::of(matches::INFALLIBLE_DESTRUCTURING_MATCH),
        LintId::of(matches::MATCH_LIKE_MATCHES_MACRO),
        LintId::of(matches::MATCH_OVERLAPPING_ARM),
        LintId::of(matches::MATCH_REF_PATS),
        LintId::of(matches::REDUNDANT_PATTERN_MATCHING),
        LintId::of(matches::SINGLE_MATCH),
        LintId::of(mem_replace::MEM_REPLACE_OPTION_WITH_NONE),
        LintId::of(mem_replace::MEM_REPLACE_WITH_DEFAULT),
        LintId::of(methods::BYTES_NTH),
        LintId::of(methods::CHARS_LAST_CMP),
        LintId::of(methods::CHARS_NEXT_CMP),
        LintId::of(methods::INTO_ITER_ON_REF),
        LintId::of(methods::ITER_CLONED_COLLECT),
        LintId::of(methods::ITER_NEXT_SLICE),
        LintId::of(methods::ITER_NTH_ZERO),
        LintId::of(methods::ITER_SKIP_NEXT),
        LintId::of(methods::MANUAL_SATURATING_ARITHMETIC),
        LintId::of(methods::MAP_COLLECT_RESULT_UNIT),
        LintId::of(methods::NEW_RET_NO_SELF),
        LintId::of(methods::OK_EXPECT),
        LintId::of(methods::OPTION_MAP_OR_NONE),
        LintId::of(methods::RESULT_MAP_OR_INTO_OPTION),
        LintId::of(methods::SHOULD_IMPLEMENT_TRAIT),
        LintId::of(methods::SINGLE_CHAR_ADD_STR),
        LintId::of(methods::STRING_EXTEND_CHARS),
        LintId::of(methods::UNNECESSARY_FOLD),
        LintId::of(methods::UNNECESSARY_LAZY_EVALUATIONS),
        LintId::of(methods::UNWRAP_OR_ELSE_DEFAULT),
        LintId::of(methods::WRONG_SELF_CONVENTION),
        LintId::of(misc::TOPLEVEL_REF_ARG),
        LintId::of(misc::ZERO_PTR),
        LintId::of(misc_early::BUILTIN_TYPE_SHADOW),
        LintId::of(misc_early::DOUBLE_NEG),
        LintId::of(misc_early::DUPLICATE_UNDERSCORE_ARGUMENT),
        LintId::of(misc_early::MIXED_CASE_HEX_LITERALS),
        LintId::of(misc_early::REDUNDANT_PATTERN),
        LintId::of(mut_mutex_lock::MUT_MUTEX_LOCK),
        LintId::of(mut_reference::UNNECESSARY_MUT_PASSED),
        LintId::of(needless_borrow::NEEDLESS_BORROW),
        LintId::of(neg_multiply::NEG_MULTIPLY),
        LintId::of(new_without_default::NEW_WITHOUT_DEFAULT),
        LintId::of(non_copy_const::BORROW_INTERIOR_MUTABLE_CONST),
        LintId::of(non_copy_const::DECLARE_INTERIOR_MUTABLE_CONST),
        LintId::of(non_expressive_names::JUST_UNDERSCORES_AND_DIGITS),
        LintId::of(non_expressive_names::MANY_SINGLE_CHAR_NAMES),
        LintId::of(ptr::CMP_NULL),
        LintId::of(ptr::PTR_ARG),
        LintId::of(ptr_eq::PTR_EQ),
        LintId::of(question_mark::QUESTION_MARK),
        LintId::of(ranges::MANUAL_RANGE_CONTAINS),
        LintId::of(redundant_field_names::REDUNDANT_FIELD_NAMES),
        LintId::of(redundant_static_lifetimes::REDUNDANT_STATIC_LIFETIMES),
        LintId::of(returns::LET_AND_RETURN),
        LintId::of(returns::NEEDLESS_RETURN),
        LintId::of(self_named_constructors::SELF_NAMED_CONSTRUCTORS),
        LintId::of(single_component_path_imports::SINGLE_COMPONENT_PATH_IMPORTS),
        LintId::of(tabs_in_doc_comments::TABS_IN_DOC_COMMENTS),
        LintId::of(to_digit_is_some::TO_DIGIT_IS_SOME),
        LintId::of(try_err::TRY_ERR),
        LintId::of(unsafe_removed_from_name::UNSAFE_REMOVED_FROM_NAME),
        LintId::of(unused_unit::UNUSED_UNIT),
        LintId::of(upper_case_acronyms::UPPER_CASE_ACRONYMS),
        LintId::of(write::PRINTLN_EMPTY_STRING),
        LintId::of(write::PRINT_LITERAL),
        LintId::of(write::PRINT_WITH_NEWLINE),
        LintId::of(write::WRITELN_EMPTY_STRING),
        LintId::of(write::WRITE_LITERAL),
        LintId::of(write::WRITE_WITH_NEWLINE),
    ]);

    store.register_group(true, "clippy::complexity", Some("clippy_complexity"), vec![
        LintId::of(attrs::DEPRECATED_CFG_ATTR),
        LintId::of(booleans::NONMINIMAL_BOOL),
        LintId::of(casts::CHAR_LIT_AS_U8),
        LintId::of(casts::UNNECESSARY_CAST),
        LintId::of(derivable_impls::DERIVABLE_IMPLS),
        LintId::of(double_comparison::DOUBLE_COMPARISONS),
        LintId::of(double_parens::DOUBLE_PARENS),
        LintId::of(duration_subsec::DURATION_SUBSEC),
        LintId::of(eval_order_dependence::DIVERGING_SUB_EXPRESSION),
        LintId::of(explicit_write::EXPLICIT_WRITE),
        LintId::of(format::USELESS_FORMAT),
        LintId::of(functions::TOO_MANY_ARGUMENTS),
        LintId::of(get_last_with_len::GET_LAST_WITH_LEN),
        LintId::of(identity_op::IDENTITY_OP),
        LintId::of(int_plus_one::INT_PLUS_ONE),
        LintId::of(lifetimes::EXTRA_UNUSED_LIFETIMES),
        LintId::of(lifetimes::NEEDLESS_LIFETIMES),
        LintId::of(loops::EXPLICIT_COUNTER_LOOP),
        LintId::of(loops::MANUAL_FLATTEN),
        LintId::of(loops::SINGLE_ELEMENT_LOOP),
        LintId::of(loops::WHILE_LET_LOOP),
        LintId::of(manual_strip::MANUAL_STRIP),
        LintId::of(manual_unwrap_or::MANUAL_UNWRAP_OR),
        LintId::of(map_unit_fn::OPTION_MAP_UNIT_FN),
        LintId::of(map_unit_fn::RESULT_MAP_UNIT_FN),
        LintId::of(matches::MATCH_AS_REF),
        LintId::of(matches::MATCH_SINGLE_BINDING),
        LintId::of(matches::WILDCARD_IN_OR_PATTERNS),
        LintId::of(methods::BIND_INSTEAD_OF_MAP),
        LintId::of(methods::CLONE_ON_COPY),
        LintId::of(methods::FILTER_MAP_IDENTITY),
        LintId::of(methods::FILTER_NEXT),
        LintId::of(methods::FLAT_MAP_IDENTITY),
        LintId::of(methods::INSPECT_FOR_EACH),
        LintId::of(methods::ITER_COUNT),
        LintId::of(methods::MANUAL_FILTER_MAP),
        LintId::of(methods::MANUAL_FIND_MAP),
        LintId::of(methods::MANUAL_SPLIT_ONCE),
        LintId::of(methods::MAP_IDENTITY),
        LintId::of(methods::OPTION_AS_REF_DEREF),
        LintId::of(methods::OPTION_FILTER_MAP),
        LintId::of(methods::SEARCH_IS_SOME),
        LintId::of(methods::SKIP_WHILE_NEXT),
        LintId::of(methods::UNNECESSARY_FILTER_MAP),
        LintId::of(methods::USELESS_ASREF),
        LintId::of(misc::SHORT_CIRCUIT_STATEMENT),
        LintId::of(misc_early::UNNEEDED_WILDCARD_PATTERN),
        LintId::of(misc_early::ZERO_PREFIXED_LITERAL),
        LintId::of(needless_arbitrary_self_type::NEEDLESS_ARBITRARY_SELF_TYPE),
        LintId::of(needless_bool::BOOL_COMPARISON),
        LintId::of(needless_bool::NEEDLESS_BOOL),
        LintId::of(needless_borrowed_ref::NEEDLESS_BORROWED_REFERENCE),
        LintId::of(needless_option_as_deref::NEEDLESS_OPTION_AS_DEREF),
        LintId::of(needless_question_mark::NEEDLESS_QUESTION_MARK),
        LintId::of(needless_update::NEEDLESS_UPDATE),
        LintId::of(neg_cmp_op_on_partial_ord::NEG_CMP_OP_ON_PARTIAL_ORD),
        LintId::of(no_effect::NO_EFFECT),
        LintId::of(no_effect::UNNECESSARY_OPERATION),
        LintId::of(overflow_check_conditional::OVERFLOW_CHECK_CONDITIONAL),
        LintId::of(partialeq_ne_impl::PARTIALEQ_NE_IMPL),
        LintId::of(precedence::PRECEDENCE),
        LintId::of(ptr_offset_with_cast::PTR_OFFSET_WITH_CAST),
        LintId::of(ranges::RANGE_ZIP_WITH_LEN),
        LintId::of(redundant_closure_call::REDUNDANT_CLOSURE_CALL),
        LintId::of(redundant_slicing::REDUNDANT_SLICING),
        LintId::of(reference::DEREF_ADDROF),
        LintId::of(reference::REF_IN_DEREF),
        LintId::of(repeat_once::REPEAT_ONCE),
        LintId::of(strings::STRING_FROM_UTF8_AS_BYTES),
        LintId::of(strlen_on_c_strings::STRLEN_ON_C_STRINGS),
        LintId::of(swap::MANUAL_SWAP),
        LintId::of(temporary_assignment::TEMPORARY_ASSIGNMENT),
        LintId::of(transmute::CROSSPOINTER_TRANSMUTE),
        LintId::of(transmute::TRANSMUTES_EXPRESSIBLE_AS_PTR_CASTS),
        LintId::of(transmute::TRANSMUTE_BYTES_TO_STR),
        LintId::of(transmute::TRANSMUTE_FLOAT_TO_INT),
        LintId::of(transmute::TRANSMUTE_INT_TO_BOOL),
        LintId::of(transmute::TRANSMUTE_INT_TO_CHAR),
        LintId::of(transmute::TRANSMUTE_INT_TO_FLOAT),
        LintId::of(transmute::TRANSMUTE_PTR_TO_REF),
        LintId::of(types::BORROWED_BOX),
        LintId::of(types::TYPE_COMPLEXITY),
        LintId::of(types::VEC_BOX),
        LintId::of(unit_types::UNIT_ARG),
        LintId::of(unnecessary_sort_by::UNNECESSARY_SORT_BY),
        LintId::of(unwrap::UNNECESSARY_UNWRAP),
        LintId::of(useless_conversion::USELESS_CONVERSION),
        LintId::of(zero_div_zero::ZERO_DIVIDED_BY_ZERO),
    ]);

    store.register_group(true, "clippy::correctness", Some("clippy_correctness"), vec![
        LintId::of(absurd_extreme_comparisons::ABSURD_EXTREME_COMPARISONS),
        LintId::of(approx_const::APPROX_CONSTANT),
        LintId::of(async_yields_async::ASYNC_YIELDS_ASYNC),
        LintId::of(attrs::DEPRECATED_SEMVER),
        LintId::of(attrs::MISMATCHED_TARGET_OS),
        LintId::of(attrs::USELESS_ATTRIBUTE),
        LintId::of(bit_mask::BAD_BIT_MASK),
        LintId::of(bit_mask::INEFFECTIVE_BIT_MASK),
        LintId::of(booleans::LOGIC_BUG),
        LintId::of(casts::CAST_REF_TO_MUT),
        LintId::of(copies::IFS_SAME_COND),
        LintId::of(copies::IF_SAME_THEN_ELSE),
        LintId::of(derive::DERIVE_HASH_XOR_EQ),
        LintId::of(derive::DERIVE_ORD_XOR_PARTIAL_ORD),
        LintId::of(drop_forget_ref::DROP_COPY),
        LintId::of(drop_forget_ref::DROP_REF),
        LintId::of(drop_forget_ref::FORGET_COPY),
        LintId::of(drop_forget_ref::FORGET_REF),
        LintId::of(enum_clike::ENUM_CLIKE_UNPORTABLE_VARIANT),
        LintId::of(eq_op::EQ_OP),
        LintId::of(erasing_op::ERASING_OP),
        LintId::of(formatting::POSSIBLE_MISSING_COMMA),
        LintId::of(functions::NOT_UNSAFE_PTR_ARG_DEREF),
        LintId::of(if_let_mutex::IF_LET_MUTEX),
        LintId::of(indexing_slicing::OUT_OF_BOUNDS_INDEXING),
        LintId::of(infinite_iter::INFINITE_ITER),
        LintId::of(inherent_to_string::INHERENT_TO_STRING_SHADOW_DISPLAY),
        LintId::of(inline_fn_without_body::INLINE_FN_WITHOUT_BODY),
        LintId::of(let_underscore::LET_UNDERSCORE_LOCK),
        LintId::of(literal_representation::MISTYPED_LITERAL_SUFFIXES),
        LintId::of(loops::ITER_NEXT_LOOP),
        LintId::of(loops::NEVER_LOOP),
        LintId::of(loops::WHILE_IMMUTABLE_CONDITION),
        LintId::of(mem_discriminant::MEM_DISCRIMINANT_NON_ENUM),
        LintId::of(mem_replace::MEM_REPLACE_WITH_UNINIT),
        LintId::of(methods::CLONE_DOUBLE_REF),
        LintId::of(methods::ITERATOR_STEP_BY_ZERO),
        LintId::of(methods::SUSPICIOUS_SPLITN),
        LintId::of(methods::UNINIT_ASSUMED_INIT),
        LintId::of(methods::ZST_OFFSET),
        LintId::of(minmax::MIN_MAX),
        LintId::of(misc::CMP_NAN),
        LintId::of(misc::FLOAT_CMP),
        LintId::of(misc::MODULO_ONE),
        LintId::of(non_octal_unix_permissions::NON_OCTAL_UNIX_PERMISSIONS),
        LintId::of(open_options::NONSENSICAL_OPEN_OPTIONS),
        LintId::of(option_env_unwrap::OPTION_ENV_UNWRAP),
        LintId::of(ptr::INVALID_NULL_PTR_USAGE),
        LintId::of(ptr::MUT_FROM_REF),
        LintId::of(ranges::REVERSED_EMPTY_RANGES),
        LintId::of(regex::INVALID_REGEX),
        LintId::of(self_assignment::SELF_ASSIGNMENT),
        LintId::of(serde_api::SERDE_API_MISUSE),
        LintId::of(size_of_in_element_count::SIZE_OF_IN_ELEMENT_COUNT),
        LintId::of(swap::ALMOST_SWAPPED),
        LintId::of(to_string_in_display::TO_STRING_IN_DISPLAY),
        LintId::of(transmute::UNSOUND_COLLECTION_TRANSMUTE),
        LintId::of(transmute::WRONG_TRANSMUTE),
        LintId::of(transmuting_null::TRANSMUTING_NULL),
        LintId::of(undropped_manually_drops::UNDROPPED_MANUALLY_DROPS),
        LintId::of(unicode::INVISIBLE_CHARACTERS),
        LintId::of(unit_return_expecting_ord::UNIT_RETURN_EXPECTING_ORD),
        LintId::of(unit_types::UNIT_CMP),
        LintId::of(unnamed_address::FN_ADDRESS_COMPARISONS),
        LintId::of(unnamed_address::VTABLE_ADDRESS_COMPARISONS),
        LintId::of(unused_io_amount::UNUSED_IO_AMOUNT),
        LintId::of(unwrap::PANICKING_UNWRAP),
        LintId::of(vec_resize_to_zero::VEC_RESIZE_TO_ZERO),
    ]);

    store.register_group(true, "clippy::suspicious", None, vec![
        LintId::of(assign_ops::MISREFACTORED_ASSIGN_OP),
        LintId::of(attrs::BLANKET_CLIPPY_RESTRICTION_LINTS),
        LintId::of(eval_order_dependence::EVAL_ORDER_DEPENDENCE),
        LintId::of(float_equality_without_abs::FLOAT_EQUALITY_WITHOUT_ABS),
        LintId::of(formatting::SUSPICIOUS_ASSIGNMENT_FORMATTING),
        LintId::of(formatting::SUSPICIOUS_ELSE_FORMATTING),
        LintId::of(formatting::SUSPICIOUS_UNARY_OP_FORMATTING),
        LintId::of(loops::EMPTY_LOOP),
        LintId::of(loops::FOR_LOOPS_OVER_FALLIBLES),
        LintId::of(loops::MUT_RANGE_BOUND),
        LintId::of(methods::SUSPICIOUS_MAP),
        LintId::of(mut_key::MUTABLE_KEY_TYPE),
        LintId::of(suspicious_trait_impl::SUSPICIOUS_ARITHMETIC_IMPL),
        LintId::of(suspicious_trait_impl::SUSPICIOUS_OP_ASSIGN_IMPL),
    ]);

    store.register_group(true, "clippy::perf", Some("clippy_perf"), vec![
        LintId::of(entry::MAP_ENTRY),
        LintId::of(escape::BOXED_LOCAL),
        LintId::of(large_const_arrays::LARGE_CONST_ARRAYS),
        LintId::of(large_enum_variant::LARGE_ENUM_VARIANT),
        LintId::of(loops::MANUAL_MEMCPY),
        LintId::of(loops::NEEDLESS_COLLECT),
        LintId::of(methods::EXPECT_FUN_CALL),
        LintId::of(methods::EXTEND_WITH_DRAIN),
        LintId::of(methods::ITER_NTH),
        LintId::of(methods::MANUAL_STR_REPEAT),
        LintId::of(methods::OR_FUN_CALL),
        LintId::of(methods::SINGLE_CHAR_PATTERN),
        LintId::of(misc::CMP_OWNED),
        LintId::of(mutex_atomic::MUTEX_ATOMIC),
        LintId::of(redundant_clone::REDUNDANT_CLONE),
        LintId::of(slow_vector_initialization::SLOW_VECTOR_INITIALIZATION),
        LintId::of(stable_sort_primitive::STABLE_SORT_PRIMITIVE),
        LintId::of(types::BOX_VEC),
        LintId::of(types::REDUNDANT_ALLOCATION),
        LintId::of(vec::USELESS_VEC),
        LintId::of(vec_init_then_push::VEC_INIT_THEN_PUSH),
    ]);

    store.register_group(true, "clippy::cargo", Some("clippy_cargo"), vec![
        LintId::of(cargo_common_metadata::CARGO_COMMON_METADATA),
        LintId::of(feature_name::NEGATIVE_FEATURE_NAMES),
        LintId::of(feature_name::REDUNDANT_FEATURE_NAMES),
        LintId::of(multiple_crate_versions::MULTIPLE_CRATE_VERSIONS),
        LintId::of(wildcard_dependencies::WILDCARD_DEPENDENCIES),
    ]);

    store.register_group(true, "clippy::nursery", Some("clippy_nursery"), vec![
        LintId::of(attrs::EMPTY_LINE_AFTER_OUTER_ATTR),
        LintId::of(cognitive_complexity::COGNITIVE_COMPLEXITY),
        LintId::of(copies::BRANCHES_SHARING_CODE),
        LintId::of(disallowed_method::DISALLOWED_METHOD),
        LintId::of(disallowed_type::DISALLOWED_TYPE),
        LintId::of(fallible_impl_from::FALLIBLE_IMPL_FROM),
        LintId::of(floating_point_arithmetic::IMPRECISE_FLOPS),
        LintId::of(floating_point_arithmetic::SUBOPTIMAL_FLOPS),
        LintId::of(future_not_send::FUTURE_NOT_SEND),
        LintId::of(let_if_seq::USELESS_LET_IF_SEQ),
        LintId::of(missing_const_for_fn::MISSING_CONST_FOR_FN),
        LintId::of(mutable_debug_assertion::DEBUG_ASSERT_WITH_MUT_CALL),
        LintId::of(mutex_atomic::MUTEX_INTEGER),
        LintId::of(nonstandard_macro_braces::NONSTANDARD_MACRO_BRACES),
        LintId::of(option_if_let_else::OPTION_IF_LET_ELSE),
        LintId::of(path_buf_push_overwrite::PATH_BUF_PUSH_OVERWRITE),
        LintId::of(redundant_pub_crate::REDUNDANT_PUB_CRATE),
        LintId::of(regex::TRIVIAL_REGEX),
        LintId::of(strings::STRING_LIT_AS_BYTES),
        LintId::of(suspicious_operation_groupings::SUSPICIOUS_OPERATION_GROUPINGS),
        LintId::of(transmute::USELESS_TRANSMUTE),
        LintId::of(use_self::USE_SELF),
    ]);

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
    store.register_late_pass(|| Box::new(map_clone::MapClone));
    store.register_late_pass(|| Box::new(map_err_ignore::MapErrIgnore));
    store.register_late_pass(|| Box::new(shadow::Shadow));
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
    store.register_late_pass(|| Box::new(regex::Regex::default()));
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
    store.register_late_pass(|| Box::new(mem_discriminant::MemDiscriminant));
    store.register_late_pass(|| Box::new(mem_forget::MemForget));
    store.register_late_pass(|| Box::new(arithmetic::Arithmetic::default()));
    store.register_late_pass(|| Box::new(assign_ops::AssignOps));
    store.register_late_pass(|| Box::new(let_if_seq::LetIfSeq));
    store.register_late_pass(|| Box::new(eval_order_dependence::EvalOrderDependence));
    store.register_late_pass(|| Box::new(missing_doc::MissingDoc::new()));
    store.register_late_pass(|| Box::new(missing_inline::MissingInline));
    store.register_late_pass(move || Box::new(exhaustive_items::ExhaustiveItems));
    store.register_late_pass(|| Box::new(if_let_some_result::OkIfLet));
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
    let disallowed_methods = conf.disallowed_methods.iter().cloned().collect::<FxHashSet<_>>();
    store.register_late_pass(move || Box::new(disallowed_method::DisallowedMethod::new(&disallowed_methods)));
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

    // uplifted lints
    ls.register_renamed("clippy::invalid_ref", "invalid_value");
    ls.register_renamed("clippy::into_iter_on_array", "array_into_iter");
    ls.register_renamed("clippy::unused_label", "unused_labels");
    ls.register_renamed("clippy::drop_bounds", "drop_bounds");
    ls.register_renamed("clippy::temporary_cstring_as_ptr", "temporary_cstring_as_ptr");
    ls.register_renamed("clippy::panic_params", "non_fmt_panics");
    ls.register_renamed("clippy::unknown_clippy_lints", "unknown_lints");
    ls.register_renamed("clippy::invalid_atomic_ordering", "invalid_atomic_ordering");
}

// only exists to let the dogfood integration test works.
// Don't run clippy as an executable directly
#[allow(dead_code)]
fn main() {
    panic!("Please use the cargo-clippy executable");
}
