// error-pattern:cargo-clippy

#![feature(box_syntax)]
#![feature(box_patterns)]
#![feature(never_type)]
#![feature(rustc_private)]
#![feature(slice_patterns)]
#![feature(stmt_expr_attributes)]
#![allow(clippy::missing_docs_in_private_items, clippy::must_use_candidate)]
#![recursion_limit = "512"]
#![warn(rust_2018_idioms, trivial_casts, trivial_numeric_casts)]
#![deny(rustc::internal)]
#![cfg_attr(feature = "deny-warnings", deny(warnings))]
#![feature(crate_visibility_modifier)]
#![feature(concat_idents)]

// FIXME: switch to something more ergonomic here, once available.
// (Currently there is no way to opt into sysroot crates without `extern crate`.)
#[allow(unused_extern_crates)]
extern crate fmt_macros;
#[allow(unused_extern_crates)]
extern crate rustc;
#[allow(unused_extern_crates)]
extern crate rustc_data_structures;
#[allow(unused_extern_crates)]
extern crate rustc_driver;
#[allow(unused_extern_crates)]
extern crate rustc_errors;
#[allow(unused_extern_crates)]
extern crate rustc_index;
#[allow(unused_extern_crates)]
extern crate rustc_mir;
#[allow(unused_extern_crates)]
extern crate rustc_target;
#[allow(unused_extern_crates)]
extern crate rustc_typeck;
#[allow(unused_extern_crates)]
extern crate syntax;
#[allow(unused_extern_crates)]
extern crate syntax_pos;

use rustc::lint::{self, LintId};
use rustc::session::Session;
use rustc_data_structures::fx::FxHashSet;
use toml;

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
/// Currently the categories `style`, `correctness`, `complexity` and `perf` are enabled by default.
/// As said in the README.md of this repository, if the lint level mapping changes, please update
/// README.md.
///
/// # Example
///
/// ```
/// # #![feature(rustc_private)]
/// # #[allow(unused_extern_crates)]
/// # extern crate rustc;
/// # #[macro_use]
/// # use clippy_lints::declare_clippy_lint;
/// use rustc::declare_tool_lint;
///
/// declare_clippy_lint! {
///     /// **What it does:** Checks for ... (describe what the lint matches).
///     ///
///     /// **Why is this bad?** Supply the reason for linting the code.
///     ///
///     /// **Known problems:** None. (Or describe where it could go wrong.)
///     ///
///     /// **Example:**
///     ///
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

mod consts;
#[macro_use]
mod utils;

// begin lints modules, do not remove this comment, it’s used in `update_lints`
pub mod approx_const;
pub mod arithmetic;
pub mod assertions_on_constants;
pub mod assign_ops;
pub mod attrs;
pub mod bit_mask;
pub mod blacklisted_name;
pub mod block_in_if_condition;
pub mod booleans;
pub mod bytecount;
pub mod cargo_common_metadata;
pub mod checked_conversions;
pub mod cognitive_complexity;
pub mod collapsible_if;
pub mod comparison_chain;
pub mod copies;
pub mod copy_iterator;
pub mod dbg_macro;
pub mod default_trait_access;
pub mod derive;
pub mod doc;
pub mod double_comparison;
pub mod double_parens;
pub mod drop_bounds;
pub mod drop_forget_ref;
pub mod duration_subsec;
pub mod else_if_without_else;
pub mod empty_enum;
pub mod entry;
pub mod enum_clike;
pub mod enum_glob_use;
pub mod enum_variants;
pub mod eq_op;
pub mod erasing_op;
pub mod escape;
pub mod eta_reduction;
pub mod eval_order_dependence;
pub mod excessive_precision;
pub mod explicit_write;
pub mod fallible_impl_from;
pub mod format;
pub mod formatting;
pub mod functions;
pub mod get_last_with_len;
pub mod identity_conversion;
pub mod identity_op;
pub mod if_not_else;
pub mod implicit_return;
pub mod indexing_slicing;
pub mod infallible_destructuring_match;
pub mod infinite_iter;
pub mod inherent_impl;
pub mod inherent_to_string;
pub mod inline_fn_without_body;
pub mod int_plus_one;
pub mod integer_division;
pub mod items_after_statements;
pub mod large_enum_variant;
pub mod len_zero;
pub mod let_if_seq;
pub mod lifetimes;
pub mod literal_representation;
pub mod loops;
pub mod main_recursion;
pub mod map_clone;
pub mod map_unit_fn;
pub mod matches;
pub mod mem_discriminant;
pub mod mem_forget;
pub mod mem_replace;
pub mod methods;
pub mod minmax;
pub mod misc;
pub mod misc_early;
pub mod missing_const_for_fn;
pub mod missing_doc;
pub mod missing_inline;
pub mod mul_add;
pub mod multiple_crate_versions;
pub mod mut_mut;
pub mod mut_reference;
pub mod mutable_debug_assertion;
pub mod mutex_atomic;
pub mod needless_bool;
pub mod needless_borrow;
pub mod needless_borrowed_ref;
pub mod needless_continue;
pub mod needless_pass_by_value;
pub mod needless_update;
pub mod neg_cmp_op_on_partial_ord;
pub mod neg_multiply;
pub mod new_without_default;
pub mod no_effect;
pub mod non_copy_const;
pub mod non_expressive_names;
pub mod ok_if_let;
pub mod open_options;
pub mod overflow_check_conditional;
pub mod panic_unimplemented;
pub mod partialeq_ne_impl;
pub mod path_buf_push_overwrite;
pub mod precedence;
pub mod ptr;
pub mod ptr_offset_with_cast;
pub mod question_mark;
pub mod ranges;
pub mod redundant_clone;
pub mod redundant_field_names;
pub mod redundant_pattern_matching;
pub mod redundant_static_lifetimes;
pub mod reference;
pub mod regex;
pub mod replace_consts;
pub mod returns;
pub mod serde_api;
pub mod shadow;
pub mod slow_vector_initialization;
pub mod strings;
pub mod suspicious_trait_impl;
pub mod swap;
pub mod temporary_assignment;
pub mod trait_bounds;
pub mod transmute;
pub mod transmuting_null;
pub mod trivially_copy_pass_by_ref;
pub mod try_err;
pub mod types;
pub mod unicode;
pub mod unsafe_removed_from_name;
pub mod unused_io_amount;
pub mod unused_label;
pub mod unused_self;
pub mod unwrap;
pub mod use_self;
pub mod vec;
pub mod wildcard_dependencies;
pub mod write;
pub mod zero_div_zero;
// end lints modules, do not remove this comment, it’s used in `update_lints`

pub use crate::utils::conf::Conf;

mod reexport {
    crate use syntax::ast::Name;
}

/// Register all pre expansion lints
///
/// Pre-expansion lints run before any macro expansion has happened.
///
/// Note that due to the architecture of the compiler, currently `cfg_attr` attributes on crate
/// level (i.e `#![cfg_attr(...)]`) will still be expanded even when using a pre-expansion pass.
///
/// Used in `./src/driver.rs`.
pub fn register_pre_expansion_lints(store: &mut rustc::lint::LintStore, conf: &Conf) {
    store.register_pre_expansion_pass(|| box write::Write);
    store.register_pre_expansion_pass(|| box redundant_field_names::RedundantFieldNames);
    let single_char_binding_names_threshold = conf.single_char_binding_names_threshold;
    store.register_pre_expansion_pass(move || box non_expressive_names::NonExpressiveNames {
        single_char_binding_names_threshold,
    });
    store.register_pre_expansion_pass(|| box attrs::DeprecatedCfgAttribute);
    store.register_pre_expansion_pass(|| box dbg_macro::DbgMacro);
}

#[doc(hidden)]
pub fn read_conf(args: &[syntax::ast::NestedMetaItem], sess: &Session) -> Conf {
    match utils::conf::file_from_args(args) {
        Ok(file_name) => {
            // if the user specified a file, it must exist, otherwise default to `clippy.toml` but
            // do not require the file to exist
            let file_name = if let Some(file_name) = file_name {
                Some(file_name)
            } else {
                match utils::conf::lookup_conf_file() {
                    Ok(path) => path,
                    Err(error) => {
                        sess.struct_err(&format!("error finding Clippy's configuration file: {}", error))
                            .emit();
                        None
                    },
                }
            };

            let file_name = file_name.map(|file_name| {
                if file_name.is_relative() {
                    sess.local_crate_source_file
                        .as_ref()
                        .and_then(|file| std::path::Path::new(&file).parent().map(std::path::Path::to_path_buf))
                        .unwrap_or_default()
                        .join(file_name)
                } else {
                    file_name
                }
            });

            let (conf, errors) = utils::conf::read(file_name.as_ref().map(std::convert::AsRef::as_ref));

            // all conf errors are non-fatal, we just use the default conf in case of error
            for error in errors {
                sess.struct_err(&format!(
                    "error reading Clippy's configuration file `{}`: {}",
                    file_name.as_ref().and_then(|p| p.to_str()).unwrap_or(""),
                    error
                ))
                .emit();
            }

            conf
        },
        Err((err, span)) => {
            sess.struct_span_err(span, err)
                .span_note(span, "Clippy will use default configuration")
                .emit();
            toml::from_str("").expect("we never error on empty config files")
        },
    }
}

/// Register all lints and lint groups with the rustc plugin registry
///
/// Used in `./src/driver.rs`.
#[allow(clippy::too_many_lines)]
#[rustfmt::skip]
pub fn register_plugins(store: &mut lint::LintStore, sess: &Session, conf: &Conf) {
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
        "clippy::str_to_string",
        "using `str::to_string` is common even today and specialization will likely happen soon",
    );
    store.register_removed(
        "clippy::string_to_string",
        "using `string::to_string` is common even today and specialization will likely happen soon",
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
        "clippy::invalid_ref",
        "superseded by rustc lint `invalid_value`",
    );
    store.register_removed(
        "clippy::unused_collect",
        "`collect` has been marked as #[must_use] in rustc and that covers all cases of this lint",
    );
    // end deprecated lints, do not remove this comment, it’s used in `update_lints`

    // begin register lints, do not remove this comment, it’s used in `update_lints`
    store.register_lints(&[
        &approx_const::APPROX_CONSTANT,
        &arithmetic::FLOAT_ARITHMETIC,
        &arithmetic::INTEGER_ARITHMETIC,
        &assertions_on_constants::ASSERTIONS_ON_CONSTANTS,
        &assign_ops::ASSIGN_OP_PATTERN,
        &assign_ops::MISREFACTORED_ASSIGN_OP,
        &attrs::DEPRECATED_CFG_ATTR,
        &attrs::DEPRECATED_SEMVER,
        &attrs::EMPTY_LINE_AFTER_OUTER_ATTR,
        &attrs::INLINE_ALWAYS,
        &attrs::UNKNOWN_CLIPPY_LINTS,
        &attrs::USELESS_ATTRIBUTE,
        &bit_mask::BAD_BIT_MASK,
        &bit_mask::INEFFECTIVE_BIT_MASK,
        &bit_mask::VERBOSE_BIT_MASK,
        &blacklisted_name::BLACKLISTED_NAME,
        &block_in_if_condition::BLOCK_IN_IF_CONDITION_EXPR,
        &block_in_if_condition::BLOCK_IN_IF_CONDITION_STMT,
        &booleans::LOGIC_BUG,
        &booleans::NONMINIMAL_BOOL,
        &bytecount::NAIVE_BYTECOUNT,
        &cargo_common_metadata::CARGO_COMMON_METADATA,
        &checked_conversions::CHECKED_CONVERSIONS,
        &cognitive_complexity::COGNITIVE_COMPLEXITY,
        &collapsible_if::COLLAPSIBLE_IF,
        &comparison_chain::COMPARISON_CHAIN,
        &copies::IFS_SAME_COND,
        &copies::IF_SAME_THEN_ELSE,
        &copies::MATCH_SAME_ARMS,
        &copy_iterator::COPY_ITERATOR,
        &dbg_macro::DBG_MACRO,
        &default_trait_access::DEFAULT_TRAIT_ACCESS,
        &derive::DERIVE_HASH_XOR_EQ,
        &derive::EXPL_IMPL_CLONE_ON_COPY,
        &doc::DOC_MARKDOWN,
        &doc::MISSING_SAFETY_DOC,
        &doc::NEEDLESS_DOCTEST_MAIN,
        &double_comparison::DOUBLE_COMPARISONS,
        &double_parens::DOUBLE_PARENS,
        &drop_bounds::DROP_BOUNDS,
        &drop_forget_ref::DROP_COPY,
        &drop_forget_ref::DROP_REF,
        &drop_forget_ref::FORGET_COPY,
        &drop_forget_ref::FORGET_REF,
        &duration_subsec::DURATION_SUBSEC,
        &else_if_without_else::ELSE_IF_WITHOUT_ELSE,
        &empty_enum::EMPTY_ENUM,
        &entry::MAP_ENTRY,
        &enum_clike::ENUM_CLIKE_UNPORTABLE_VARIANT,
        &enum_glob_use::ENUM_GLOB_USE,
        &enum_variants::ENUM_VARIANT_NAMES,
        &enum_variants::MODULE_INCEPTION,
        &enum_variants::MODULE_NAME_REPETITIONS,
        &enum_variants::PUB_ENUM_VARIANT_NAMES,
        &eq_op::EQ_OP,
        &eq_op::OP_REF,
        &erasing_op::ERASING_OP,
        &escape::BOXED_LOCAL,
        &eta_reduction::REDUNDANT_CLOSURE,
        &eta_reduction::REDUNDANT_CLOSURE_FOR_METHOD_CALLS,
        &eval_order_dependence::DIVERGING_SUB_EXPRESSION,
        &eval_order_dependence::EVAL_ORDER_DEPENDENCE,
        &excessive_precision::EXCESSIVE_PRECISION,
        &explicit_write::EXPLICIT_WRITE,
        &fallible_impl_from::FALLIBLE_IMPL_FROM,
        &format::USELESS_FORMAT,
        &formatting::POSSIBLE_MISSING_COMMA,
        &formatting::SUSPICIOUS_ASSIGNMENT_FORMATTING,
        &formatting::SUSPICIOUS_ELSE_FORMATTING,
        &formatting::SUSPICIOUS_UNARY_OP_FORMATTING,
        &functions::DOUBLE_MUST_USE,
        &functions::MUST_USE_CANDIDATE,
        &functions::MUST_USE_UNIT,
        &functions::NOT_UNSAFE_PTR_ARG_DEREF,
        &functions::TOO_MANY_ARGUMENTS,
        &functions::TOO_MANY_LINES,
        &get_last_with_len::GET_LAST_WITH_LEN,
        &identity_conversion::IDENTITY_CONVERSION,
        &identity_op::IDENTITY_OP,
        &if_not_else::IF_NOT_ELSE,
        &implicit_return::IMPLICIT_RETURN,
        &indexing_slicing::INDEXING_SLICING,
        &indexing_slicing::OUT_OF_BOUNDS_INDEXING,
        &infallible_destructuring_match::INFALLIBLE_DESTRUCTURING_MATCH,
        &infinite_iter::INFINITE_ITER,
        &infinite_iter::MAYBE_INFINITE_ITER,
        &inherent_impl::MULTIPLE_INHERENT_IMPL,
        &inherent_to_string::INHERENT_TO_STRING,
        &inherent_to_string::INHERENT_TO_STRING_SHADOW_DISPLAY,
        &inline_fn_without_body::INLINE_FN_WITHOUT_BODY,
        &int_plus_one::INT_PLUS_ONE,
        &integer_division::INTEGER_DIVISION,
        &items_after_statements::ITEMS_AFTER_STATEMENTS,
        &large_enum_variant::LARGE_ENUM_VARIANT,
        &len_zero::LEN_WITHOUT_IS_EMPTY,
        &len_zero::LEN_ZERO,
        &let_if_seq::USELESS_LET_IF_SEQ,
        &lifetimes::EXTRA_UNUSED_LIFETIMES,
        &lifetimes::NEEDLESS_LIFETIMES,
        &literal_representation::DECIMAL_LITERAL_REPRESENTATION,
        &literal_representation::INCONSISTENT_DIGIT_GROUPING,
        &literal_representation::LARGE_DIGIT_GROUPS,
        &literal_representation::MISTYPED_LITERAL_SUFFIXES,
        &literal_representation::UNREADABLE_LITERAL,
        &loops::EMPTY_LOOP,
        &loops::EXPLICIT_COUNTER_LOOP,
        &loops::EXPLICIT_INTO_ITER_LOOP,
        &loops::EXPLICIT_ITER_LOOP,
        &loops::FOR_KV_MAP,
        &loops::FOR_LOOP_OVER_OPTION,
        &loops::FOR_LOOP_OVER_RESULT,
        &loops::ITER_NEXT_LOOP,
        &loops::MANUAL_MEMCPY,
        &loops::MUT_RANGE_BOUND,
        &loops::NEEDLESS_COLLECT,
        &loops::NEEDLESS_RANGE_LOOP,
        &loops::NEVER_LOOP,
        &loops::REVERSE_RANGE_LOOP,
        &loops::WHILE_IMMUTABLE_CONDITION,
        &loops::WHILE_LET_LOOP,
        &loops::WHILE_LET_ON_ITERATOR,
        &main_recursion::MAIN_RECURSION,
        &map_clone::MAP_CLONE,
        &map_unit_fn::OPTION_MAP_UNIT_FN,
        &map_unit_fn::RESULT_MAP_UNIT_FN,
        &matches::MATCH_AS_REF,
        &matches::MATCH_BOOL,
        &matches::MATCH_OVERLAPPING_ARM,
        &matches::MATCH_REF_PATS,
        &matches::MATCH_WILD_ERR_ARM,
        &matches::SINGLE_MATCH,
        &matches::SINGLE_MATCH_ELSE,
        &matches::WILDCARD_ENUM_MATCH_ARM,
        &mem_discriminant::MEM_DISCRIMINANT_NON_ENUM,
        &mem_forget::MEM_FORGET,
        &mem_replace::MEM_REPLACE_OPTION_WITH_NONE,
        &mem_replace::MEM_REPLACE_WITH_UNINIT,
        &methods::CHARS_LAST_CMP,
        &methods::CHARS_NEXT_CMP,
        &methods::CLONE_DOUBLE_REF,
        &methods::CLONE_ON_COPY,
        &methods::CLONE_ON_REF_PTR,
        &methods::EXPECT_FUN_CALL,
        &methods::FILTER_MAP,
        &methods::FILTER_MAP_NEXT,
        &methods::FILTER_NEXT,
        &methods::FIND_MAP,
        &methods::FLAT_MAP_IDENTITY,
        &methods::GET_UNWRAP,
        &methods::INEFFICIENT_TO_STRING,
        &methods::INTO_ITER_ON_ARRAY,
        &methods::INTO_ITER_ON_REF,
        &methods::ITER_CLONED_COLLECT,
        &methods::ITER_NTH,
        &methods::ITER_SKIP_NEXT,
        &methods::MANUAL_SATURATING_ARITHMETIC,
        &methods::MAP_FLATTEN,
        &methods::NEW_RET_NO_SELF,
        &methods::OK_EXPECT,
        &methods::OPTION_AND_THEN_SOME,
        &methods::OPTION_EXPECT_USED,
        &methods::OPTION_MAP_OR_NONE,
        &methods::OPTION_MAP_UNWRAP_OR,
        &methods::OPTION_MAP_UNWRAP_OR_ELSE,
        &methods::OPTION_UNWRAP_USED,
        &methods::OR_FUN_CALL,
        &methods::RESULT_EXPECT_USED,
        &methods::RESULT_MAP_UNWRAP_OR_ELSE,
        &methods::RESULT_UNWRAP_USED,
        &methods::SEARCH_IS_SOME,
        &methods::SHOULD_IMPLEMENT_TRAIT,
        &methods::SINGLE_CHAR_PATTERN,
        &methods::STRING_EXTEND_CHARS,
        &methods::SUSPICIOUS_MAP,
        &methods::TEMPORARY_CSTRING_AS_PTR,
        &methods::UNINIT_ASSUMED_INIT,
        &methods::UNNECESSARY_FILTER_MAP,
        &methods::UNNECESSARY_FOLD,
        &methods::USELESS_ASREF,
        &methods::WRONG_PUB_SELF_CONVENTION,
        &methods::WRONG_SELF_CONVENTION,
        &minmax::MIN_MAX,
        &misc::CMP_NAN,
        &misc::CMP_OWNED,
        &misc::FLOAT_CMP,
        &misc::FLOAT_CMP_CONST,
        &misc::MODULO_ONE,
        &misc::SHORT_CIRCUIT_STATEMENT,
        &misc::TOPLEVEL_REF_ARG,
        &misc::USED_UNDERSCORE_BINDING,
        &misc::ZERO_PTR,
        &misc_early::BUILTIN_TYPE_SHADOW,
        &misc_early::DOUBLE_NEG,
        &misc_early::DUPLICATE_UNDERSCORE_ARGUMENT,
        &misc_early::MIXED_CASE_HEX_LITERALS,
        &misc_early::REDUNDANT_CLOSURE_CALL,
        &misc_early::REDUNDANT_PATTERN,
        &misc_early::UNNEEDED_FIELD_PATTERN,
        &misc_early::UNNEEDED_WILDCARD_PATTERN,
        &misc_early::UNSEPARATED_LITERAL_SUFFIX,
        &misc_early::ZERO_PREFIXED_LITERAL,
        &missing_const_for_fn::MISSING_CONST_FOR_FN,
        &missing_doc::MISSING_DOCS_IN_PRIVATE_ITEMS,
        &missing_inline::MISSING_INLINE_IN_PUBLIC_ITEMS,
        &mul_add::MANUAL_MUL_ADD,
        &multiple_crate_versions::MULTIPLE_CRATE_VERSIONS,
        &mut_mut::MUT_MUT,
        &mut_reference::UNNECESSARY_MUT_PASSED,
        &mutable_debug_assertion::DEBUG_ASSERT_WITH_MUT_CALL,
        &mutex_atomic::MUTEX_ATOMIC,
        &mutex_atomic::MUTEX_INTEGER,
        &needless_bool::BOOL_COMPARISON,
        &needless_bool::NEEDLESS_BOOL,
        &needless_borrow::NEEDLESS_BORROW,
        &needless_borrowed_ref::NEEDLESS_BORROWED_REFERENCE,
        &needless_continue::NEEDLESS_CONTINUE,
        &needless_pass_by_value::NEEDLESS_PASS_BY_VALUE,
        &needless_update::NEEDLESS_UPDATE,
        &neg_cmp_op_on_partial_ord::NEG_CMP_OP_ON_PARTIAL_ORD,
        &neg_multiply::NEG_MULTIPLY,
        &new_without_default::NEW_WITHOUT_DEFAULT,
        &no_effect::NO_EFFECT,
        &no_effect::UNNECESSARY_OPERATION,
        &non_copy_const::BORROW_INTERIOR_MUTABLE_CONST,
        &non_copy_const::DECLARE_INTERIOR_MUTABLE_CONST,
        &non_expressive_names::JUST_UNDERSCORES_AND_DIGITS,
        &non_expressive_names::MANY_SINGLE_CHAR_NAMES,
        &non_expressive_names::SIMILAR_NAMES,
        &ok_if_let::IF_LET_SOME_RESULT,
        &open_options::NONSENSICAL_OPEN_OPTIONS,
        &overflow_check_conditional::OVERFLOW_CHECK_CONDITIONAL,
        &panic_unimplemented::PANIC,
        &panic_unimplemented::PANIC_PARAMS,
        &panic_unimplemented::TODO,
        &panic_unimplemented::UNIMPLEMENTED,
        &panic_unimplemented::UNREACHABLE,
        &partialeq_ne_impl::PARTIALEQ_NE_IMPL,
        &path_buf_push_overwrite::PATH_BUF_PUSH_OVERWRITE,
        &precedence::PRECEDENCE,
        &ptr::CMP_NULL,
        &ptr::MUT_FROM_REF,
        &ptr::PTR_ARG,
        &ptr_offset_with_cast::PTR_OFFSET_WITH_CAST,
        &question_mark::QUESTION_MARK,
        &ranges::ITERATOR_STEP_BY_ZERO,
        &ranges::RANGE_MINUS_ONE,
        &ranges::RANGE_PLUS_ONE,
        &ranges::RANGE_ZIP_WITH_LEN,
        &redundant_clone::REDUNDANT_CLONE,
        &redundant_field_names::REDUNDANT_FIELD_NAMES,
        &redundant_pattern_matching::REDUNDANT_PATTERN_MATCHING,
        &redundant_static_lifetimes::REDUNDANT_STATIC_LIFETIMES,
        &reference::DEREF_ADDROF,
        &reference::REF_IN_DEREF,
        &regex::INVALID_REGEX,
        &regex::REGEX_MACRO,
        &regex::TRIVIAL_REGEX,
        &replace_consts::REPLACE_CONSTS,
        &returns::LET_AND_RETURN,
        &returns::NEEDLESS_RETURN,
        &returns::UNUSED_UNIT,
        &serde_api::SERDE_API_MISUSE,
        &shadow::SHADOW_REUSE,
        &shadow::SHADOW_SAME,
        &shadow::SHADOW_UNRELATED,
        &slow_vector_initialization::SLOW_VECTOR_INITIALIZATION,
        &strings::STRING_ADD,
        &strings::STRING_ADD_ASSIGN,
        &strings::STRING_LIT_AS_BYTES,
        &suspicious_trait_impl::SUSPICIOUS_ARITHMETIC_IMPL,
        &suspicious_trait_impl::SUSPICIOUS_OP_ASSIGN_IMPL,
        &swap::ALMOST_SWAPPED,
        &swap::MANUAL_SWAP,
        &temporary_assignment::TEMPORARY_ASSIGNMENT,
        &trait_bounds::TYPE_REPETITION_IN_BOUNDS,
        &transmute::CROSSPOINTER_TRANSMUTE,
        &transmute::TRANSMUTE_BYTES_TO_STR,
        &transmute::TRANSMUTE_INT_TO_BOOL,
        &transmute::TRANSMUTE_INT_TO_CHAR,
        &transmute::TRANSMUTE_INT_TO_FLOAT,
        &transmute::TRANSMUTE_PTR_TO_PTR,
        &transmute::TRANSMUTE_PTR_TO_REF,
        &transmute::UNSOUND_COLLECTION_TRANSMUTE,
        &transmute::USELESS_TRANSMUTE,
        &transmute::WRONG_TRANSMUTE,
        &transmuting_null::TRANSMUTING_NULL,
        &trivially_copy_pass_by_ref::TRIVIALLY_COPY_PASS_BY_REF,
        &try_err::TRY_ERR,
        &types::ABSURD_EXTREME_COMPARISONS,
        &types::BORROWED_BOX,
        &types::BOX_VEC,
        &types::CAST_LOSSLESS,
        &types::CAST_POSSIBLE_TRUNCATION,
        &types::CAST_POSSIBLE_WRAP,
        &types::CAST_PRECISION_LOSS,
        &types::CAST_PTR_ALIGNMENT,
        &types::CAST_REF_TO_MUT,
        &types::CAST_SIGN_LOSS,
        &types::CHAR_LIT_AS_U8,
        &types::FN_TO_NUMERIC_CAST,
        &types::FN_TO_NUMERIC_CAST_WITH_TRUNCATION,
        &types::IMPLICIT_HASHER,
        &types::INVALID_UPCAST_COMPARISONS,
        &types::LET_UNIT_VALUE,
        &types::LINKEDLIST,
        &types::OPTION_OPTION,
        &types::TYPE_COMPLEXITY,
        &types::UNIT_ARG,
        &types::UNIT_CMP,
        &types::UNNECESSARY_CAST,
        &types::VEC_BOX,
        &unicode::NON_ASCII_LITERAL,
        &unicode::UNICODE_NOT_NFC,
        &unicode::ZERO_WIDTH_SPACE,
        &unsafe_removed_from_name::UNSAFE_REMOVED_FROM_NAME,
        &unused_io_amount::UNUSED_IO_AMOUNT,
        &unused_label::UNUSED_LABEL,
        &unused_self::UNUSED_SELF,
        &unwrap::PANICKING_UNWRAP,
        &unwrap::UNNECESSARY_UNWRAP,
        &use_self::USE_SELF,
        &vec::USELESS_VEC,
        &wildcard_dependencies::WILDCARD_DEPENDENCIES,
        &write::PRINTLN_EMPTY_STRING,
        &write::PRINT_LITERAL,
        &write::PRINT_STDOUT,
        &write::PRINT_WITH_NEWLINE,
        &write::USE_DEBUG,
        &write::WRITELN_EMPTY_STRING,
        &write::WRITE_LITERAL,
        &write::WRITE_WITH_NEWLINE,
        &zero_div_zero::ZERO_DIVIDED_BY_ZERO,
    ]);
    // end register lints, do not remove this comment, it’s used in `update_lints`

    store.register_late_pass(|| box serde_api::SerdeAPI);
    store.register_late_pass(|| box utils::internal_lints::CompilerLintFunctions::new());
    store.register_late_pass(|| box utils::internal_lints::LintWithoutLintPass::default());
    store.register_late_pass(|| box utils::internal_lints::OuterExpnDataPass);
    store.register_late_pass(|| box utils::inspector::DeepCodeInspector);
    store.register_late_pass(|| box utils::author::Author);
    store.register_late_pass(|| box types::Types);
    store.register_late_pass(|| box booleans::NonminimalBool);
    store.register_late_pass(|| box eq_op::EqOp);
    store.register_late_pass(|| box enum_glob_use::EnumGlobUse);
    store.register_late_pass(|| box enum_clike::UnportableVariant);
    store.register_late_pass(|| box excessive_precision::ExcessivePrecision);
    let verbose_bit_mask_threshold = conf.verbose_bit_mask_threshold;
    store.register_late_pass(move || box bit_mask::BitMask::new(verbose_bit_mask_threshold));
    store.register_late_pass(|| box ptr::Ptr);
    store.register_late_pass(|| box needless_bool::NeedlessBool);
    store.register_late_pass(|| box needless_bool::BoolComparison);
    store.register_late_pass(|| box approx_const::ApproxConstant);
    store.register_late_pass(|| box misc::MiscLints);
    store.register_late_pass(|| box eta_reduction::EtaReduction);
    store.register_late_pass(|| box identity_op::IdentityOp);
    store.register_late_pass(|| box erasing_op::ErasingOp);
    store.register_late_pass(|| box mut_mut::MutMut);
    store.register_late_pass(|| box mut_reference::UnnecessaryMutPassed);
    store.register_late_pass(|| box len_zero::LenZero);
    store.register_late_pass(|| box attrs::Attributes);
    store.register_late_pass(|| box block_in_if_condition::BlockInIfCondition);
    store.register_late_pass(|| box unicode::Unicode);
    store.register_late_pass(|| box strings::StringAdd);
    store.register_late_pass(|| box implicit_return::ImplicitReturn);
    store.register_late_pass(|| box methods::Methods);
    store.register_late_pass(|| box map_clone::MapClone);
    store.register_late_pass(|| box shadow::Shadow);
    store.register_late_pass(|| box types::LetUnitValue);
    store.register_late_pass(|| box types::UnitCmp);
    store.register_late_pass(|| box loops::Loops);
    store.register_late_pass(|| box main_recursion::MainRecursion::default());
    store.register_late_pass(|| box lifetimes::Lifetimes);
    store.register_late_pass(|| box entry::HashMapPass);
    store.register_late_pass(|| box ranges::Ranges);
    store.register_late_pass(|| box types::Casts);
    let type_complexity_threshold = conf.type_complexity_threshold;
    store.register_late_pass(move || box types::TypeComplexity::new(type_complexity_threshold));
    store.register_late_pass(|| box matches::Matches);
    store.register_late_pass(|| box minmax::MinMaxPass);
    store.register_late_pass(|| box open_options::OpenOptions);
    store.register_late_pass(|| box zero_div_zero::ZeroDiv);
    store.register_late_pass(|| box mutex_atomic::Mutex);
    store.register_late_pass(|| box needless_update::NeedlessUpdate);
    store.register_late_pass(|| box needless_borrow::NeedlessBorrow::default());
    store.register_late_pass(|| box needless_borrowed_ref::NeedlessBorrowedRef);
    store.register_late_pass(|| box no_effect::NoEffect);
    store.register_late_pass(|| box temporary_assignment::TemporaryAssignment);
    store.register_late_pass(|| box transmute::Transmute);
    let cognitive_complexity_threshold = conf.cognitive_complexity_threshold;
    store.register_late_pass(move || box cognitive_complexity::CognitiveComplexity::new(cognitive_complexity_threshold));
    let too_large_for_stack = conf.too_large_for_stack;
    store.register_late_pass(move || box escape::BoxedLocal{too_large_for_stack});
    store.register_late_pass(|| box panic_unimplemented::PanicUnimplemented);
    store.register_late_pass(|| box strings::StringLitAsBytes);
    store.register_late_pass(|| box derive::Derive);
    store.register_late_pass(|| box types::CharLitAsU8);
    store.register_late_pass(|| box vec::UselessVec);
    store.register_late_pass(|| box drop_bounds::DropBounds);
    store.register_late_pass(|| box get_last_with_len::GetLastWithLen);
    store.register_late_pass(|| box drop_forget_ref::DropForgetRef);
    store.register_late_pass(|| box empty_enum::EmptyEnum);
    store.register_late_pass(|| box types::AbsurdExtremeComparisons);
    store.register_late_pass(|| box types::InvalidUpcastComparisons);
    store.register_late_pass(|| box regex::Regex::default());
    store.register_late_pass(|| box copies::CopyAndPaste);
    store.register_late_pass(|| box copy_iterator::CopyIterator);
    store.register_late_pass(|| box format::UselessFormat);
    store.register_late_pass(|| box swap::Swap);
    store.register_late_pass(|| box overflow_check_conditional::OverflowCheckConditional);
    store.register_late_pass(|| box unused_label::UnusedLabel);
    store.register_late_pass(|| box new_without_default::NewWithoutDefault::default());
    let blacklisted_names = conf.blacklisted_names.iter().cloned().collect::<FxHashSet<_>>();
    store.register_late_pass(move || box blacklisted_name::BlacklistedName::new(blacklisted_names.clone()));
    let too_many_arguments_threshold1 = conf.too_many_arguments_threshold;
    let too_many_lines_threshold2 = conf.too_many_lines_threshold;
    store.register_late_pass(move || box functions::Functions::new(too_many_arguments_threshold1, too_many_lines_threshold2));
    let doc_valid_idents = conf.doc_valid_idents.iter().cloned().collect::<FxHashSet<_>>();
    store.register_late_pass(move || box doc::DocMarkdown::new(doc_valid_idents.clone()));
    store.register_late_pass(|| box neg_multiply::NegMultiply);
    store.register_late_pass(|| box mem_discriminant::MemDiscriminant);
    store.register_late_pass(|| box mem_forget::MemForget);
    store.register_late_pass(|| box mem_replace::MemReplace);
    store.register_late_pass(|| box arithmetic::Arithmetic::default());
    store.register_late_pass(|| box assign_ops::AssignOps);
    store.register_late_pass(|| box let_if_seq::LetIfSeq);
    store.register_late_pass(|| box eval_order_dependence::EvalOrderDependence);
    store.register_late_pass(|| box missing_doc::MissingDoc::new());
    store.register_late_pass(|| box missing_inline::MissingInline);
    store.register_late_pass(|| box ok_if_let::OkIfLet);
    store.register_late_pass(|| box redundant_pattern_matching::RedundantPatternMatching);
    store.register_late_pass(|| box partialeq_ne_impl::PartialEqNeImpl);
    store.register_late_pass(|| box unused_io_amount::UnusedIoAmount);
    let enum_variant_size_threshold = conf.enum_variant_size_threshold;
    store.register_late_pass(move || box large_enum_variant::LargeEnumVariant::new(enum_variant_size_threshold));
    store.register_late_pass(|| box explicit_write::ExplicitWrite);
    store.register_late_pass(|| box needless_pass_by_value::NeedlessPassByValue);
    let trivially_copy_pass_by_ref = trivially_copy_pass_by_ref::TriviallyCopyPassByRef::new(
        conf.trivial_copy_size_limit,
        &sess.target,
    );
    store.register_late_pass(move || box trivially_copy_pass_by_ref);
    store.register_late_pass(|| box try_err::TryErr);
    store.register_late_pass(|| box use_self::UseSelf);
    store.register_late_pass(|| box bytecount::ByteCount);
    store.register_late_pass(|| box infinite_iter::InfiniteIter);
    store.register_late_pass(|| box inline_fn_without_body::InlineFnWithoutBody);
    store.register_late_pass(|| box identity_conversion::IdentityConversion::default());
    store.register_late_pass(|| box types::ImplicitHasher);
    store.register_late_pass(|| box fallible_impl_from::FallibleImplFrom);
    store.register_late_pass(|| box replace_consts::ReplaceConsts);
    store.register_late_pass(|| box types::UnitArg);
    store.register_late_pass(|| box double_comparison::DoubleComparisons);
    store.register_late_pass(|| box question_mark::QuestionMark);
    store.register_late_pass(|| box suspicious_trait_impl::SuspiciousImpl);
    store.register_late_pass(|| box map_unit_fn::MapUnit);
    store.register_late_pass(|| box infallible_destructuring_match::InfallibleDestructingMatch);
    store.register_late_pass(|| box inherent_impl::MultipleInherentImpl::default());
    store.register_late_pass(|| box neg_cmp_op_on_partial_ord::NoNegCompOpForPartialOrd);
    store.register_late_pass(|| box unwrap::Unwrap);
    store.register_late_pass(|| box duration_subsec::DurationSubsec);
    store.register_late_pass(|| box default_trait_access::DefaultTraitAccess);
    store.register_late_pass(|| box indexing_slicing::IndexingSlicing);
    store.register_late_pass(|| box non_copy_const::NonCopyConst);
    store.register_late_pass(|| box ptr_offset_with_cast::PtrOffsetWithCast);
    store.register_late_pass(|| box redundant_clone::RedundantClone);
    store.register_late_pass(|| box slow_vector_initialization::SlowVectorInit);
    store.register_late_pass(|| box types::RefToMut);
    store.register_late_pass(|| box assertions_on_constants::AssertionsOnConstants);
    store.register_late_pass(|| box missing_const_for_fn::MissingConstForFn);
    store.register_late_pass(|| box transmuting_null::TransmutingNull);
    store.register_late_pass(|| box path_buf_push_overwrite::PathBufPushOverwrite);
    store.register_late_pass(|| box checked_conversions::CheckedConversions);
    store.register_late_pass(|| box integer_division::IntegerDivision);
    store.register_late_pass(|| box inherent_to_string::InherentToString);
    store.register_late_pass(|| box trait_bounds::TraitBounds);
    store.register_late_pass(|| box comparison_chain::ComparisonChain);
    store.register_late_pass(|| box mul_add::MulAddCheck);
    store.register_early_pass(|| box reference::DerefAddrOf);
    store.register_early_pass(|| box reference::RefInDeref);
    store.register_early_pass(|| box double_parens::DoubleParens);
    store.register_early_pass(|| box unsafe_removed_from_name::UnsafeNameRemoval);
    store.register_early_pass(|| box if_not_else::IfNotElse);
    store.register_early_pass(|| box else_if_without_else::ElseIfWithoutElse);
    store.register_early_pass(|| box int_plus_one::IntPlusOne);
    store.register_early_pass(|| box formatting::Formatting);
    store.register_early_pass(|| box misc_early::MiscEarlyLints);
    store.register_early_pass(|| box returns::Return);
    store.register_early_pass(|| box collapsible_if::CollapsibleIf);
    store.register_early_pass(|| box items_after_statements::ItemsAfterStatements);
    store.register_early_pass(|| box precedence::Precedence);
    store.register_early_pass(|| box needless_continue::NeedlessContinue);
    store.register_early_pass(|| box redundant_static_lifetimes::RedundantStaticLifetimes);
    store.register_early_pass(|| box cargo_common_metadata::CargoCommonMetadata);
    store.register_early_pass(|| box multiple_crate_versions::MultipleCrateVersions);
    store.register_early_pass(|| box wildcard_dependencies::WildcardDependencies);
    store.register_early_pass(|| box literal_representation::LiteralDigitGrouping);
    let literal_representation_threshold = conf.literal_representation_threshold;
    store.register_early_pass(move || box literal_representation::DecimalLiteralRepresentation::new(literal_representation_threshold));
    store.register_early_pass(|| box utils::internal_lints::ClippyLintsInternal);
    let enum_variant_name_threshold = conf.enum_variant_name_threshold;
    store.register_early_pass(move || box enum_variants::EnumVariantNames::new(enum_variant_name_threshold));
    store.register_late_pass(|| box unused_self::UnusedSelf);
    store.register_late_pass(|| box mutable_debug_assertion::DebugAssertWithMutCall);

    store.register_group(true, "clippy::restriction", Some("clippy_restriction"), vec![
        LintId::of(&arithmetic::FLOAT_ARITHMETIC),
        LintId::of(&arithmetic::INTEGER_ARITHMETIC),
        LintId::of(&dbg_macro::DBG_MACRO),
        LintId::of(&else_if_without_else::ELSE_IF_WITHOUT_ELSE),
        LintId::of(&implicit_return::IMPLICIT_RETURN),
        LintId::of(&indexing_slicing::INDEXING_SLICING),
        LintId::of(&inherent_impl::MULTIPLE_INHERENT_IMPL),
        LintId::of(&integer_division::INTEGER_DIVISION),
        LintId::of(&literal_representation::DECIMAL_LITERAL_REPRESENTATION),
        LintId::of(&matches::WILDCARD_ENUM_MATCH_ARM),
        LintId::of(&mem_forget::MEM_FORGET),
        LintId::of(&methods::CLONE_ON_REF_PTR),
        LintId::of(&methods::GET_UNWRAP),
        LintId::of(&methods::OPTION_EXPECT_USED),
        LintId::of(&methods::OPTION_UNWRAP_USED),
        LintId::of(&methods::RESULT_EXPECT_USED),
        LintId::of(&methods::RESULT_UNWRAP_USED),
        LintId::of(&methods::WRONG_PUB_SELF_CONVENTION),
        LintId::of(&misc::FLOAT_CMP_CONST),
        LintId::of(&missing_doc::MISSING_DOCS_IN_PRIVATE_ITEMS),
        LintId::of(&missing_inline::MISSING_INLINE_IN_PUBLIC_ITEMS),
        LintId::of(&panic_unimplemented::PANIC),
        LintId::of(&panic_unimplemented::TODO),
        LintId::of(&panic_unimplemented::UNIMPLEMENTED),
        LintId::of(&panic_unimplemented::UNREACHABLE),
        LintId::of(&shadow::SHADOW_REUSE),
        LintId::of(&shadow::SHADOW_SAME),
        LintId::of(&strings::STRING_ADD),
        LintId::of(&write::PRINT_STDOUT),
        LintId::of(&write::USE_DEBUG),
    ]);

    store.register_group(true, "clippy::pedantic", Some("clippy_pedantic"), vec![
        LintId::of(&attrs::INLINE_ALWAYS),
        LintId::of(&checked_conversions::CHECKED_CONVERSIONS),
        LintId::of(&copies::MATCH_SAME_ARMS),
        LintId::of(&copy_iterator::COPY_ITERATOR),
        LintId::of(&default_trait_access::DEFAULT_TRAIT_ACCESS),
        LintId::of(&derive::EXPL_IMPL_CLONE_ON_COPY),
        LintId::of(&doc::DOC_MARKDOWN),
        LintId::of(&empty_enum::EMPTY_ENUM),
        LintId::of(&enum_glob_use::ENUM_GLOB_USE),
        LintId::of(&enum_variants::MODULE_NAME_REPETITIONS),
        LintId::of(&enum_variants::PUB_ENUM_VARIANT_NAMES),
        LintId::of(&eta_reduction::REDUNDANT_CLOSURE_FOR_METHOD_CALLS),
        LintId::of(&functions::MUST_USE_CANDIDATE),
        LintId::of(&functions::TOO_MANY_LINES),
        LintId::of(&if_not_else::IF_NOT_ELSE),
        LintId::of(&infinite_iter::MAYBE_INFINITE_ITER),
        LintId::of(&items_after_statements::ITEMS_AFTER_STATEMENTS),
        LintId::of(&literal_representation::LARGE_DIGIT_GROUPS),
        LintId::of(&loops::EXPLICIT_INTO_ITER_LOOP),
        LintId::of(&loops::EXPLICIT_ITER_LOOP),
        LintId::of(&matches::SINGLE_MATCH_ELSE),
        LintId::of(&methods::FILTER_MAP),
        LintId::of(&methods::FILTER_MAP_NEXT),
        LintId::of(&methods::FIND_MAP),
        LintId::of(&methods::MAP_FLATTEN),
        LintId::of(&methods::OPTION_MAP_UNWRAP_OR),
        LintId::of(&methods::OPTION_MAP_UNWRAP_OR_ELSE),
        LintId::of(&methods::RESULT_MAP_UNWRAP_OR_ELSE),
        LintId::of(&misc::USED_UNDERSCORE_BINDING),
        LintId::of(&misc_early::UNSEPARATED_LITERAL_SUFFIX),
        LintId::of(&mut_mut::MUT_MUT),
        LintId::of(&needless_continue::NEEDLESS_CONTINUE),
        LintId::of(&needless_pass_by_value::NEEDLESS_PASS_BY_VALUE),
        LintId::of(&non_expressive_names::SIMILAR_NAMES),
        LintId::of(&replace_consts::REPLACE_CONSTS),
        LintId::of(&shadow::SHADOW_UNRELATED),
        LintId::of(&strings::STRING_ADD_ASSIGN),
        LintId::of(&trait_bounds::TYPE_REPETITION_IN_BOUNDS),
        LintId::of(&types::CAST_LOSSLESS),
        LintId::of(&types::CAST_POSSIBLE_TRUNCATION),
        LintId::of(&types::CAST_POSSIBLE_WRAP),
        LintId::of(&types::CAST_PRECISION_LOSS),
        LintId::of(&types::CAST_SIGN_LOSS),
        LintId::of(&types::INVALID_UPCAST_COMPARISONS),
        LintId::of(&types::LINKEDLIST),
        LintId::of(&unicode::NON_ASCII_LITERAL),
        LintId::of(&unicode::UNICODE_NOT_NFC),
        LintId::of(&unused_self::UNUSED_SELF),
        LintId::of(&use_self::USE_SELF),
    ]);

    store.register_group(true, "clippy::internal", Some("clippy_internal"), vec![
        LintId::of(&utils::internal_lints::CLIPPY_LINTS_INTERNAL),
        LintId::of(&utils::internal_lints::COMPILER_LINT_FUNCTIONS),
        LintId::of(&utils::internal_lints::LINT_WITHOUT_LINT_PASS),
        LintId::of(&utils::internal_lints::OUTER_EXPN_EXPN_DATA),
    ]);

    store.register_group(true, "clippy::all", Some("clippy"), vec![
        LintId::of(&approx_const::APPROX_CONSTANT),
        LintId::of(&assertions_on_constants::ASSERTIONS_ON_CONSTANTS),
        LintId::of(&assign_ops::ASSIGN_OP_PATTERN),
        LintId::of(&assign_ops::MISREFACTORED_ASSIGN_OP),
        LintId::of(&attrs::DEPRECATED_CFG_ATTR),
        LintId::of(&attrs::DEPRECATED_SEMVER),
        LintId::of(&attrs::UNKNOWN_CLIPPY_LINTS),
        LintId::of(&attrs::USELESS_ATTRIBUTE),
        LintId::of(&bit_mask::BAD_BIT_MASK),
        LintId::of(&bit_mask::INEFFECTIVE_BIT_MASK),
        LintId::of(&bit_mask::VERBOSE_BIT_MASK),
        LintId::of(&blacklisted_name::BLACKLISTED_NAME),
        LintId::of(&block_in_if_condition::BLOCK_IN_IF_CONDITION_EXPR),
        LintId::of(&block_in_if_condition::BLOCK_IN_IF_CONDITION_STMT),
        LintId::of(&booleans::LOGIC_BUG),
        LintId::of(&booleans::NONMINIMAL_BOOL),
        LintId::of(&bytecount::NAIVE_BYTECOUNT),
        LintId::of(&cognitive_complexity::COGNITIVE_COMPLEXITY),
        LintId::of(&collapsible_if::COLLAPSIBLE_IF),
        LintId::of(&comparison_chain::COMPARISON_CHAIN),
        LintId::of(&copies::IFS_SAME_COND),
        LintId::of(&copies::IF_SAME_THEN_ELSE),
        LintId::of(&derive::DERIVE_HASH_XOR_EQ),
        LintId::of(&doc::MISSING_SAFETY_DOC),
        LintId::of(&doc::NEEDLESS_DOCTEST_MAIN),
        LintId::of(&double_comparison::DOUBLE_COMPARISONS),
        LintId::of(&double_parens::DOUBLE_PARENS),
        LintId::of(&drop_bounds::DROP_BOUNDS),
        LintId::of(&drop_forget_ref::DROP_COPY),
        LintId::of(&drop_forget_ref::DROP_REF),
        LintId::of(&drop_forget_ref::FORGET_COPY),
        LintId::of(&drop_forget_ref::FORGET_REF),
        LintId::of(&duration_subsec::DURATION_SUBSEC),
        LintId::of(&entry::MAP_ENTRY),
        LintId::of(&enum_clike::ENUM_CLIKE_UNPORTABLE_VARIANT),
        LintId::of(&enum_variants::ENUM_VARIANT_NAMES),
        LintId::of(&enum_variants::MODULE_INCEPTION),
        LintId::of(&eq_op::EQ_OP),
        LintId::of(&eq_op::OP_REF),
        LintId::of(&erasing_op::ERASING_OP),
        LintId::of(&escape::BOXED_LOCAL),
        LintId::of(&eta_reduction::REDUNDANT_CLOSURE),
        LintId::of(&eval_order_dependence::DIVERGING_SUB_EXPRESSION),
        LintId::of(&eval_order_dependence::EVAL_ORDER_DEPENDENCE),
        LintId::of(&excessive_precision::EXCESSIVE_PRECISION),
        LintId::of(&explicit_write::EXPLICIT_WRITE),
        LintId::of(&format::USELESS_FORMAT),
        LintId::of(&formatting::POSSIBLE_MISSING_COMMA),
        LintId::of(&formatting::SUSPICIOUS_ASSIGNMENT_FORMATTING),
        LintId::of(&formatting::SUSPICIOUS_ELSE_FORMATTING),
        LintId::of(&formatting::SUSPICIOUS_UNARY_OP_FORMATTING),
        LintId::of(&functions::DOUBLE_MUST_USE),
        LintId::of(&functions::MUST_USE_UNIT),
        LintId::of(&functions::NOT_UNSAFE_PTR_ARG_DEREF),
        LintId::of(&functions::TOO_MANY_ARGUMENTS),
        LintId::of(&get_last_with_len::GET_LAST_WITH_LEN),
        LintId::of(&identity_conversion::IDENTITY_CONVERSION),
        LintId::of(&identity_op::IDENTITY_OP),
        LintId::of(&indexing_slicing::OUT_OF_BOUNDS_INDEXING),
        LintId::of(&infallible_destructuring_match::INFALLIBLE_DESTRUCTURING_MATCH),
        LintId::of(&infinite_iter::INFINITE_ITER),
        LintId::of(&inherent_to_string::INHERENT_TO_STRING),
        LintId::of(&inherent_to_string::INHERENT_TO_STRING_SHADOW_DISPLAY),
        LintId::of(&inline_fn_without_body::INLINE_FN_WITHOUT_BODY),
        LintId::of(&int_plus_one::INT_PLUS_ONE),
        LintId::of(&large_enum_variant::LARGE_ENUM_VARIANT),
        LintId::of(&len_zero::LEN_WITHOUT_IS_EMPTY),
        LintId::of(&len_zero::LEN_ZERO),
        LintId::of(&let_if_seq::USELESS_LET_IF_SEQ),
        LintId::of(&lifetimes::EXTRA_UNUSED_LIFETIMES),
        LintId::of(&lifetimes::NEEDLESS_LIFETIMES),
        LintId::of(&literal_representation::INCONSISTENT_DIGIT_GROUPING),
        LintId::of(&literal_representation::MISTYPED_LITERAL_SUFFIXES),
        LintId::of(&literal_representation::UNREADABLE_LITERAL),
        LintId::of(&loops::EMPTY_LOOP),
        LintId::of(&loops::EXPLICIT_COUNTER_LOOP),
        LintId::of(&loops::FOR_KV_MAP),
        LintId::of(&loops::FOR_LOOP_OVER_OPTION),
        LintId::of(&loops::FOR_LOOP_OVER_RESULT),
        LintId::of(&loops::ITER_NEXT_LOOP),
        LintId::of(&loops::MANUAL_MEMCPY),
        LintId::of(&loops::MUT_RANGE_BOUND),
        LintId::of(&loops::NEEDLESS_COLLECT),
        LintId::of(&loops::NEEDLESS_RANGE_LOOP),
        LintId::of(&loops::NEVER_LOOP),
        LintId::of(&loops::REVERSE_RANGE_LOOP),
        LintId::of(&loops::WHILE_IMMUTABLE_CONDITION),
        LintId::of(&loops::WHILE_LET_LOOP),
        LintId::of(&loops::WHILE_LET_ON_ITERATOR),
        LintId::of(&main_recursion::MAIN_RECURSION),
        LintId::of(&map_clone::MAP_CLONE),
        LintId::of(&map_unit_fn::OPTION_MAP_UNIT_FN),
        LintId::of(&map_unit_fn::RESULT_MAP_UNIT_FN),
        LintId::of(&matches::MATCH_AS_REF),
        LintId::of(&matches::MATCH_BOOL),
        LintId::of(&matches::MATCH_OVERLAPPING_ARM),
        LintId::of(&matches::MATCH_REF_PATS),
        LintId::of(&matches::MATCH_WILD_ERR_ARM),
        LintId::of(&matches::SINGLE_MATCH),
        LintId::of(&mem_discriminant::MEM_DISCRIMINANT_NON_ENUM),
        LintId::of(&mem_replace::MEM_REPLACE_OPTION_WITH_NONE),
        LintId::of(&mem_replace::MEM_REPLACE_WITH_UNINIT),
        LintId::of(&methods::CHARS_LAST_CMP),
        LintId::of(&methods::CHARS_NEXT_CMP),
        LintId::of(&methods::CLONE_DOUBLE_REF),
        LintId::of(&methods::CLONE_ON_COPY),
        LintId::of(&methods::EXPECT_FUN_CALL),
        LintId::of(&methods::FILTER_NEXT),
        LintId::of(&methods::FLAT_MAP_IDENTITY),
        LintId::of(&methods::INEFFICIENT_TO_STRING),
        LintId::of(&methods::INTO_ITER_ON_ARRAY),
        LintId::of(&methods::INTO_ITER_ON_REF),
        LintId::of(&methods::ITER_CLONED_COLLECT),
        LintId::of(&methods::ITER_NTH),
        LintId::of(&methods::ITER_SKIP_NEXT),
        LintId::of(&methods::MANUAL_SATURATING_ARITHMETIC),
        LintId::of(&methods::NEW_RET_NO_SELF),
        LintId::of(&methods::OK_EXPECT),
        LintId::of(&methods::OPTION_AND_THEN_SOME),
        LintId::of(&methods::OPTION_MAP_OR_NONE),
        LintId::of(&methods::OR_FUN_CALL),
        LintId::of(&methods::SEARCH_IS_SOME),
        LintId::of(&methods::SHOULD_IMPLEMENT_TRAIT),
        LintId::of(&methods::SINGLE_CHAR_PATTERN),
        LintId::of(&methods::STRING_EXTEND_CHARS),
        LintId::of(&methods::SUSPICIOUS_MAP),
        LintId::of(&methods::TEMPORARY_CSTRING_AS_PTR),
        LintId::of(&methods::UNINIT_ASSUMED_INIT),
        LintId::of(&methods::UNNECESSARY_FILTER_MAP),
        LintId::of(&methods::UNNECESSARY_FOLD),
        LintId::of(&methods::USELESS_ASREF),
        LintId::of(&methods::WRONG_SELF_CONVENTION),
        LintId::of(&minmax::MIN_MAX),
        LintId::of(&misc::CMP_NAN),
        LintId::of(&misc::CMP_OWNED),
        LintId::of(&misc::FLOAT_CMP),
        LintId::of(&misc::MODULO_ONE),
        LintId::of(&misc::SHORT_CIRCUIT_STATEMENT),
        LintId::of(&misc::TOPLEVEL_REF_ARG),
        LintId::of(&misc::ZERO_PTR),
        LintId::of(&misc_early::BUILTIN_TYPE_SHADOW),
        LintId::of(&misc_early::DOUBLE_NEG),
        LintId::of(&misc_early::DUPLICATE_UNDERSCORE_ARGUMENT),
        LintId::of(&misc_early::MIXED_CASE_HEX_LITERALS),
        LintId::of(&misc_early::REDUNDANT_CLOSURE_CALL),
        LintId::of(&misc_early::REDUNDANT_PATTERN),
        LintId::of(&misc_early::UNNEEDED_FIELD_PATTERN),
        LintId::of(&misc_early::UNNEEDED_WILDCARD_PATTERN),
        LintId::of(&misc_early::ZERO_PREFIXED_LITERAL),
        LintId::of(&mut_reference::UNNECESSARY_MUT_PASSED),
        LintId::of(&mutable_debug_assertion::DEBUG_ASSERT_WITH_MUT_CALL),
        LintId::of(&mutex_atomic::MUTEX_ATOMIC),
        LintId::of(&needless_bool::BOOL_COMPARISON),
        LintId::of(&needless_bool::NEEDLESS_BOOL),
        LintId::of(&needless_borrowed_ref::NEEDLESS_BORROWED_REFERENCE),
        LintId::of(&needless_update::NEEDLESS_UPDATE),
        LintId::of(&neg_cmp_op_on_partial_ord::NEG_CMP_OP_ON_PARTIAL_ORD),
        LintId::of(&neg_multiply::NEG_MULTIPLY),
        LintId::of(&new_without_default::NEW_WITHOUT_DEFAULT),
        LintId::of(&no_effect::NO_EFFECT),
        LintId::of(&no_effect::UNNECESSARY_OPERATION),
        LintId::of(&non_copy_const::BORROW_INTERIOR_MUTABLE_CONST),
        LintId::of(&non_copy_const::DECLARE_INTERIOR_MUTABLE_CONST),
        LintId::of(&non_expressive_names::JUST_UNDERSCORES_AND_DIGITS),
        LintId::of(&non_expressive_names::MANY_SINGLE_CHAR_NAMES),
        LintId::of(&ok_if_let::IF_LET_SOME_RESULT),
        LintId::of(&open_options::NONSENSICAL_OPEN_OPTIONS),
        LintId::of(&overflow_check_conditional::OVERFLOW_CHECK_CONDITIONAL),
        LintId::of(&panic_unimplemented::PANIC_PARAMS),
        LintId::of(&partialeq_ne_impl::PARTIALEQ_NE_IMPL),
        LintId::of(&precedence::PRECEDENCE),
        LintId::of(&ptr::CMP_NULL),
        LintId::of(&ptr::MUT_FROM_REF),
        LintId::of(&ptr::PTR_ARG),
        LintId::of(&ptr_offset_with_cast::PTR_OFFSET_WITH_CAST),
        LintId::of(&question_mark::QUESTION_MARK),
        LintId::of(&ranges::ITERATOR_STEP_BY_ZERO),
        LintId::of(&ranges::RANGE_MINUS_ONE),
        LintId::of(&ranges::RANGE_PLUS_ONE),
        LintId::of(&ranges::RANGE_ZIP_WITH_LEN),
        LintId::of(&redundant_clone::REDUNDANT_CLONE),
        LintId::of(&redundant_field_names::REDUNDANT_FIELD_NAMES),
        LintId::of(&redundant_pattern_matching::REDUNDANT_PATTERN_MATCHING),
        LintId::of(&redundant_static_lifetimes::REDUNDANT_STATIC_LIFETIMES),
        LintId::of(&reference::DEREF_ADDROF),
        LintId::of(&reference::REF_IN_DEREF),
        LintId::of(&regex::INVALID_REGEX),
        LintId::of(&regex::REGEX_MACRO),
        LintId::of(&regex::TRIVIAL_REGEX),
        LintId::of(&returns::LET_AND_RETURN),
        LintId::of(&returns::NEEDLESS_RETURN),
        LintId::of(&returns::UNUSED_UNIT),
        LintId::of(&serde_api::SERDE_API_MISUSE),
        LintId::of(&slow_vector_initialization::SLOW_VECTOR_INITIALIZATION),
        LintId::of(&strings::STRING_LIT_AS_BYTES),
        LintId::of(&suspicious_trait_impl::SUSPICIOUS_ARITHMETIC_IMPL),
        LintId::of(&suspicious_trait_impl::SUSPICIOUS_OP_ASSIGN_IMPL),
        LintId::of(&swap::ALMOST_SWAPPED),
        LintId::of(&swap::MANUAL_SWAP),
        LintId::of(&temporary_assignment::TEMPORARY_ASSIGNMENT),
        LintId::of(&transmute::CROSSPOINTER_TRANSMUTE),
        LintId::of(&transmute::TRANSMUTE_BYTES_TO_STR),
        LintId::of(&transmute::TRANSMUTE_INT_TO_BOOL),
        LintId::of(&transmute::TRANSMUTE_INT_TO_CHAR),
        LintId::of(&transmute::TRANSMUTE_INT_TO_FLOAT),
        LintId::of(&transmute::TRANSMUTE_PTR_TO_PTR),
        LintId::of(&transmute::TRANSMUTE_PTR_TO_REF),
        LintId::of(&transmute::UNSOUND_COLLECTION_TRANSMUTE),
        LintId::of(&transmute::USELESS_TRANSMUTE),
        LintId::of(&transmute::WRONG_TRANSMUTE),
        LintId::of(&transmuting_null::TRANSMUTING_NULL),
        LintId::of(&trivially_copy_pass_by_ref::TRIVIALLY_COPY_PASS_BY_REF),
        LintId::of(&try_err::TRY_ERR),
        LintId::of(&types::ABSURD_EXTREME_COMPARISONS),
        LintId::of(&types::BORROWED_BOX),
        LintId::of(&types::BOX_VEC),
        LintId::of(&types::CAST_PTR_ALIGNMENT),
        LintId::of(&types::CAST_REF_TO_MUT),
        LintId::of(&types::CHAR_LIT_AS_U8),
        LintId::of(&types::FN_TO_NUMERIC_CAST),
        LintId::of(&types::FN_TO_NUMERIC_CAST_WITH_TRUNCATION),
        LintId::of(&types::IMPLICIT_HASHER),
        LintId::of(&types::LET_UNIT_VALUE),
        LintId::of(&types::OPTION_OPTION),
        LintId::of(&types::TYPE_COMPLEXITY),
        LintId::of(&types::UNIT_ARG),
        LintId::of(&types::UNIT_CMP),
        LintId::of(&types::UNNECESSARY_CAST),
        LintId::of(&types::VEC_BOX),
        LintId::of(&unicode::ZERO_WIDTH_SPACE),
        LintId::of(&unsafe_removed_from_name::UNSAFE_REMOVED_FROM_NAME),
        LintId::of(&unused_io_amount::UNUSED_IO_AMOUNT),
        LintId::of(&unused_label::UNUSED_LABEL),
        LintId::of(&unwrap::PANICKING_UNWRAP),
        LintId::of(&unwrap::UNNECESSARY_UNWRAP),
        LintId::of(&vec::USELESS_VEC),
        LintId::of(&write::PRINTLN_EMPTY_STRING),
        LintId::of(&write::PRINT_LITERAL),
        LintId::of(&write::PRINT_WITH_NEWLINE),
        LintId::of(&write::WRITELN_EMPTY_STRING),
        LintId::of(&write::WRITE_LITERAL),
        LintId::of(&write::WRITE_WITH_NEWLINE),
        LintId::of(&zero_div_zero::ZERO_DIVIDED_BY_ZERO),
    ]);

    store.register_group(true, "clippy::style", Some("clippy_style"), vec![
        LintId::of(&assertions_on_constants::ASSERTIONS_ON_CONSTANTS),
        LintId::of(&assign_ops::ASSIGN_OP_PATTERN),
        LintId::of(&attrs::UNKNOWN_CLIPPY_LINTS),
        LintId::of(&bit_mask::VERBOSE_BIT_MASK),
        LintId::of(&blacklisted_name::BLACKLISTED_NAME),
        LintId::of(&block_in_if_condition::BLOCK_IN_IF_CONDITION_EXPR),
        LintId::of(&block_in_if_condition::BLOCK_IN_IF_CONDITION_STMT),
        LintId::of(&collapsible_if::COLLAPSIBLE_IF),
        LintId::of(&comparison_chain::COMPARISON_CHAIN),
        LintId::of(&doc::MISSING_SAFETY_DOC),
        LintId::of(&doc::NEEDLESS_DOCTEST_MAIN),
        LintId::of(&enum_variants::ENUM_VARIANT_NAMES),
        LintId::of(&enum_variants::MODULE_INCEPTION),
        LintId::of(&eq_op::OP_REF),
        LintId::of(&eta_reduction::REDUNDANT_CLOSURE),
        LintId::of(&excessive_precision::EXCESSIVE_PRECISION),
        LintId::of(&formatting::SUSPICIOUS_ASSIGNMENT_FORMATTING),
        LintId::of(&formatting::SUSPICIOUS_ELSE_FORMATTING),
        LintId::of(&formatting::SUSPICIOUS_UNARY_OP_FORMATTING),
        LintId::of(&functions::DOUBLE_MUST_USE),
        LintId::of(&functions::MUST_USE_UNIT),
        LintId::of(&infallible_destructuring_match::INFALLIBLE_DESTRUCTURING_MATCH),
        LintId::of(&inherent_to_string::INHERENT_TO_STRING),
        LintId::of(&len_zero::LEN_WITHOUT_IS_EMPTY),
        LintId::of(&len_zero::LEN_ZERO),
        LintId::of(&let_if_seq::USELESS_LET_IF_SEQ),
        LintId::of(&literal_representation::INCONSISTENT_DIGIT_GROUPING),
        LintId::of(&literal_representation::UNREADABLE_LITERAL),
        LintId::of(&loops::EMPTY_LOOP),
        LintId::of(&loops::FOR_KV_MAP),
        LintId::of(&loops::NEEDLESS_RANGE_LOOP),
        LintId::of(&loops::WHILE_LET_ON_ITERATOR),
        LintId::of(&main_recursion::MAIN_RECURSION),
        LintId::of(&map_clone::MAP_CLONE),
        LintId::of(&matches::MATCH_BOOL),
        LintId::of(&matches::MATCH_OVERLAPPING_ARM),
        LintId::of(&matches::MATCH_REF_PATS),
        LintId::of(&matches::MATCH_WILD_ERR_ARM),
        LintId::of(&matches::SINGLE_MATCH),
        LintId::of(&mem_replace::MEM_REPLACE_OPTION_WITH_NONE),
        LintId::of(&methods::CHARS_LAST_CMP),
        LintId::of(&methods::INTO_ITER_ON_REF),
        LintId::of(&methods::ITER_CLONED_COLLECT),
        LintId::of(&methods::ITER_SKIP_NEXT),
        LintId::of(&methods::MANUAL_SATURATING_ARITHMETIC),
        LintId::of(&methods::NEW_RET_NO_SELF),
        LintId::of(&methods::OK_EXPECT),
        LintId::of(&methods::OPTION_MAP_OR_NONE),
        LintId::of(&methods::SHOULD_IMPLEMENT_TRAIT),
        LintId::of(&methods::STRING_EXTEND_CHARS),
        LintId::of(&methods::UNNECESSARY_FOLD),
        LintId::of(&methods::WRONG_SELF_CONVENTION),
        LintId::of(&misc::TOPLEVEL_REF_ARG),
        LintId::of(&misc::ZERO_PTR),
        LintId::of(&misc_early::BUILTIN_TYPE_SHADOW),
        LintId::of(&misc_early::DOUBLE_NEG),
        LintId::of(&misc_early::DUPLICATE_UNDERSCORE_ARGUMENT),
        LintId::of(&misc_early::MIXED_CASE_HEX_LITERALS),
        LintId::of(&misc_early::REDUNDANT_PATTERN),
        LintId::of(&misc_early::UNNEEDED_FIELD_PATTERN),
        LintId::of(&mut_reference::UNNECESSARY_MUT_PASSED),
        LintId::of(&neg_multiply::NEG_MULTIPLY),
        LintId::of(&new_without_default::NEW_WITHOUT_DEFAULT),
        LintId::of(&non_expressive_names::JUST_UNDERSCORES_AND_DIGITS),
        LintId::of(&non_expressive_names::MANY_SINGLE_CHAR_NAMES),
        LintId::of(&ok_if_let::IF_LET_SOME_RESULT),
        LintId::of(&panic_unimplemented::PANIC_PARAMS),
        LintId::of(&ptr::CMP_NULL),
        LintId::of(&ptr::PTR_ARG),
        LintId::of(&question_mark::QUESTION_MARK),
        LintId::of(&redundant_field_names::REDUNDANT_FIELD_NAMES),
        LintId::of(&redundant_pattern_matching::REDUNDANT_PATTERN_MATCHING),
        LintId::of(&redundant_static_lifetimes::REDUNDANT_STATIC_LIFETIMES),
        LintId::of(&regex::REGEX_MACRO),
        LintId::of(&regex::TRIVIAL_REGEX),
        LintId::of(&returns::LET_AND_RETURN),
        LintId::of(&returns::NEEDLESS_RETURN),
        LintId::of(&returns::UNUSED_UNIT),
        LintId::of(&strings::STRING_LIT_AS_BYTES),
        LintId::of(&try_err::TRY_ERR),
        LintId::of(&types::FN_TO_NUMERIC_CAST),
        LintId::of(&types::FN_TO_NUMERIC_CAST_WITH_TRUNCATION),
        LintId::of(&types::IMPLICIT_HASHER),
        LintId::of(&types::LET_UNIT_VALUE),
        LintId::of(&unsafe_removed_from_name::UNSAFE_REMOVED_FROM_NAME),
        LintId::of(&write::PRINTLN_EMPTY_STRING),
        LintId::of(&write::PRINT_LITERAL),
        LintId::of(&write::PRINT_WITH_NEWLINE),
        LintId::of(&write::WRITELN_EMPTY_STRING),
        LintId::of(&write::WRITE_LITERAL),
        LintId::of(&write::WRITE_WITH_NEWLINE),
    ]);

    store.register_group(true, "clippy::complexity", Some("clippy_complexity"), vec![
        LintId::of(&assign_ops::MISREFACTORED_ASSIGN_OP),
        LintId::of(&attrs::DEPRECATED_CFG_ATTR),
        LintId::of(&booleans::NONMINIMAL_BOOL),
        LintId::of(&cognitive_complexity::COGNITIVE_COMPLEXITY),
        LintId::of(&double_comparison::DOUBLE_COMPARISONS),
        LintId::of(&double_parens::DOUBLE_PARENS),
        LintId::of(&duration_subsec::DURATION_SUBSEC),
        LintId::of(&eval_order_dependence::DIVERGING_SUB_EXPRESSION),
        LintId::of(&eval_order_dependence::EVAL_ORDER_DEPENDENCE),
        LintId::of(&explicit_write::EXPLICIT_WRITE),
        LintId::of(&format::USELESS_FORMAT),
        LintId::of(&functions::TOO_MANY_ARGUMENTS),
        LintId::of(&get_last_with_len::GET_LAST_WITH_LEN),
        LintId::of(&identity_conversion::IDENTITY_CONVERSION),
        LintId::of(&identity_op::IDENTITY_OP),
        LintId::of(&int_plus_one::INT_PLUS_ONE),
        LintId::of(&lifetimes::EXTRA_UNUSED_LIFETIMES),
        LintId::of(&lifetimes::NEEDLESS_LIFETIMES),
        LintId::of(&loops::EXPLICIT_COUNTER_LOOP),
        LintId::of(&loops::MUT_RANGE_BOUND),
        LintId::of(&loops::WHILE_LET_LOOP),
        LintId::of(&map_unit_fn::OPTION_MAP_UNIT_FN),
        LintId::of(&map_unit_fn::RESULT_MAP_UNIT_FN),
        LintId::of(&matches::MATCH_AS_REF),
        LintId::of(&methods::CHARS_NEXT_CMP),
        LintId::of(&methods::CLONE_ON_COPY),
        LintId::of(&methods::FILTER_NEXT),
        LintId::of(&methods::FLAT_MAP_IDENTITY),
        LintId::of(&methods::OPTION_AND_THEN_SOME),
        LintId::of(&methods::SEARCH_IS_SOME),
        LintId::of(&methods::SUSPICIOUS_MAP),
        LintId::of(&methods::UNNECESSARY_FILTER_MAP),
        LintId::of(&methods::USELESS_ASREF),
        LintId::of(&misc::SHORT_CIRCUIT_STATEMENT),
        LintId::of(&misc_early::REDUNDANT_CLOSURE_CALL),
        LintId::of(&misc_early::UNNEEDED_WILDCARD_PATTERN),
        LintId::of(&misc_early::ZERO_PREFIXED_LITERAL),
        LintId::of(&needless_bool::BOOL_COMPARISON),
        LintId::of(&needless_bool::NEEDLESS_BOOL),
        LintId::of(&needless_borrowed_ref::NEEDLESS_BORROWED_REFERENCE),
        LintId::of(&needless_update::NEEDLESS_UPDATE),
        LintId::of(&neg_cmp_op_on_partial_ord::NEG_CMP_OP_ON_PARTIAL_ORD),
        LintId::of(&no_effect::NO_EFFECT),
        LintId::of(&no_effect::UNNECESSARY_OPERATION),
        LintId::of(&overflow_check_conditional::OVERFLOW_CHECK_CONDITIONAL),
        LintId::of(&partialeq_ne_impl::PARTIALEQ_NE_IMPL),
        LintId::of(&precedence::PRECEDENCE),
        LintId::of(&ptr_offset_with_cast::PTR_OFFSET_WITH_CAST),
        LintId::of(&ranges::RANGE_MINUS_ONE),
        LintId::of(&ranges::RANGE_PLUS_ONE),
        LintId::of(&ranges::RANGE_ZIP_WITH_LEN),
        LintId::of(&reference::DEREF_ADDROF),
        LintId::of(&reference::REF_IN_DEREF),
        LintId::of(&swap::MANUAL_SWAP),
        LintId::of(&temporary_assignment::TEMPORARY_ASSIGNMENT),
        LintId::of(&transmute::CROSSPOINTER_TRANSMUTE),
        LintId::of(&transmute::TRANSMUTE_BYTES_TO_STR),
        LintId::of(&transmute::TRANSMUTE_INT_TO_BOOL),
        LintId::of(&transmute::TRANSMUTE_INT_TO_CHAR),
        LintId::of(&transmute::TRANSMUTE_INT_TO_FLOAT),
        LintId::of(&transmute::TRANSMUTE_PTR_TO_PTR),
        LintId::of(&transmute::TRANSMUTE_PTR_TO_REF),
        LintId::of(&transmute::USELESS_TRANSMUTE),
        LintId::of(&types::BORROWED_BOX),
        LintId::of(&types::CHAR_LIT_AS_U8),
        LintId::of(&types::OPTION_OPTION),
        LintId::of(&types::TYPE_COMPLEXITY),
        LintId::of(&types::UNIT_ARG),
        LintId::of(&types::UNNECESSARY_CAST),
        LintId::of(&types::VEC_BOX),
        LintId::of(&unused_label::UNUSED_LABEL),
        LintId::of(&unwrap::UNNECESSARY_UNWRAP),
        LintId::of(&zero_div_zero::ZERO_DIVIDED_BY_ZERO),
    ]);

    store.register_group(true, "clippy::correctness", Some("clippy_correctness"), vec![
        LintId::of(&approx_const::APPROX_CONSTANT),
        LintId::of(&attrs::DEPRECATED_SEMVER),
        LintId::of(&attrs::USELESS_ATTRIBUTE),
        LintId::of(&bit_mask::BAD_BIT_MASK),
        LintId::of(&bit_mask::INEFFECTIVE_BIT_MASK),
        LintId::of(&booleans::LOGIC_BUG),
        LintId::of(&copies::IFS_SAME_COND),
        LintId::of(&copies::IF_SAME_THEN_ELSE),
        LintId::of(&derive::DERIVE_HASH_XOR_EQ),
        LintId::of(&drop_bounds::DROP_BOUNDS),
        LintId::of(&drop_forget_ref::DROP_COPY),
        LintId::of(&drop_forget_ref::DROP_REF),
        LintId::of(&drop_forget_ref::FORGET_COPY),
        LintId::of(&drop_forget_ref::FORGET_REF),
        LintId::of(&enum_clike::ENUM_CLIKE_UNPORTABLE_VARIANT),
        LintId::of(&eq_op::EQ_OP),
        LintId::of(&erasing_op::ERASING_OP),
        LintId::of(&formatting::POSSIBLE_MISSING_COMMA),
        LintId::of(&functions::NOT_UNSAFE_PTR_ARG_DEREF),
        LintId::of(&indexing_slicing::OUT_OF_BOUNDS_INDEXING),
        LintId::of(&infinite_iter::INFINITE_ITER),
        LintId::of(&inherent_to_string::INHERENT_TO_STRING_SHADOW_DISPLAY),
        LintId::of(&inline_fn_without_body::INLINE_FN_WITHOUT_BODY),
        LintId::of(&literal_representation::MISTYPED_LITERAL_SUFFIXES),
        LintId::of(&loops::FOR_LOOP_OVER_OPTION),
        LintId::of(&loops::FOR_LOOP_OVER_RESULT),
        LintId::of(&loops::ITER_NEXT_LOOP),
        LintId::of(&loops::NEVER_LOOP),
        LintId::of(&loops::REVERSE_RANGE_LOOP),
        LintId::of(&loops::WHILE_IMMUTABLE_CONDITION),
        LintId::of(&mem_discriminant::MEM_DISCRIMINANT_NON_ENUM),
        LintId::of(&mem_replace::MEM_REPLACE_WITH_UNINIT),
        LintId::of(&methods::CLONE_DOUBLE_REF),
        LintId::of(&methods::INTO_ITER_ON_ARRAY),
        LintId::of(&methods::TEMPORARY_CSTRING_AS_PTR),
        LintId::of(&methods::UNINIT_ASSUMED_INIT),
        LintId::of(&minmax::MIN_MAX),
        LintId::of(&misc::CMP_NAN),
        LintId::of(&misc::FLOAT_CMP),
        LintId::of(&misc::MODULO_ONE),
        LintId::of(&mutable_debug_assertion::DEBUG_ASSERT_WITH_MUT_CALL),
        LintId::of(&non_copy_const::BORROW_INTERIOR_MUTABLE_CONST),
        LintId::of(&non_copy_const::DECLARE_INTERIOR_MUTABLE_CONST),
        LintId::of(&open_options::NONSENSICAL_OPEN_OPTIONS),
        LintId::of(&ptr::MUT_FROM_REF),
        LintId::of(&ranges::ITERATOR_STEP_BY_ZERO),
        LintId::of(&regex::INVALID_REGEX),
        LintId::of(&serde_api::SERDE_API_MISUSE),
        LintId::of(&suspicious_trait_impl::SUSPICIOUS_ARITHMETIC_IMPL),
        LintId::of(&suspicious_trait_impl::SUSPICIOUS_OP_ASSIGN_IMPL),
        LintId::of(&swap::ALMOST_SWAPPED),
        LintId::of(&transmute::UNSOUND_COLLECTION_TRANSMUTE),
        LintId::of(&transmute::WRONG_TRANSMUTE),
        LintId::of(&transmuting_null::TRANSMUTING_NULL),
        LintId::of(&types::ABSURD_EXTREME_COMPARISONS),
        LintId::of(&types::CAST_PTR_ALIGNMENT),
        LintId::of(&types::CAST_REF_TO_MUT),
        LintId::of(&types::UNIT_CMP),
        LintId::of(&unicode::ZERO_WIDTH_SPACE),
        LintId::of(&unused_io_amount::UNUSED_IO_AMOUNT),
        LintId::of(&unwrap::PANICKING_UNWRAP),
    ]);

    store.register_group(true, "clippy::perf", Some("clippy_perf"), vec![
        LintId::of(&bytecount::NAIVE_BYTECOUNT),
        LintId::of(&entry::MAP_ENTRY),
        LintId::of(&escape::BOXED_LOCAL),
        LintId::of(&large_enum_variant::LARGE_ENUM_VARIANT),
        LintId::of(&loops::MANUAL_MEMCPY),
        LintId::of(&loops::NEEDLESS_COLLECT),
        LintId::of(&methods::EXPECT_FUN_CALL),
        LintId::of(&methods::INEFFICIENT_TO_STRING),
        LintId::of(&methods::ITER_NTH),
        LintId::of(&methods::OR_FUN_CALL),
        LintId::of(&methods::SINGLE_CHAR_PATTERN),
        LintId::of(&misc::CMP_OWNED),
        LintId::of(&mutex_atomic::MUTEX_ATOMIC),
        LintId::of(&redundant_clone::REDUNDANT_CLONE),
        LintId::of(&slow_vector_initialization::SLOW_VECTOR_INITIALIZATION),
        LintId::of(&trivially_copy_pass_by_ref::TRIVIALLY_COPY_PASS_BY_REF),
        LintId::of(&types::BOX_VEC),
        LintId::of(&vec::USELESS_VEC),
    ]);

    store.register_group(true, "clippy::cargo", Some("clippy_cargo"), vec![
        LintId::of(&cargo_common_metadata::CARGO_COMMON_METADATA),
        LintId::of(&multiple_crate_versions::MULTIPLE_CRATE_VERSIONS),
        LintId::of(&wildcard_dependencies::WILDCARD_DEPENDENCIES),
    ]);

    store.register_group(true, "clippy::nursery", Some("clippy_nursery"), vec![
        LintId::of(&attrs::EMPTY_LINE_AFTER_OUTER_ATTR),
        LintId::of(&fallible_impl_from::FALLIBLE_IMPL_FROM),
        LintId::of(&missing_const_for_fn::MISSING_CONST_FOR_FN),
        LintId::of(&mul_add::MANUAL_MUL_ADD),
        LintId::of(&mutex_atomic::MUTEX_INTEGER),
        LintId::of(&needless_borrow::NEEDLESS_BORROW),
        LintId::of(&path_buf_push_overwrite::PATH_BUF_PUSH_OVERWRITE),
    ]);
}

#[rustfmt::skip]
fn register_removed_non_tool_lints(store: &mut rustc::lint::LintStore) {
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
        "str_to_string",
        "using `str::to_string` is common even today and specialization will likely happen soon",
    );
    store.register_removed(
        "string_to_string",
        "using `string::to_string` is common even today and specialization will likely happen soon",
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
}

/// Register renamed lints.
///
/// Used in `./src/driver.rs`.
pub fn register_renamed(ls: &mut rustc::lint::LintStore) {
    ls.register_renamed("clippy::stutter", "clippy::module_name_repetitions");
    ls.register_renamed("clippy::new_without_default_derive", "clippy::new_without_default");
    ls.register_renamed("clippy::cyclomatic_complexity", "clippy::cognitive_complexity");
    ls.register_renamed("clippy::const_static_lifetime", "clippy::redundant_static_lifetimes");
}

// only exists to let the dogfood integration test works.
// Don't run clippy as an executable directly
#[allow(dead_code)]
fn main() {
    panic!("Please use the cargo-clippy executable");
}
