// error-pattern:cargo-clippy

#![feature(box_syntax)]
#![feature(alloc)]
#![feature(custom_attribute)]
#![feature(i128_type)]
#![feature(i128)]
#![feature(rustc_private)]
#![feature(slice_patterns)]
#![feature(stmt_expr_attributes)]
#![feature(conservative_impl_trait)]

#![allow(indexing_slicing, shadow_reuse, unknown_lints, missing_docs_in_private_items)]

extern crate syntax;
extern crate syntax_pos;
#[macro_use]
extern crate rustc;
extern crate rustc_data_structures;

extern crate toml;

// Only for the compile time checking of paths
extern crate core;
extern crate alloc;

// for unicode nfc normalization
extern crate unicode_normalization;

// for semver check in attrs.rs
extern crate semver;

// for regex checking
extern crate regex_syntax;

// for finding minimal boolean expressions
extern crate quine_mc_cluskey;

extern crate rustc_errors;
extern crate rustc_plugin;
extern crate rustc_const_eval;
extern crate rustc_const_math;

#[macro_use]
extern crate matches as matches_macro;

#[macro_use]
extern crate serde_derive;
extern crate serde;

#[macro_use]
extern crate lazy_static;

extern crate itertools;
extern crate pulldown_cmark;

macro_rules! declare_restriction_lint {
    { pub $name:tt, $description:tt } => {
        declare_lint! { pub $name, Allow, $description }
    };
}

pub mod consts;
#[macro_use]
pub mod utils;

// begin lints modules, do not remove this comment, it’s used in `update_lints`
pub mod approx_const;
pub mod arithmetic;
pub mod array_indexing;
pub mod assign_ops;
pub mod attrs;
pub mod bit_mask;
pub mod blacklisted_name;
pub mod block_in_if_condition;
pub mod booleans;
pub mod collapsible_if;
pub mod copies;
pub mod cyclomatic_complexity;
pub mod derive;
pub mod doc;
pub mod double_parens;
pub mod drop_forget_ref;
pub mod empty_enum;
pub mod entry;
pub mod enum_clike;
pub mod enum_glob_use;
pub mod enum_variants;
pub mod eq_op;
pub mod escape;
pub mod eta_reduction;
pub mod eval_order_dependence;
pub mod format;
pub mod formatting;
pub mod functions;
pub mod identity_op;
pub mod if_let_redundant_pattern_matching;
pub mod if_not_else;
pub mod items_after_statements;
pub mod large_enum_variant;
pub mod len_zero;
pub mod let_if_seq;
pub mod lifetimes;
pub mod literal_digit_grouping;
pub mod loops;
pub mod map_clone;
pub mod matches;
pub mod mem_forget;
pub mod methods;
pub mod minmax;
pub mod misc;
pub mod misc_early;
pub mod missing_doc;
pub mod mut_mut;
pub mod mut_reference;
pub mod mutex_atomic;
pub mod needless_bool;
pub mod needless_borrow;
pub mod needless_borrowed_ref;
pub mod needless_continue;
pub mod needless_pass_by_value;
pub mod needless_update;
pub mod neg_multiply;
pub mod new_without_default;
pub mod no_effect;
pub mod non_expressive_names;
pub mod ok_if_let;
pub mod open_options;
pub mod overflow_check_conditional;
pub mod panic;
pub mod partialeq_ne_impl;
pub mod precedence;
pub mod print;
pub mod ptr;
pub mod ranges;
pub mod reference;
pub mod regex;
pub mod returns;
pub mod serde_api;
pub mod shadow;
pub mod should_assert_eq;
pub mod strings;
pub mod swap;
pub mod temporary_assignment;
pub mod transmute;
pub mod types;
pub mod unicode;
pub mod unsafe_removed_from_name;
pub mod unused_io_amount;
pub mod unused_label;
pub mod vec;
pub mod zero_div_zero;
// end lints modules, do not remove this comment, it’s used in `update_lints`

mod reexport {
    pub use syntax::ast::{Name, NodeId};
}

#[cfg_attr(rustfmt, rustfmt_skip)]
pub fn register_plugins(reg: &mut rustc_plugin::Registry) {
    let conf = match utils::conf::file_from_args(reg.args()) {
        Ok(file_name) => {
            // if the user specified a file, it must exist, otherwise default to `clippy.toml` but
            // do not require the file to exist
            let file_name = if let Some(file_name) = file_name {
                Some(file_name)
            } else {
                match utils::conf::lookup_conf_file() {
                    Ok(path) => path,
                    Err(error) => {
                        reg.sess.struct_err(&format!("error reading Clippy's configuration file: {}", error)).emit();
                        None
                    }
                }
            };

            let (conf, errors) = utils::conf::read(file_name.as_ref().map(|p| p.as_ref()));

            // all conf errors are non-fatal, we just use the default conf in case of error
            for error in errors {
                reg.sess.struct_err(&format!("error reading Clippy's configuration file: {}", error)).emit();
            }

            conf
        }
        Err((err, span)) => {
            reg.sess.struct_span_err(span, err)
                    .span_note(span, "Clippy will use default configuration")
                    .emit();
            toml::from_str("").expect("we never error on empty config files")
        }
    };

    let mut store = reg.sess.lint_store.borrow_mut();
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
    // end deprecated lints, do not remove this comment, it’s used in `update_lints`

    reg.register_late_lint_pass(box serde_api::Serde);
    reg.register_early_lint_pass(box utils::internal_lints::Clippy);
    reg.register_late_lint_pass(box utils::internal_lints::LintWithoutLintPass::default());
    reg.register_late_lint_pass(box utils::inspector::Pass);
    reg.register_late_lint_pass(box utils::author::Pass);
    reg.register_late_lint_pass(box types::TypePass);
    reg.register_late_lint_pass(box booleans::NonminimalBool);
    reg.register_late_lint_pass(box eq_op::EqOp);
    reg.register_early_lint_pass(box enum_variants::EnumVariantNames::new(conf.enum_variant_name_threshold));
    reg.register_late_lint_pass(box enum_glob_use::EnumGlobUse);
    reg.register_late_lint_pass(box enum_clike::UnportableVariant);
    reg.register_late_lint_pass(box bit_mask::BitMask);
    reg.register_late_lint_pass(box ptr::PointerPass);
    reg.register_late_lint_pass(box needless_bool::NeedlessBool);
    reg.register_late_lint_pass(box needless_bool::BoolComparison);
    reg.register_late_lint_pass(box approx_const::Pass);
    reg.register_late_lint_pass(box misc::Pass);
    reg.register_early_lint_pass(box precedence::Precedence);
    reg.register_early_lint_pass(box needless_continue::NeedlessContinue);
    reg.register_late_lint_pass(box eta_reduction::EtaPass);
    reg.register_late_lint_pass(box identity_op::IdentityOp);
    reg.register_early_lint_pass(box items_after_statements::ItemsAfterStatements);
    reg.register_late_lint_pass(box mut_mut::MutMut);
    reg.register_late_lint_pass(box mut_reference::UnnecessaryMutPassed);
    reg.register_late_lint_pass(box len_zero::LenZero);
    reg.register_late_lint_pass(box attrs::AttrPass);
    reg.register_early_lint_pass(box collapsible_if::CollapsibleIf);
    reg.register_late_lint_pass(box block_in_if_condition::BlockInIfCondition);
    reg.register_late_lint_pass(box unicode::Unicode);
    reg.register_late_lint_pass(box strings::StringAdd);
    reg.register_early_lint_pass(box returns::ReturnPass);
    reg.register_late_lint_pass(box methods::Pass);
    reg.register_late_lint_pass(box shadow::Pass);
    reg.register_late_lint_pass(box types::LetPass);
    reg.register_late_lint_pass(box types::UnitCmp);
    reg.register_late_lint_pass(box loops::Pass::default());
    reg.register_late_lint_pass(box lifetimes::LifetimePass);
    reg.register_late_lint_pass(box entry::HashMapLint);
    reg.register_late_lint_pass(box ranges::StepByZero);
    reg.register_late_lint_pass(box types::CastPass);
    reg.register_late_lint_pass(box types::TypeComplexityPass::new(conf.type_complexity_threshold));
    reg.register_late_lint_pass(box matches::MatchPass);
    reg.register_late_lint_pass(box minmax::MinMaxPass);
    reg.register_late_lint_pass(box open_options::NonSensical);
    reg.register_late_lint_pass(box zero_div_zero::Pass);
    reg.register_late_lint_pass(box mutex_atomic::MutexAtomic);
    reg.register_late_lint_pass(box needless_update::Pass);
    reg.register_late_lint_pass(box needless_borrow::NeedlessBorrow);
    reg.register_late_lint_pass(box no_effect::Pass);
    reg.register_late_lint_pass(box map_clone::Pass);
    reg.register_late_lint_pass(box temporary_assignment::Pass);
    reg.register_late_lint_pass(box transmute::Transmute);
    reg.register_late_lint_pass(
        box cyclomatic_complexity::CyclomaticComplexity::new(conf.cyclomatic_complexity_threshold)
    );
    reg.register_late_lint_pass(box escape::Pass{too_large_for_stack: conf.too_large_for_stack});
    reg.register_early_lint_pass(box misc_early::MiscEarly);
    reg.register_late_lint_pass(box array_indexing::ArrayIndexing);
    reg.register_late_lint_pass(box panic::Pass);
    reg.register_late_lint_pass(box strings::StringLitAsBytes);
    reg.register_late_lint_pass(box derive::Derive);
    reg.register_late_lint_pass(box types::CharLitAsU8);
    reg.register_late_lint_pass(box print::Pass);
    reg.register_late_lint_pass(box vec::Pass);
    reg.register_early_lint_pass(box non_expressive_names::NonExpressiveNames {
        single_char_binding_names_threshold: conf.single_char_binding_names_threshold,
    });
    reg.register_late_lint_pass(box drop_forget_ref::Pass);
    reg.register_late_lint_pass(box empty_enum::EmptyEnum);
    reg.register_late_lint_pass(box types::AbsurdExtremeComparisons);
    reg.register_late_lint_pass(box types::InvalidUpcastComparisons);
    reg.register_late_lint_pass(box regex::Pass::default());
    reg.register_late_lint_pass(box copies::CopyAndPaste);
    reg.register_late_lint_pass(box format::Pass);
    reg.register_early_lint_pass(box formatting::Formatting);
    reg.register_late_lint_pass(box swap::Swap);
    reg.register_early_lint_pass(box if_not_else::IfNotElse);
    reg.register_late_lint_pass(box overflow_check_conditional::OverflowCheckConditional);
    reg.register_late_lint_pass(box unused_label::UnusedLabel);
    reg.register_late_lint_pass(box new_without_default::NewWithoutDefault);
    reg.register_late_lint_pass(box blacklisted_name::BlackListedName::new(conf.blacklisted_names));
    reg.register_late_lint_pass(box functions::Functions::new(conf.too_many_arguments_threshold));
    reg.register_early_lint_pass(box doc::Doc::new(conf.doc_valid_idents));
    reg.register_late_lint_pass(box neg_multiply::NegMultiply);
    reg.register_early_lint_pass(box unsafe_removed_from_name::UnsafeNameRemoval);
    reg.register_late_lint_pass(box mem_forget::MemForget);
    reg.register_late_lint_pass(box arithmetic::Arithmetic::default());
    reg.register_late_lint_pass(box assign_ops::AssignOps);
    reg.register_late_lint_pass(box let_if_seq::LetIfSeq);
    reg.register_late_lint_pass(box eval_order_dependence::EvalOrderDependence);
    reg.register_late_lint_pass(box missing_doc::MissingDoc::new());
    reg.register_late_lint_pass(box ok_if_let::Pass);
    reg.register_late_lint_pass(box if_let_redundant_pattern_matching::Pass);
    reg.register_late_lint_pass(box partialeq_ne_impl::Pass);
    reg.register_early_lint_pass(box reference::Pass);
    reg.register_early_lint_pass(box double_parens::DoubleParens);
    reg.register_late_lint_pass(box unused_io_amount::UnusedIoAmount);
    reg.register_late_lint_pass(box large_enum_variant::LargeEnumVariant::new(conf.enum_variant_size_threshold));
    reg.register_late_lint_pass(box should_assert_eq::ShouldAssertEq);
    reg.register_late_lint_pass(box needless_pass_by_value::NeedlessPassByValue);
    reg.register_early_lint_pass(box literal_digit_grouping::LiteralDigitGrouping);

    reg.register_lint_group("clippy_restrictions", vec![
        arithmetic::FLOAT_ARITHMETIC,
        arithmetic::INTEGER_ARITHMETIC,
        array_indexing::INDEXING_SLICING,
        assign_ops::ASSIGN_OPS,
    ]);

    reg.register_lint_group("clippy_pedantic", vec![
        booleans::NONMINIMAL_BOOL,
        empty_enum::EMPTY_ENUM,
        enum_glob_use::ENUM_GLOB_USE,
        enum_variants::PUB_ENUM_VARIANT_NAMES,
        enum_variants::STUTTER,
        if_not_else::IF_NOT_ELSE,
        items_after_statements::ITEMS_AFTER_STATEMENTS,
        matches::SINGLE_MATCH_ELSE,
        mem_forget::MEM_FORGET,
        methods::FILTER_MAP,
        methods::OPTION_MAP_UNWRAP_OR,
        methods::OPTION_MAP_UNWRAP_OR_ELSE,
        methods::OPTION_UNWRAP_USED,
        methods::RESULT_UNWRAP_USED,
        methods::WRONG_PUB_SELF_CONVENTION,
        misc::USED_UNDERSCORE_BINDING,
        misc_early::UNSEPARATED_LITERAL_SUFFIX,
        missing_doc::MISSING_DOCS_IN_PRIVATE_ITEMS,
        mut_mut::MUT_MUT,
        mutex_atomic::MUTEX_INTEGER,
        non_expressive_names::SIMILAR_NAMES,
        print::PRINT_STDOUT,
        print::USE_DEBUG,
        shadow::SHADOW_REUSE,
        shadow::SHADOW_SAME,
        shadow::SHADOW_UNRELATED,
        strings::STRING_ADD,
        strings::STRING_ADD_ASSIGN,
        types::CAST_POSSIBLE_TRUNCATION,
        types::CAST_POSSIBLE_WRAP,
        types::CAST_PRECISION_LOSS,
        types::CAST_SIGN_LOSS,
        types::INVALID_UPCAST_COMPARISONS,
        unicode::NON_ASCII_LITERAL,
        unicode::UNICODE_NOT_NFC,
    ]);

    reg.register_lint_group("clippy_internal", vec![
        utils::internal_lints::CLIPPY_LINTS_INTERNAL,
        utils::internal_lints::LINT_WITHOUT_LINT_PASS,
    ]);

    reg.register_lint_group("clippy", vec![
        approx_const::APPROX_CONSTANT,
        array_indexing::OUT_OF_BOUNDS_INDEXING,
        assign_ops::ASSIGN_OP_PATTERN,
        assign_ops::MISREFACTORED_ASSIGN_OP,
        attrs::DEPRECATED_SEMVER,
        attrs::INLINE_ALWAYS,
        attrs::USELESS_ATTRIBUTE,
        bit_mask::BAD_BIT_MASK,
        bit_mask::INEFFECTIVE_BIT_MASK,
        blacklisted_name::BLACKLISTED_NAME,
        block_in_if_condition::BLOCK_IN_IF_CONDITION_EXPR,
        block_in_if_condition::BLOCK_IN_IF_CONDITION_STMT,
        booleans::LOGIC_BUG,
        collapsible_if::COLLAPSIBLE_IF,
        copies::IF_SAME_THEN_ELSE,
        copies::IFS_SAME_COND,
        copies::MATCH_SAME_ARMS,
        cyclomatic_complexity::CYCLOMATIC_COMPLEXITY,
        derive::DERIVE_HASH_XOR_EQ,
        derive::EXPL_IMPL_CLONE_ON_COPY,
        doc::DOC_MARKDOWN,
        double_parens::DOUBLE_PARENS,
        drop_forget_ref::DROP_COPY,
        drop_forget_ref::DROP_REF,
        drop_forget_ref::FORGET_COPY,
        drop_forget_ref::FORGET_REF,
        entry::MAP_ENTRY,
        enum_clike::ENUM_CLIKE_UNPORTABLE_VARIANT,
        enum_variants::ENUM_VARIANT_NAMES,
        enum_variants::MODULE_INCEPTION,
        eq_op::EQ_OP,
        eq_op::OP_REF,
        escape::BOXED_LOCAL,
        eta_reduction::REDUNDANT_CLOSURE,
        eval_order_dependence::DIVERGING_SUB_EXPRESSION,
        eval_order_dependence::EVAL_ORDER_DEPENDENCE,
        format::USELESS_FORMAT,
        formatting::POSSIBLE_MISSING_COMMA,
        formatting::SUSPICIOUS_ASSIGNMENT_FORMATTING,
        formatting::SUSPICIOUS_ELSE_FORMATTING,
        functions::NOT_UNSAFE_PTR_ARG_DEREF,
        functions::TOO_MANY_ARGUMENTS,
        identity_op::IDENTITY_OP,
        if_let_redundant_pattern_matching::IF_LET_REDUNDANT_PATTERN_MATCHING,
        large_enum_variant::LARGE_ENUM_VARIANT,
        len_zero::LEN_WITHOUT_IS_EMPTY,
        len_zero::LEN_ZERO,
        let_if_seq::USELESS_LET_IF_SEQ,
        lifetimes::NEEDLESS_LIFETIMES,
        lifetimes::UNUSED_LIFETIMES,
        literal_digit_grouping::INCONSISTENT_DIGIT_GROUPING,
        literal_digit_grouping::LARGE_DIGIT_GROUPS,
        literal_digit_grouping::UNREADABLE_LITERAL,
        loops::EMPTY_LOOP,
        loops::EXPLICIT_COUNTER_LOOP,
        loops::EXPLICIT_INTO_ITER_LOOP,
        loops::EXPLICIT_ITER_LOOP,
        loops::FOR_KV_MAP,
        loops::FOR_LOOP_OVER_OPTION,
        loops::FOR_LOOP_OVER_RESULT,
        loops::ITER_NEXT_LOOP,
        loops::NEEDLESS_RANGE_LOOP,
        loops::NEVER_LOOP,
        loops::REVERSE_RANGE_LOOP,
        loops::UNUSED_COLLECT,
        loops::WHILE_LET_LOOP,
        loops::WHILE_LET_ON_ITERATOR,
        map_clone::MAP_CLONE,
        matches::MATCH_BOOL,
        matches::MATCH_OVERLAPPING_ARM,
        matches::MATCH_REF_PATS,
        matches::MATCH_WILD_ERR_ARM,
        matches::SINGLE_MATCH,
        methods::CHARS_NEXT_CMP,
        methods::CLONE_DOUBLE_REF,
        methods::CLONE_ON_COPY,
        methods::FILTER_NEXT,
        methods::GET_UNWRAP,
        methods::ITER_CLONED_COLLECT,
        methods::ITER_NTH,
        methods::ITER_SKIP_NEXT,
        methods::NEW_RET_NO_SELF,
        methods::OK_EXPECT,
        methods::OR_FUN_CALL,
        methods::SEARCH_IS_SOME,
        methods::SHOULD_IMPLEMENT_TRAIT,
        methods::SINGLE_CHAR_PATTERN,
        methods::STRING_EXTEND_CHARS,
        methods::TEMPORARY_CSTRING_AS_PTR,
        methods::WRONG_SELF_CONVENTION,
        minmax::MIN_MAX,
        misc::CMP_NAN,
        misc::CMP_OWNED,
        misc::FLOAT_CMP,
        misc::MODULO_ONE,
        misc::REDUNDANT_PATTERN,
        misc::SHORT_CIRCUIT_STATEMENT,
        misc::TOPLEVEL_REF_ARG,
        misc::ZERO_PTR,
        misc_early::BUILTIN_TYPE_SHADOW,
        misc_early::DOUBLE_NEG,
        misc_early::DUPLICATE_UNDERSCORE_ARGUMENT,
        misc_early::MIXED_CASE_HEX_LITERALS,
        misc_early::REDUNDANT_CLOSURE_CALL,
        misc_early::UNNEEDED_FIELD_PATTERN,
        misc_early::ZERO_PREFIXED_LITERAL,
        mut_reference::UNNECESSARY_MUT_PASSED,
        mutex_atomic::MUTEX_ATOMIC,
        needless_bool::BOOL_COMPARISON,
        needless_bool::NEEDLESS_BOOL,
        needless_borrow::NEEDLESS_BORROW,
        needless_borrowed_ref::NEEDLESS_BORROWED_REFERENCE,
        needless_continue::NEEDLESS_CONTINUE,
        needless_pass_by_value::NEEDLESS_PASS_BY_VALUE,
        needless_update::NEEDLESS_UPDATE,
        neg_multiply::NEG_MULTIPLY,
        new_without_default::NEW_WITHOUT_DEFAULT,
        new_without_default::NEW_WITHOUT_DEFAULT_DERIVE,
        no_effect::NO_EFFECT,
        no_effect::UNNECESSARY_OPERATION,
        non_expressive_names::MANY_SINGLE_CHAR_NAMES,
        ok_if_let::IF_LET_SOME_RESULT,
        open_options::NONSENSICAL_OPEN_OPTIONS,
        overflow_check_conditional::OVERFLOW_CHECK_CONDITIONAL,
        panic::PANIC_PARAMS,
        partialeq_ne_impl::PARTIALEQ_NE_IMPL,
        precedence::PRECEDENCE,
        print::PRINT_WITH_NEWLINE,
        ptr::CMP_NULL,
        ptr::MUT_FROM_REF,
        ptr::PTR_ARG,
        ranges::ITERATOR_STEP_BY_ZERO,
        ranges::RANGE_ZIP_WITH_LEN,
        reference::DEREF_ADDROF,
        regex::INVALID_REGEX,
        regex::REGEX_MACRO,
        regex::TRIVIAL_REGEX,
        returns::LET_AND_RETURN,
        returns::NEEDLESS_RETURN,
        serde_api::SERDE_API_MISUSE,
        should_assert_eq::SHOULD_ASSERT_EQ,
        strings::STRING_LIT_AS_BYTES,
        swap::ALMOST_SWAPPED,
        swap::MANUAL_SWAP,
        temporary_assignment::TEMPORARY_ASSIGNMENT,
        transmute::CROSSPOINTER_TRANSMUTE,
        transmute::TRANSMUTE_PTR_TO_REF,
        transmute::USELESS_TRANSMUTE,
        transmute::WRONG_TRANSMUTE,
        types::ABSURD_EXTREME_COMPARISONS,
        types::BORROWED_BOX,
        types::BOX_VEC,
        types::CHAR_LIT_AS_U8,
        types::LET_UNIT_VALUE,
        types::LINKEDLIST,
        types::TYPE_COMPLEXITY,
        types::UNIT_CMP,
        types::UNNECESSARY_CAST,
        unicode::ZERO_WIDTH_SPACE,
        unsafe_removed_from_name::UNSAFE_REMOVED_FROM_NAME,
        unused_io_amount::UNUSED_IO_AMOUNT,
        unused_label::UNUSED_LABEL,
        vec::USELESS_VEC,
        zero_div_zero::ZERO_DIVIDED_BY_ZERO,
    ]);
}

// only exists to let the dogfood integration test works.
// Don't run clippy as an executable directly
#[allow(dead_code, print_stdout)]
fn main() {
    panic!("Please use the cargo-clippy executable");
}
