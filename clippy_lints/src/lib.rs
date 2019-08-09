// error-pattern:cargo-clippy

#![feature(box_syntax)]
#![feature(box_patterns)]
#![feature(never_type)]
#![feature(rustc_private)]
#![feature(slice_patterns)]
#![feature(stmt_expr_attributes)]
#![allow(clippy::missing_docs_in_private_items)]
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
extern crate rustc_errors;
#[allow(unused_extern_crates)]
extern crate rustc_mir;
#[allow(unused_extern_crates)]
extern crate rustc_plugin;
#[allow(unused_extern_crates)]
extern crate rustc_target;
#[allow(unused_extern_crates)]
extern crate rustc_typeck;
#[allow(unused_extern_crates)]
extern crate syntax;
#[allow(unused_extern_crates)]
extern crate syntax_pos;

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
pub mod multiple_crate_versions;
pub mod mut_mut;
pub mod mut_reference;
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
pub fn register_pre_expansion_lints(
    session: &rustc::session::Session,
    store: &mut rustc::lint::LintStore,
    conf: &Conf,
) {
    store.register_pre_expansion_pass(Some(session), true, false, box write::Write);
    store.register_pre_expansion_pass(
        Some(session),
        true,
        false,
        box redundant_field_names::RedundantFieldNames,
    );
    store.register_pre_expansion_pass(
        Some(session),
        true,
        false,
        box non_expressive_names::NonExpressiveNames {
            single_char_binding_names_threshold: conf.single_char_binding_names_threshold,
        },
    );
    store.register_pre_expansion_pass(Some(session), true, false, box attrs::DeprecatedCfgAttribute);
    store.register_pre_expansion_pass(Some(session), true, false, box dbg_macro::DbgMacro);
}

#[doc(hidden)]
pub fn read_conf(reg: &rustc_plugin::Registry<'_>) -> Conf {
    match utils::conf::file_from_args(reg.args()) {
        Ok(file_name) => {
            // if the user specified a file, it must exist, otherwise default to `clippy.toml` but
            // do not require the file to exist
            let file_name = if let Some(file_name) = file_name {
                Some(file_name)
            } else {
                match utils::conf::lookup_conf_file() {
                    Ok(path) => path,
                    Err(error) => {
                        reg.sess
                            .struct_err(&format!("error finding Clippy's configuration file: {}", error))
                            .emit();
                        None
                    },
                }
            };

            let file_name = file_name.map(|file_name| {
                if file_name.is_relative() {
                    reg.sess
                        .local_crate_source_file
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
                reg.sess
                    .struct_err(&format!(
                        "error reading Clippy's configuration file `{}`: {}",
                        file_name.as_ref().and_then(|p| p.to_str()).unwrap_or(""),
                        error
                    ))
                    .emit();
            }

            conf
        },
        Err((err, span)) => {
            reg.sess
                .struct_span_err(span, err)
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
pub fn register_plugins(reg: &mut rustc_plugin::Registry<'_>, conf: &Conf) {
    let mut store = reg.sess.lint_store.borrow_mut();
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
    // end deprecated lints, do not remove this comment, it’s used in `update_lints`

    reg.register_late_lint_pass(box serde_api::SerdeAPI);
    reg.register_early_lint_pass(box utils::internal_lints::ClippyLintsInternal);
    reg.register_late_lint_pass(box utils::internal_lints::CompilerLintFunctions::new());
    reg.register_late_lint_pass(box utils::internal_lints::LintWithoutLintPass::default());
    reg.register_late_lint_pass(box utils::internal_lints::OuterExpnInfoPass);
    reg.register_late_lint_pass(box utils::inspector::DeepCodeInspector);
    reg.register_late_lint_pass(box utils::author::Author);
    reg.register_late_lint_pass(box types::Types);
    reg.register_late_lint_pass(box booleans::NonminimalBool);
    reg.register_late_lint_pass(box eq_op::EqOp);
    reg.register_early_lint_pass(box enum_variants::EnumVariantNames::new(conf.enum_variant_name_threshold));
    reg.register_late_lint_pass(box enum_glob_use::EnumGlobUse);
    reg.register_late_lint_pass(box enum_clike::UnportableVariant);
    reg.register_late_lint_pass(box excessive_precision::ExcessivePrecision);
    reg.register_late_lint_pass(box bit_mask::BitMask::new(conf.verbose_bit_mask_threshold));
    reg.register_late_lint_pass(box ptr::Ptr);
    reg.register_late_lint_pass(box needless_bool::NeedlessBool);
    reg.register_late_lint_pass(box needless_bool::BoolComparison);
    reg.register_late_lint_pass(box approx_const::ApproxConstant);
    reg.register_late_lint_pass(box misc::MiscLints);
    reg.register_early_lint_pass(box precedence::Precedence);
    reg.register_early_lint_pass(box needless_continue::NeedlessContinue);
    reg.register_late_lint_pass(box eta_reduction::EtaReduction);
    reg.register_late_lint_pass(box identity_op::IdentityOp);
    reg.register_late_lint_pass(box erasing_op::ErasingOp);
    reg.register_early_lint_pass(box items_after_statements::ItemsAfterStatements);
    reg.register_late_lint_pass(box mut_mut::MutMut);
    reg.register_late_lint_pass(box mut_reference::UnnecessaryMutPassed);
    reg.register_late_lint_pass(box len_zero::LenZero);
    reg.register_late_lint_pass(box attrs::Attributes);
    reg.register_early_lint_pass(box collapsible_if::CollapsibleIf);
    reg.register_late_lint_pass(box block_in_if_condition::BlockInIfCondition);
    reg.register_late_lint_pass(box unicode::Unicode);
    reg.register_late_lint_pass(box strings::StringAdd);
    reg.register_early_lint_pass(box returns::Return);
    reg.register_late_lint_pass(box implicit_return::ImplicitReturn);
    reg.register_late_lint_pass(box methods::Methods);
    reg.register_late_lint_pass(box map_clone::MapClone);
    reg.register_late_lint_pass(box shadow::Shadow);
    reg.register_late_lint_pass(box types::LetUnitValue);
    reg.register_late_lint_pass(box types::UnitCmp);
    reg.register_late_lint_pass(box loops::Loops);
    reg.register_late_lint_pass(box main_recursion::MainRecursion::default());
    reg.register_late_lint_pass(box lifetimes::Lifetimes);
    reg.register_late_lint_pass(box entry::HashMapPass);
    reg.register_late_lint_pass(box ranges::Ranges);
    reg.register_late_lint_pass(box types::Casts);
    reg.register_late_lint_pass(box types::TypeComplexity::new(conf.type_complexity_threshold));
    reg.register_late_lint_pass(box matches::Matches);
    reg.register_late_lint_pass(box minmax::MinMaxPass);
    reg.register_late_lint_pass(box open_options::OpenOptions);
    reg.register_late_lint_pass(box zero_div_zero::ZeroDiv);
    reg.register_late_lint_pass(box mutex_atomic::Mutex);
    reg.register_late_lint_pass(box needless_update::NeedlessUpdate);
    reg.register_late_lint_pass(box needless_borrow::NeedlessBorrow::default());
    reg.register_late_lint_pass(box needless_borrowed_ref::NeedlessBorrowedRef);
    reg.register_late_lint_pass(box no_effect::NoEffect);
    reg.register_late_lint_pass(box temporary_assignment::TemporaryAssignment);
    reg.register_late_lint_pass(box transmute::Transmute);
    reg.register_late_lint_pass(
        box cognitive_complexity::CognitiveComplexity::new(conf.cognitive_complexity_threshold)
    );
    reg.register_late_lint_pass(box escape::BoxedLocal{too_large_for_stack: conf.too_large_for_stack});
    reg.register_early_lint_pass(box misc_early::MiscEarlyLints);
    reg.register_late_lint_pass(box panic_unimplemented::PanicUnimplemented);
    reg.register_late_lint_pass(box strings::StringLitAsBytes);
    reg.register_late_lint_pass(box derive::Derive);
    reg.register_late_lint_pass(box types::CharLitAsU8);
    reg.register_late_lint_pass(box vec::UselessVec);
    reg.register_late_lint_pass(box drop_bounds::DropBounds);
    reg.register_late_lint_pass(box get_last_with_len::GetLastWithLen);
    reg.register_late_lint_pass(box drop_forget_ref::DropForgetRef);
    reg.register_late_lint_pass(box empty_enum::EmptyEnum);
    reg.register_late_lint_pass(box types::AbsurdExtremeComparisons);
    reg.register_late_lint_pass(box types::InvalidUpcastComparisons);
    reg.register_late_lint_pass(box regex::Regex::default());
    reg.register_late_lint_pass(box copies::CopyAndPaste);
    reg.register_late_lint_pass(box copy_iterator::CopyIterator);
    reg.register_late_lint_pass(box format::UselessFormat);
    reg.register_early_lint_pass(box formatting::Formatting);
    reg.register_late_lint_pass(box swap::Swap);
    reg.register_early_lint_pass(box if_not_else::IfNotElse);
    reg.register_early_lint_pass(box else_if_without_else::ElseIfWithoutElse);
    reg.register_early_lint_pass(box int_plus_one::IntPlusOne);
    reg.register_late_lint_pass(box overflow_check_conditional::OverflowCheckConditional);
    reg.register_late_lint_pass(box unused_label::UnusedLabel);
    reg.register_late_lint_pass(box new_without_default::NewWithoutDefault::default());
    reg.register_late_lint_pass(box blacklisted_name::BlacklistedName::new(
            conf.blacklisted_names.iter().cloned().collect()
    ));
    reg.register_late_lint_pass(box functions::Functions::new(conf.too_many_arguments_threshold, conf.too_many_lines_threshold));
    reg.register_early_lint_pass(box doc::DocMarkdown::new(conf.doc_valid_idents.iter().cloned().collect()));
    reg.register_late_lint_pass(box neg_multiply::NegMultiply);
    reg.register_early_lint_pass(box unsafe_removed_from_name::UnsafeNameRemoval);
    reg.register_late_lint_pass(box mem_discriminant::MemDiscriminant);
    reg.register_late_lint_pass(box mem_forget::MemForget);
    reg.register_late_lint_pass(box mem_replace::MemReplace);
    reg.register_late_lint_pass(box arithmetic::Arithmetic::default());
    reg.register_late_lint_pass(box assign_ops::AssignOps);
    reg.register_late_lint_pass(box let_if_seq::LetIfSeq);
    reg.register_late_lint_pass(box eval_order_dependence::EvalOrderDependence);
    reg.register_late_lint_pass(box missing_doc::MissingDoc::new());
    reg.register_late_lint_pass(box missing_inline::MissingInline);
    reg.register_late_lint_pass(box ok_if_let::OkIfLet);
    reg.register_late_lint_pass(box redundant_pattern_matching::RedundantPatternMatching);
    reg.register_late_lint_pass(box partialeq_ne_impl::PartialEqNeImpl);
    reg.register_early_lint_pass(box reference::DerefAddrOf);
    reg.register_early_lint_pass(box reference::RefInDeref);
    reg.register_early_lint_pass(box double_parens::DoubleParens);
    reg.register_late_lint_pass(box unused_io_amount::UnusedIoAmount);
    reg.register_late_lint_pass(box large_enum_variant::LargeEnumVariant::new(conf.enum_variant_size_threshold));
    reg.register_late_lint_pass(box explicit_write::ExplicitWrite);
    reg.register_late_lint_pass(box needless_pass_by_value::NeedlessPassByValue);
    reg.register_late_lint_pass(box trivially_copy_pass_by_ref::TriviallyCopyPassByRef::new(
            conf.trivial_copy_size_limit,
            &reg.sess.target,
    ));
    reg.register_early_lint_pass(box literal_representation::LiteralDigitGrouping);
    reg.register_early_lint_pass(box literal_representation::DecimalLiteralRepresentation::new(
            conf.literal_representation_threshold
    ));
    reg.register_late_lint_pass(box try_err::TryErr);
    reg.register_late_lint_pass(box use_self::UseSelf);
    reg.register_late_lint_pass(box bytecount::ByteCount);
    reg.register_late_lint_pass(box infinite_iter::InfiniteIter);
    reg.register_late_lint_pass(box inline_fn_without_body::InlineFnWithoutBody);
    reg.register_late_lint_pass(box identity_conversion::IdentityConversion::default());
    reg.register_late_lint_pass(box types::ImplicitHasher);
    reg.register_early_lint_pass(box redundant_static_lifetimes::RedundantStaticLifetimes);
    reg.register_late_lint_pass(box fallible_impl_from::FallibleImplFrom);
    reg.register_late_lint_pass(box replace_consts::ReplaceConsts);
    reg.register_late_lint_pass(box types::UnitArg);
    reg.register_late_lint_pass(box double_comparison::DoubleComparisons);
    reg.register_late_lint_pass(box question_mark::QuestionMark);
    reg.register_late_lint_pass(box suspicious_trait_impl::SuspiciousImpl);
    reg.register_early_lint_pass(box cargo_common_metadata::CargoCommonMetadata);
    reg.register_early_lint_pass(box multiple_crate_versions::MultipleCrateVersions);
    reg.register_early_lint_pass(box wildcard_dependencies::WildcardDependencies);
    reg.register_late_lint_pass(box map_unit_fn::MapUnit);
    reg.register_late_lint_pass(box infallible_destructuring_match::InfallibleDestructingMatch);
    reg.register_late_lint_pass(box inherent_impl::MultipleInherentImpl::default());
    reg.register_late_lint_pass(box neg_cmp_op_on_partial_ord::NoNegCompOpForPartialOrd);
    reg.register_late_lint_pass(box unwrap::Unwrap);
    reg.register_late_lint_pass(box duration_subsec::DurationSubsec);
    reg.register_late_lint_pass(box default_trait_access::DefaultTraitAccess);
    reg.register_late_lint_pass(box indexing_slicing::IndexingSlicing);
    reg.register_late_lint_pass(box non_copy_const::NonCopyConst);
    reg.register_late_lint_pass(box ptr_offset_with_cast::PtrOffsetWithCast);
    reg.register_late_lint_pass(box redundant_clone::RedundantClone);
    reg.register_late_lint_pass(box slow_vector_initialization::SlowVectorInit);
    reg.register_late_lint_pass(box types::RefToMut);
    reg.register_late_lint_pass(box assertions_on_constants::AssertionsOnConstants);
    reg.register_late_lint_pass(box missing_const_for_fn::MissingConstForFn);
    reg.register_late_lint_pass(box transmuting_null::TransmutingNull);
    reg.register_late_lint_pass(box path_buf_push_overwrite::PathBufPushOverwrite);
    reg.register_late_lint_pass(box checked_conversions::CheckedConversions);
    reg.register_late_lint_pass(box integer_division::IntegerDivision);
    reg.register_late_lint_pass(box inherent_to_string::InherentToString);
    reg.register_late_lint_pass(box trait_bounds::TraitBounds);

    reg.register_lint_group("clippy::restriction", Some("clippy_restriction"), vec![
        arithmetic::FLOAT_ARITHMETIC,
        arithmetic::INTEGER_ARITHMETIC,
        dbg_macro::DBG_MACRO,
        else_if_without_else::ELSE_IF_WITHOUT_ELSE,
        implicit_return::IMPLICIT_RETURN,
        indexing_slicing::INDEXING_SLICING,
        inherent_impl::MULTIPLE_INHERENT_IMPL,
        integer_division::INTEGER_DIVISION,
        literal_representation::DECIMAL_LITERAL_REPRESENTATION,
        matches::WILDCARD_ENUM_MATCH_ARM,
        mem_forget::MEM_FORGET,
        methods::CLONE_ON_REF_PTR,
        methods::GET_UNWRAP,
        methods::OPTION_UNWRAP_USED,
        methods::RESULT_UNWRAP_USED,
        methods::WRONG_PUB_SELF_CONVENTION,
        misc::FLOAT_CMP_CONST,
        missing_doc::MISSING_DOCS_IN_PRIVATE_ITEMS,
        missing_inline::MISSING_INLINE_IN_PUBLIC_ITEMS,
        panic_unimplemented::UNIMPLEMENTED,
        shadow::SHADOW_REUSE,
        shadow::SHADOW_SAME,
        strings::STRING_ADD,
        write::PRINT_STDOUT,
        write::USE_DEBUG,
    ]);

    reg.register_lint_group("clippy::pedantic", Some("clippy_pedantic"), vec![
        attrs::INLINE_ALWAYS,
        checked_conversions::CHECKED_CONVERSIONS,
        copies::MATCH_SAME_ARMS,
        copy_iterator::COPY_ITERATOR,
        default_trait_access::DEFAULT_TRAIT_ACCESS,
        derive::EXPL_IMPL_CLONE_ON_COPY,
        doc::DOC_MARKDOWN,
        empty_enum::EMPTY_ENUM,
        enum_glob_use::ENUM_GLOB_USE,
        enum_variants::MODULE_NAME_REPETITIONS,
        enum_variants::PUB_ENUM_VARIANT_NAMES,
        eta_reduction::REDUNDANT_CLOSURE_FOR_METHOD_CALLS,
        functions::TOO_MANY_LINES,
        if_not_else::IF_NOT_ELSE,
        infinite_iter::MAYBE_INFINITE_ITER,
        items_after_statements::ITEMS_AFTER_STATEMENTS,
        literal_representation::LARGE_DIGIT_GROUPS,
        loops::EXPLICIT_INTO_ITER_LOOP,
        loops::EXPLICIT_ITER_LOOP,
        matches::SINGLE_MATCH_ELSE,
        methods::FILTER_MAP,
        methods::FILTER_MAP_NEXT,
        methods::FIND_MAP,
        methods::MAP_FLATTEN,
        methods::OPTION_MAP_UNWRAP_OR,
        methods::OPTION_MAP_UNWRAP_OR_ELSE,
        methods::RESULT_MAP_UNWRAP_OR_ELSE,
        misc::USED_UNDERSCORE_BINDING,
        misc_early::UNSEPARATED_LITERAL_SUFFIX,
        mut_mut::MUT_MUT,
        needless_continue::NEEDLESS_CONTINUE,
        needless_pass_by_value::NEEDLESS_PASS_BY_VALUE,
        non_expressive_names::SIMILAR_NAMES,
        replace_consts::REPLACE_CONSTS,
        shadow::SHADOW_UNRELATED,
        strings::STRING_ADD_ASSIGN,
        types::CAST_POSSIBLE_TRUNCATION,
        types::CAST_POSSIBLE_WRAP,
        types::CAST_PRECISION_LOSS,
        types::CAST_SIGN_LOSS,
        types::INVALID_UPCAST_COMPARISONS,
        types::LINKEDLIST,
        unicode::NON_ASCII_LITERAL,
        unicode::UNICODE_NOT_NFC,
        use_self::USE_SELF,
    ]);

    reg.register_lint_group("clippy::internal", Some("clippy_internal"), vec![
        utils::internal_lints::CLIPPY_LINTS_INTERNAL,
        utils::internal_lints::COMPILER_LINT_FUNCTIONS,
        utils::internal_lints::LINT_WITHOUT_LINT_PASS,
        utils::internal_lints::OUTER_EXPN_EXPN_INFO,
    ]);

    reg.register_lint_group("clippy::all", Some("clippy"), vec![
        approx_const::APPROX_CONSTANT,
        assertions_on_constants::ASSERTIONS_ON_CONSTANTS,
        assign_ops::ASSIGN_OP_PATTERN,
        assign_ops::MISREFACTORED_ASSIGN_OP,
        attrs::DEPRECATED_CFG_ATTR,
        attrs::DEPRECATED_SEMVER,
        attrs::UNKNOWN_CLIPPY_LINTS,
        attrs::USELESS_ATTRIBUTE,
        bit_mask::BAD_BIT_MASK,
        bit_mask::INEFFECTIVE_BIT_MASK,
        bit_mask::VERBOSE_BIT_MASK,
        blacklisted_name::BLACKLISTED_NAME,
        block_in_if_condition::BLOCK_IN_IF_CONDITION_EXPR,
        block_in_if_condition::BLOCK_IN_IF_CONDITION_STMT,
        booleans::LOGIC_BUG,
        booleans::NONMINIMAL_BOOL,
        bytecount::NAIVE_BYTECOUNT,
        cognitive_complexity::COGNITIVE_COMPLEXITY,
        collapsible_if::COLLAPSIBLE_IF,
        copies::IFS_SAME_COND,
        copies::IF_SAME_THEN_ELSE,
        derive::DERIVE_HASH_XOR_EQ,
        double_comparison::DOUBLE_COMPARISONS,
        double_parens::DOUBLE_PARENS,
        drop_bounds::DROP_BOUNDS,
        drop_forget_ref::DROP_COPY,
        drop_forget_ref::DROP_REF,
        drop_forget_ref::FORGET_COPY,
        drop_forget_ref::FORGET_REF,
        duration_subsec::DURATION_SUBSEC,
        entry::MAP_ENTRY,
        enum_clike::ENUM_CLIKE_UNPORTABLE_VARIANT,
        enum_variants::ENUM_VARIANT_NAMES,
        enum_variants::MODULE_INCEPTION,
        eq_op::EQ_OP,
        eq_op::OP_REF,
        erasing_op::ERASING_OP,
        escape::BOXED_LOCAL,
        eta_reduction::REDUNDANT_CLOSURE,
        eval_order_dependence::DIVERGING_SUB_EXPRESSION,
        eval_order_dependence::EVAL_ORDER_DEPENDENCE,
        excessive_precision::EXCESSIVE_PRECISION,
        explicit_write::EXPLICIT_WRITE,
        format::USELESS_FORMAT,
        formatting::POSSIBLE_MISSING_COMMA,
        formatting::SUSPICIOUS_ASSIGNMENT_FORMATTING,
        formatting::SUSPICIOUS_ELSE_FORMATTING,
        functions::NOT_UNSAFE_PTR_ARG_DEREF,
        functions::TOO_MANY_ARGUMENTS,
        get_last_with_len::GET_LAST_WITH_LEN,
        identity_conversion::IDENTITY_CONVERSION,
        identity_op::IDENTITY_OP,
        indexing_slicing::OUT_OF_BOUNDS_INDEXING,
        infallible_destructuring_match::INFALLIBLE_DESTRUCTURING_MATCH,
        infinite_iter::INFINITE_ITER,
        inherent_to_string::INHERENT_TO_STRING,
        inherent_to_string::INHERENT_TO_STRING_SHADOW_DISPLAY,
        inline_fn_without_body::INLINE_FN_WITHOUT_BODY,
        int_plus_one::INT_PLUS_ONE,
        large_enum_variant::LARGE_ENUM_VARIANT,
        len_zero::LEN_WITHOUT_IS_EMPTY,
        len_zero::LEN_ZERO,
        let_if_seq::USELESS_LET_IF_SEQ,
        lifetimes::EXTRA_UNUSED_LIFETIMES,
        lifetimes::NEEDLESS_LIFETIMES,
        literal_representation::INCONSISTENT_DIGIT_GROUPING,
        literal_representation::MISTYPED_LITERAL_SUFFIXES,
        literal_representation::UNREADABLE_LITERAL,
        loops::EMPTY_LOOP,
        loops::EXPLICIT_COUNTER_LOOP,
        loops::FOR_KV_MAP,
        loops::FOR_LOOP_OVER_OPTION,
        loops::FOR_LOOP_OVER_RESULT,
        loops::ITER_NEXT_LOOP,
        loops::MANUAL_MEMCPY,
        loops::MUT_RANGE_BOUND,
        loops::NEEDLESS_COLLECT,
        loops::NEEDLESS_RANGE_LOOP,
        loops::NEVER_LOOP,
        loops::REVERSE_RANGE_LOOP,
        loops::UNUSED_COLLECT,
        loops::WHILE_IMMUTABLE_CONDITION,
        loops::WHILE_LET_LOOP,
        loops::WHILE_LET_ON_ITERATOR,
        main_recursion::MAIN_RECURSION,
        map_clone::MAP_CLONE,
        map_unit_fn::OPTION_MAP_UNIT_FN,
        map_unit_fn::RESULT_MAP_UNIT_FN,
        matches::MATCH_AS_REF,
        matches::MATCH_BOOL,
        matches::MATCH_OVERLAPPING_ARM,
        matches::MATCH_REF_PATS,
        matches::MATCH_WILD_ERR_ARM,
        matches::SINGLE_MATCH,
        mem_discriminant::MEM_DISCRIMINANT_NON_ENUM,
        mem_replace::MEM_REPLACE_OPTION_WITH_NONE,
        methods::CHARS_LAST_CMP,
        methods::CHARS_NEXT_CMP,
        methods::CLONE_DOUBLE_REF,
        methods::CLONE_ON_COPY,
        methods::EXPECT_FUN_CALL,
        methods::FILTER_NEXT,
        methods::INTO_ITER_ON_ARRAY,
        methods::INTO_ITER_ON_REF,
        methods::ITER_CLONED_COLLECT,
        methods::ITER_NTH,
        methods::ITER_SKIP_NEXT,
        methods::NEW_RET_NO_SELF,
        methods::OK_EXPECT,
        methods::OPTION_MAP_OR_NONE,
        methods::OR_FUN_CALL,
        methods::SEARCH_IS_SOME,
        methods::SHOULD_IMPLEMENT_TRAIT,
        methods::SINGLE_CHAR_PATTERN,
        methods::STRING_EXTEND_CHARS,
        methods::TEMPORARY_CSTRING_AS_PTR,
        methods::UNNECESSARY_FILTER_MAP,
        methods::UNNECESSARY_FOLD,
        methods::USELESS_ASREF,
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
        needless_borrowed_ref::NEEDLESS_BORROWED_REFERENCE,
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
        ok_if_let::IF_LET_SOME_RESULT,
        open_options::NONSENSICAL_OPEN_OPTIONS,
        overflow_check_conditional::OVERFLOW_CHECK_CONDITIONAL,
        panic_unimplemented::PANIC_PARAMS,
        partialeq_ne_impl::PARTIALEQ_NE_IMPL,
        precedence::PRECEDENCE,
        ptr::CMP_NULL,
        ptr::MUT_FROM_REF,
        ptr::PTR_ARG,
        ptr_offset_with_cast::PTR_OFFSET_WITH_CAST,
        question_mark::QUESTION_MARK,
        ranges::ITERATOR_STEP_BY_ZERO,
        ranges::RANGE_MINUS_ONE,
        ranges::RANGE_PLUS_ONE,
        ranges::RANGE_ZIP_WITH_LEN,
        redundant_field_names::REDUNDANT_FIELD_NAMES,
        redundant_pattern_matching::REDUNDANT_PATTERN_MATCHING,
        redundant_static_lifetimes::REDUNDANT_STATIC_LIFETIMES,
        reference::DEREF_ADDROF,
        reference::REF_IN_DEREF,
        regex::INVALID_REGEX,
        regex::REGEX_MACRO,
        regex::TRIVIAL_REGEX,
        returns::LET_AND_RETURN,
        returns::NEEDLESS_RETURN,
        returns::UNUSED_UNIT,
        serde_api::SERDE_API_MISUSE,
        slow_vector_initialization::SLOW_VECTOR_INITIALIZATION,
        strings::STRING_LIT_AS_BYTES,
        suspicious_trait_impl::SUSPICIOUS_ARITHMETIC_IMPL,
        suspicious_trait_impl::SUSPICIOUS_OP_ASSIGN_IMPL,
        swap::ALMOST_SWAPPED,
        swap::MANUAL_SWAP,
        temporary_assignment::TEMPORARY_ASSIGNMENT,
        trait_bounds::TYPE_REPETITION_IN_BOUNDS,
        transmute::CROSSPOINTER_TRANSMUTE,
        transmute::TRANSMUTE_BYTES_TO_STR,
        transmute::TRANSMUTE_INT_TO_BOOL,
        transmute::TRANSMUTE_INT_TO_CHAR,
        transmute::TRANSMUTE_INT_TO_FLOAT,
        transmute::TRANSMUTE_PTR_TO_PTR,
        transmute::TRANSMUTE_PTR_TO_REF,
        transmute::USELESS_TRANSMUTE,
        transmute::WRONG_TRANSMUTE,
        transmuting_null::TRANSMUTING_NULL,
        trivially_copy_pass_by_ref::TRIVIALLY_COPY_PASS_BY_REF,
        try_err::TRY_ERR,
        types::ABSURD_EXTREME_COMPARISONS,
        types::BORROWED_BOX,
        types::BOX_VEC,
        types::CAST_LOSSLESS,
        types::CAST_PTR_ALIGNMENT,
        types::CAST_REF_TO_MUT,
        types::CHAR_LIT_AS_U8,
        types::FN_TO_NUMERIC_CAST,
        types::FN_TO_NUMERIC_CAST_WITH_TRUNCATION,
        types::IMPLICIT_HASHER,
        types::LET_UNIT_VALUE,
        types::OPTION_OPTION,
        types::TYPE_COMPLEXITY,
        types::UNIT_ARG,
        types::UNIT_CMP,
        types::UNNECESSARY_CAST,
        types::VEC_BOX,
        unicode::ZERO_WIDTH_SPACE,
        unsafe_removed_from_name::UNSAFE_REMOVED_FROM_NAME,
        unused_io_amount::UNUSED_IO_AMOUNT,
        unused_label::UNUSED_LABEL,
        unwrap::PANICKING_UNWRAP,
        unwrap::UNNECESSARY_UNWRAP,
        vec::USELESS_VEC,
        write::PRINTLN_EMPTY_STRING,
        write::PRINT_LITERAL,
        write::PRINT_WITH_NEWLINE,
        write::WRITELN_EMPTY_STRING,
        write::WRITE_LITERAL,
        write::WRITE_WITH_NEWLINE,
        zero_div_zero::ZERO_DIVIDED_BY_ZERO,
    ]);

    reg.register_lint_group("clippy::style", Some("clippy_style"), vec![
        assertions_on_constants::ASSERTIONS_ON_CONSTANTS,
        assign_ops::ASSIGN_OP_PATTERN,
        attrs::UNKNOWN_CLIPPY_LINTS,
        bit_mask::VERBOSE_BIT_MASK,
        blacklisted_name::BLACKLISTED_NAME,
        block_in_if_condition::BLOCK_IN_IF_CONDITION_EXPR,
        block_in_if_condition::BLOCK_IN_IF_CONDITION_STMT,
        collapsible_if::COLLAPSIBLE_IF,
        enum_variants::ENUM_VARIANT_NAMES,
        enum_variants::MODULE_INCEPTION,
        eq_op::OP_REF,
        eta_reduction::REDUNDANT_CLOSURE,
        excessive_precision::EXCESSIVE_PRECISION,
        formatting::SUSPICIOUS_ASSIGNMENT_FORMATTING,
        formatting::SUSPICIOUS_ELSE_FORMATTING,
        infallible_destructuring_match::INFALLIBLE_DESTRUCTURING_MATCH,
        inherent_to_string::INHERENT_TO_STRING,
        len_zero::LEN_WITHOUT_IS_EMPTY,
        len_zero::LEN_ZERO,
        let_if_seq::USELESS_LET_IF_SEQ,
        literal_representation::INCONSISTENT_DIGIT_GROUPING,
        literal_representation::UNREADABLE_LITERAL,
        loops::EMPTY_LOOP,
        loops::FOR_KV_MAP,
        loops::NEEDLESS_RANGE_LOOP,
        loops::WHILE_LET_ON_ITERATOR,
        main_recursion::MAIN_RECURSION,
        map_clone::MAP_CLONE,
        matches::MATCH_BOOL,
        matches::MATCH_OVERLAPPING_ARM,
        matches::MATCH_REF_PATS,
        matches::MATCH_WILD_ERR_ARM,
        matches::SINGLE_MATCH,
        mem_replace::MEM_REPLACE_OPTION_WITH_NONE,
        methods::CHARS_LAST_CMP,
        methods::INTO_ITER_ON_REF,
        methods::ITER_CLONED_COLLECT,
        methods::ITER_SKIP_NEXT,
        methods::NEW_RET_NO_SELF,
        methods::OK_EXPECT,
        methods::OPTION_MAP_OR_NONE,
        methods::SHOULD_IMPLEMENT_TRAIT,
        methods::STRING_EXTEND_CHARS,
        methods::UNNECESSARY_FOLD,
        methods::WRONG_SELF_CONVENTION,
        misc::REDUNDANT_PATTERN,
        misc::TOPLEVEL_REF_ARG,
        misc::ZERO_PTR,
        misc_early::BUILTIN_TYPE_SHADOW,
        misc_early::DOUBLE_NEG,
        misc_early::DUPLICATE_UNDERSCORE_ARGUMENT,
        misc_early::MIXED_CASE_HEX_LITERALS,
        misc_early::UNNEEDED_FIELD_PATTERN,
        mut_reference::UNNECESSARY_MUT_PASSED,
        neg_multiply::NEG_MULTIPLY,
        new_without_default::NEW_WITHOUT_DEFAULT,
        non_expressive_names::JUST_UNDERSCORES_AND_DIGITS,
        non_expressive_names::MANY_SINGLE_CHAR_NAMES,
        ok_if_let::IF_LET_SOME_RESULT,
        panic_unimplemented::PANIC_PARAMS,
        ptr::CMP_NULL,
        ptr::PTR_ARG,
        question_mark::QUESTION_MARK,
        redundant_field_names::REDUNDANT_FIELD_NAMES,
        redundant_pattern_matching::REDUNDANT_PATTERN_MATCHING,
        redundant_static_lifetimes::REDUNDANT_STATIC_LIFETIMES,
        regex::REGEX_MACRO,
        regex::TRIVIAL_REGEX,
        returns::LET_AND_RETURN,
        returns::NEEDLESS_RETURN,
        returns::UNUSED_UNIT,
        strings::STRING_LIT_AS_BYTES,
        try_err::TRY_ERR,
        types::FN_TO_NUMERIC_CAST,
        types::FN_TO_NUMERIC_CAST_WITH_TRUNCATION,
        types::IMPLICIT_HASHER,
        types::LET_UNIT_VALUE,
        unsafe_removed_from_name::UNSAFE_REMOVED_FROM_NAME,
        write::PRINTLN_EMPTY_STRING,
        write::PRINT_LITERAL,
        write::PRINT_WITH_NEWLINE,
        write::WRITELN_EMPTY_STRING,
        write::WRITE_LITERAL,
        write::WRITE_WITH_NEWLINE,
    ]);

    reg.register_lint_group("clippy::complexity", Some("clippy_complexity"), vec![
        assign_ops::MISREFACTORED_ASSIGN_OP,
        attrs::DEPRECATED_CFG_ATTR,
        booleans::NONMINIMAL_BOOL,
        cognitive_complexity::COGNITIVE_COMPLEXITY,
        double_comparison::DOUBLE_COMPARISONS,
        double_parens::DOUBLE_PARENS,
        duration_subsec::DURATION_SUBSEC,
        eval_order_dependence::DIVERGING_SUB_EXPRESSION,
        eval_order_dependence::EVAL_ORDER_DEPENDENCE,
        explicit_write::EXPLICIT_WRITE,
        format::USELESS_FORMAT,
        functions::TOO_MANY_ARGUMENTS,
        get_last_with_len::GET_LAST_WITH_LEN,
        identity_conversion::IDENTITY_CONVERSION,
        identity_op::IDENTITY_OP,
        int_plus_one::INT_PLUS_ONE,
        lifetimes::EXTRA_UNUSED_LIFETIMES,
        lifetimes::NEEDLESS_LIFETIMES,
        loops::EXPLICIT_COUNTER_LOOP,
        loops::MUT_RANGE_BOUND,
        loops::WHILE_LET_LOOP,
        map_unit_fn::OPTION_MAP_UNIT_FN,
        map_unit_fn::RESULT_MAP_UNIT_FN,
        matches::MATCH_AS_REF,
        methods::CHARS_NEXT_CMP,
        methods::CLONE_ON_COPY,
        methods::FILTER_NEXT,
        methods::SEARCH_IS_SOME,
        methods::UNNECESSARY_FILTER_MAP,
        methods::USELESS_ASREF,
        misc::SHORT_CIRCUIT_STATEMENT,
        misc_early::REDUNDANT_CLOSURE_CALL,
        misc_early::ZERO_PREFIXED_LITERAL,
        needless_bool::BOOL_COMPARISON,
        needless_bool::NEEDLESS_BOOL,
        needless_borrowed_ref::NEEDLESS_BORROWED_REFERENCE,
        needless_update::NEEDLESS_UPDATE,
        neg_cmp_op_on_partial_ord::NEG_CMP_OP_ON_PARTIAL_ORD,
        no_effect::NO_EFFECT,
        no_effect::UNNECESSARY_OPERATION,
        overflow_check_conditional::OVERFLOW_CHECK_CONDITIONAL,
        partialeq_ne_impl::PARTIALEQ_NE_IMPL,
        precedence::PRECEDENCE,
        ptr_offset_with_cast::PTR_OFFSET_WITH_CAST,
        ranges::RANGE_MINUS_ONE,
        ranges::RANGE_PLUS_ONE,
        ranges::RANGE_ZIP_WITH_LEN,
        reference::DEREF_ADDROF,
        reference::REF_IN_DEREF,
        swap::MANUAL_SWAP,
        temporary_assignment::TEMPORARY_ASSIGNMENT,
        trait_bounds::TYPE_REPETITION_IN_BOUNDS,
        transmute::CROSSPOINTER_TRANSMUTE,
        transmute::TRANSMUTE_BYTES_TO_STR,
        transmute::TRANSMUTE_INT_TO_BOOL,
        transmute::TRANSMUTE_INT_TO_CHAR,
        transmute::TRANSMUTE_INT_TO_FLOAT,
        transmute::TRANSMUTE_PTR_TO_PTR,
        transmute::TRANSMUTE_PTR_TO_REF,
        transmute::USELESS_TRANSMUTE,
        types::BORROWED_BOX,
        types::CAST_LOSSLESS,
        types::CHAR_LIT_AS_U8,
        types::OPTION_OPTION,
        types::TYPE_COMPLEXITY,
        types::UNIT_ARG,
        types::UNNECESSARY_CAST,
        types::VEC_BOX,
        unused_label::UNUSED_LABEL,
        unwrap::UNNECESSARY_UNWRAP,
        zero_div_zero::ZERO_DIVIDED_BY_ZERO,
    ]);

    reg.register_lint_group("clippy::correctness", Some("clippy_correctness"), vec![
        approx_const::APPROX_CONSTANT,
        attrs::DEPRECATED_SEMVER,
        attrs::USELESS_ATTRIBUTE,
        bit_mask::BAD_BIT_MASK,
        bit_mask::INEFFECTIVE_BIT_MASK,
        booleans::LOGIC_BUG,
        copies::IFS_SAME_COND,
        copies::IF_SAME_THEN_ELSE,
        derive::DERIVE_HASH_XOR_EQ,
        drop_bounds::DROP_BOUNDS,
        drop_forget_ref::DROP_COPY,
        drop_forget_ref::DROP_REF,
        drop_forget_ref::FORGET_COPY,
        drop_forget_ref::FORGET_REF,
        enum_clike::ENUM_CLIKE_UNPORTABLE_VARIANT,
        eq_op::EQ_OP,
        erasing_op::ERASING_OP,
        formatting::POSSIBLE_MISSING_COMMA,
        functions::NOT_UNSAFE_PTR_ARG_DEREF,
        indexing_slicing::OUT_OF_BOUNDS_INDEXING,
        infinite_iter::INFINITE_ITER,
        inherent_to_string::INHERENT_TO_STRING_SHADOW_DISPLAY,
        inline_fn_without_body::INLINE_FN_WITHOUT_BODY,
        literal_representation::MISTYPED_LITERAL_SUFFIXES,
        loops::FOR_LOOP_OVER_OPTION,
        loops::FOR_LOOP_OVER_RESULT,
        loops::ITER_NEXT_LOOP,
        loops::NEVER_LOOP,
        loops::REVERSE_RANGE_LOOP,
        loops::WHILE_IMMUTABLE_CONDITION,
        mem_discriminant::MEM_DISCRIMINANT_NON_ENUM,
        methods::CLONE_DOUBLE_REF,
        methods::INTO_ITER_ON_ARRAY,
        methods::TEMPORARY_CSTRING_AS_PTR,
        minmax::MIN_MAX,
        misc::CMP_NAN,
        misc::FLOAT_CMP,
        misc::MODULO_ONE,
        non_copy_const::BORROW_INTERIOR_MUTABLE_CONST,
        non_copy_const::DECLARE_INTERIOR_MUTABLE_CONST,
        open_options::NONSENSICAL_OPEN_OPTIONS,
        ptr::MUT_FROM_REF,
        ranges::ITERATOR_STEP_BY_ZERO,
        regex::INVALID_REGEX,
        serde_api::SERDE_API_MISUSE,
        suspicious_trait_impl::SUSPICIOUS_ARITHMETIC_IMPL,
        suspicious_trait_impl::SUSPICIOUS_OP_ASSIGN_IMPL,
        swap::ALMOST_SWAPPED,
        transmute::WRONG_TRANSMUTE,
        transmuting_null::TRANSMUTING_NULL,
        types::ABSURD_EXTREME_COMPARISONS,
        types::CAST_PTR_ALIGNMENT,
        types::CAST_REF_TO_MUT,
        types::UNIT_CMP,
        unicode::ZERO_WIDTH_SPACE,
        unused_io_amount::UNUSED_IO_AMOUNT,
        unwrap::PANICKING_UNWRAP,
    ]);

    reg.register_lint_group("clippy::perf", Some("clippy_perf"), vec![
        bytecount::NAIVE_BYTECOUNT,
        entry::MAP_ENTRY,
        escape::BOXED_LOCAL,
        large_enum_variant::LARGE_ENUM_VARIANT,
        loops::MANUAL_MEMCPY,
        loops::NEEDLESS_COLLECT,
        loops::UNUSED_COLLECT,
        methods::EXPECT_FUN_CALL,
        methods::ITER_NTH,
        methods::OR_FUN_CALL,
        methods::SINGLE_CHAR_PATTERN,
        misc::CMP_OWNED,
        mutex_atomic::MUTEX_ATOMIC,
        slow_vector_initialization::SLOW_VECTOR_INITIALIZATION,
        trivially_copy_pass_by_ref::TRIVIALLY_COPY_PASS_BY_REF,
        types::BOX_VEC,
        vec::USELESS_VEC,
    ]);

    reg.register_lint_group("clippy::cargo", Some("clippy_cargo"), vec![
        cargo_common_metadata::CARGO_COMMON_METADATA,
        multiple_crate_versions::MULTIPLE_CRATE_VERSIONS,
        wildcard_dependencies::WILDCARD_DEPENDENCIES,
    ]);

    reg.register_lint_group("clippy::nursery", Some("clippy_nursery"), vec![
        attrs::EMPTY_LINE_AFTER_OUTER_ATTR,
        fallible_impl_from::FALLIBLE_IMPL_FROM,
        missing_const_for_fn::MISSING_CONST_FOR_FN,
        mutex_atomic::MUTEX_INTEGER,
        needless_borrow::NEEDLESS_BORROW,
        path_buf_push_overwrite::PATH_BUF_PUSH_OVERWRITE,
        redundant_clone::REDUNDANT_CLONE,
    ]);
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
