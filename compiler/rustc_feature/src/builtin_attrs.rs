//! Built-in attributes and `cfg` flag gating.

use std::sync::LazyLock;

use rustc_ast::ast::Safety;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::AttrStyle;
use rustc_span::{Symbol, sym};

use crate::Features;

type GateFn = fn(&Features) -> bool;

pub type GatedCfg = (Symbol, Symbol, GateFn);

/// `cfg(...)`'s that are feature gated.
const GATED_CFGS: &[GatedCfg] = &[
    // (name in cfg, feature, function to check if the feature is enabled)
    (sym::overflow_checks, sym::cfg_overflow_checks, Features::cfg_overflow_checks),
    (sym::ub_checks, sym::cfg_ub_checks, Features::cfg_ub_checks),
    (sym::contract_checks, sym::cfg_contract_checks, Features::cfg_contract_checks),
    (sym::target_thread_local, sym::cfg_target_thread_local, Features::cfg_target_thread_local),
    (
        sym::target_has_atomic_load_store,
        sym::cfg_target_has_atomic,
        Features::cfg_target_has_atomic,
    ),
    (sym::sanitize, sym::cfg_sanitize, Features::cfg_sanitize),
    (sym::version, sym::cfg_version, Features::cfg_version),
    (sym::relocation_model, sym::cfg_relocation_model, Features::cfg_relocation_model),
    (sym::sanitizer_cfi_generalize_pointers, sym::cfg_sanitizer_cfi, Features::cfg_sanitizer_cfi),
    (sym::sanitizer_cfi_normalize_integers, sym::cfg_sanitizer_cfi, Features::cfg_sanitizer_cfi),
    // this is consistent with naming of the compiler flag it's for
    (sym::fmt_debug, sym::fmt_debug, Features::fmt_debug),
    (
        sym::target_has_reliable_f16,
        sym::cfg_target_has_reliable_f16_f128,
        Features::cfg_target_has_reliable_f16_f128,
    ),
    (
        sym::target_has_reliable_f16_math,
        sym::cfg_target_has_reliable_f16_f128,
        Features::cfg_target_has_reliable_f16_f128,
    ),
    (
        sym::target_has_reliable_f128,
        sym::cfg_target_has_reliable_f16_f128,
        Features::cfg_target_has_reliable_f16_f128,
    ),
    (
        sym::target_has_reliable_f128_math,
        sym::cfg_target_has_reliable_f16_f128,
        Features::cfg_target_has_reliable_f16_f128,
    ),
    (sym::target_object_format, sym::cfg_target_object_format, Features::cfg_target_object_format),
];

/// Find a gated cfg determined by the `pred`icate which is given the cfg's name.
pub fn find_gated_cfg(pred: impl Fn(Symbol) -> bool) -> Option<&'static GatedCfg> {
    GATED_CFGS.iter().find(|(cfg_sym, ..)| pred(*cfg_sym))
}

#[derive(Clone, Debug, Copy)]
pub enum AttributeStability {
    /// An attribute that is unstable behind a specified feature fagte
    Unstable {
        /// The feature gate, for example `rustc_attrs` for rustc_* attributes.
        gate_name: Symbol,
        /// Check function to be called during the `PostExpansionVisitor` pass, which will be one of the `Features::*` functions
        gate_check: fn(&Features) -> bool,
        /// Notes to be displayed when an attempt is made to use the attribute without its feature gate.
        notes: &'static [&'static str],
    },
    /// A stable attribute, can be used on all release channels
    Stable,
}

// FIXME(jdonszelmann): move to rustc_hir::attrs
/// A template that the attribute input must match.
/// Only top-level shape (`#[attr]` vs `#[attr(...)]` vs `#[attr = ...]`) is considered now.
#[derive(Clone, Copy, Default)]
pub struct AttributeTemplate {
    /// If `true`, the attribute is allowed to be a bare word like `#[test]`.
    pub word: bool,
    /// If `Some`, the attribute is allowed to take a list of items like `#[allow(..)]`.
    pub list: Option<&'static [&'static str]>,
    /// If non-empty, the attribute is allowed to take a list containing exactly
    /// one of the listed words, like `#[coverage(off)]`.
    pub one_of: &'static [Symbol],
    /// If `Some`, the attribute is allowed to be a name/value pair where the
    /// value is a string, like `#[must_use = "reason"]`.
    pub name_value_str: Option<&'static [&'static str]>,
    /// A link to the document for this attribute.
    pub docs: Option<&'static str>,
}

pub enum AttrSuggestionStyle {
    /// The suggestion is styled for a normal attribute.
    /// The `AttrStyle` determines whether this is an inner or outer attribute.
    Attribute(AttrStyle),
    /// The suggestion is styled for an attribute embedded into another attribute.
    /// For example, attributes inside `#[cfg_attr(true, attr(...)]`.
    EmbeddedAttribute,
    /// The suggestion is styled for macros that are parsed with attribute parsers.
    /// For example, the `cfg!(predicate)` macro.
    Macro,
}

impl AttributeTemplate {
    pub fn suggestions(
        &self,
        style: AttrSuggestionStyle,
        safety: Safety,
        name: impl std::fmt::Display,
    ) -> Vec<String> {
        let (start, macro_call, end) = match style {
            AttrSuggestionStyle::Attribute(AttrStyle::Outer) => ("#[", "", "]"),
            AttrSuggestionStyle::Attribute(AttrStyle::Inner) => ("#![", "", "]"),
            AttrSuggestionStyle::Macro => ("", "!", ""),
            AttrSuggestionStyle::EmbeddedAttribute => ("", "", ""),
        };

        let mut suggestions = vec![];

        let (safety_start, safety_end) = match safety {
            Safety::Unsafe(_) => ("unsafe(", ")"),
            _ => ("", ""),
        };

        if self.word {
            debug_assert!(macro_call.is_empty(), "Macro suggestions use list style");
            suggestions.push(format!("{start}{safety_start}{name}{safety_end}{end}"));
        }
        if let Some(descr) = self.list {
            for descr in descr {
                suggestions.push(format!(
                    "{start}{safety_start}{name}{macro_call}({descr}){safety_end}{end}"
                ));
            }
        }
        suggestions.extend(
            self.one_of
                .iter()
                .map(|&word| format!("{start}{safety_start}{name}({word}){safety_end}{end}")),
        );
        if let Some(descr) = self.name_value_str {
            debug_assert!(macro_call.is_empty(), "Macro suggestions use list style");
            for descr in descr {
                suggestions
                    .push(format!("{start}{safety_start}{name} = \"{descr}\"{safety_end}{end}"));
            }
        }
        suggestions.sort();

        suggestions
    }
}

/// A convenience macro for constructing attribute templates.
/// E.g., `template!(Word, List: "description")` means that the attribute
/// supports forms `#[attr]` and `#[attr(description)]`.
#[macro_export]
macro_rules! template {
    (Word) => { $crate::template!(@ true, None, &[], None, None) };
    (Word, $link: literal) => { $crate::template!(@ true, None, &[], None, Some($link)) };
    (List: $descr: expr) => { $crate::template!(@ false, Some($descr), &[], None, None) };
    (List: $descr: expr, $link: literal) => { $crate::template!(@ false, Some($descr), &[], None, Some($link)) };
    (OneOf: $one_of: expr) => { $crate::template!(@ false, None, $one_of, None, None) };
    (NameValueStr: [$($descr: literal),* $(,)?]) => { $crate::template!(@ false, None, &[], Some(&[$($descr,)*]), None) };
    (NameValueStr: [$($descr: literal),* $(,)?], $link: literal) => { $crate::template!(@ false, None, &[], Some(&[$($descr,)*]), Some($link)) };
    (NameValueStr: $descr: literal) => { $crate::template!(@ false, None, &[], Some(&[$descr]), None) };
    (NameValueStr: $descr: literal, $link: literal) => { $crate::template!(@ false, None, &[], Some(&[$descr]), Some($link)) };
    (Word, List: $descr: expr) => { $crate::template!(@ true, Some($descr), &[], None, None) };
    (Word, List: $descr: expr, $link: literal) => { $crate::template!(@ true, Some($descr), &[], None, Some($link)) };
    (Word, NameValueStr: $descr: expr) => { $crate::template!(@ true, None, &[], Some(&[$descr]), None) };
    (Word, NameValueStr: $descr: expr, $link: literal) => { $crate::template!(@ true, None, &[], Some(&[$descr]), Some($link)) };
    (List: $descr1: expr, NameValueStr: $descr2: expr) => {
        $crate::template!(@ false, Some($descr1), &[], Some(&[$descr2]), None)
    };
    (List: $descr1: expr, NameValueStr: $descr2: expr, $link: literal) => {
        $crate::template!(@ false, Some($descr1), &[], Some(&[$descr2]), Some($link))
    };
    (Word, List: $descr1: expr, NameValueStr: $descr2: expr) => {
        $crate::template!(@ true, Some($descr1), &[], Some(&[$descr2]), None)
    };
    (Word, List: $descr1: expr, NameValueStr: $descr2: expr, $link: literal) => {
        $crate::template!(@ true, Some($descr1), &[], Some(&[$descr2]), Some($link))
    };
    (@ $word: expr, $list: expr, $one_of: expr, $name_value_str: expr, $link: expr) => { $crate::AttributeTemplate {
        word: $word, list: $list, one_of: $one_of, name_value_str: $name_value_str, docs: $link,
    } };
}

/// Attributes that have a special meaning to rustc or rustdoc.
#[rustfmt::skip]
pub static BUILTIN_ATTRIBUTES: &[Symbol] = &[
    // ==========================================================================
    // Stable attributes:
    // ==========================================================================

    // Conditional compilation:
    sym::cfg,
    sym::cfg_attr,

    // Testing:
    sym::ignore,
    sym::should_panic,

    // Macros:
    sym::automatically_derived,
    sym::macro_use,
    sym::macro_escape, // Deprecated synonym for `macro_use`.
    sym::macro_export,
    sym::proc_macro,
    sym::proc_macro_derive,
    sym::proc_macro_attribute,

    // Lints:
    sym::warn,
    sym::allow,
    sym::expect,
    sym::forbid,
    sym::deny,
    sym::must_use,
    sym::must_not_suspend,
    sym::deprecated,

    // Crate properties:
    sym::crate_name,
    sym::crate_type,

    // ABI, linking, symbols, and FFI
    sym::link,
    sym::link_name,
    sym::no_link,
    sym::repr,
    // FIXME(#82232, #143834): temporarily renamed to mitigate `#[align]` nameres ambiguity
    sym::rustc_align,
    sym::rustc_align_static,
    sym::export_name,
    sym::link_section,
    sym::no_mangle,
    sym::used,
    sym::link_ordinal,
    sym::naked,
    // See `TyAndLayout::pass_indirectly_in_non_rustic_abis` for details.
    sym::rustc_pass_indirectly_in_non_rustic_abis,

    // Limits:
    sym::recursion_limit,
    sym::type_length_limit,
    sym::move_size_limit,

    // Entry point:
    sym::no_main,

    // Modules, prelude, and resolution:
    sym::path,
    sym::no_std,
    sym::no_implicit_prelude,
    sym::non_exhaustive,

    // Runtime
    sym::windows_subsystem,
    sym::panic_handler, // RFC 2070

    // Code generation:
    sym::inline,
    sym::cold,
    sym::no_builtins,
    sym::target_feature,
    sym::track_caller,
    sym::instruction_set,
    sym::force_target_feature,
    sym::sanitize,
    sym::coverage,

    sym::doc,

    // Debugging
    sym::debugger_visualizer,
    sym::collapse_debuginfo,

    // ==========================================================================
    // Unstable attributes:
    // ==========================================================================

    // Linking:
    sym::export_stable,

    // Testing:
    sym::test_runner,

    sym::reexport_test_harness_main,

    // RFC #1268
    sym::marker,
    sym::thread_local,
    sym::no_core,
    // RFC 2412
    sym::optimize,

    sym::ffi_pure,
    sym::ffi_const,
    sym::register_tool,
    // `#[cfi_encoding = ""]`
    sym::cfi_encoding,

    // `#[coroutine]` attribute to be applied to closures to make them coroutines instead
    sym::coroutine,

    // RFC 3543
    // `#[patchable_function_entry(prefix_nops = m, entry_nops = n)]`
    sym::patchable_function_entry,

    // The `#[loop_match]` and `#[const_continue]` attributes are part of the
    // lang experiment for RFC 3720 tracked in:
    //
    // - https://github.com/rust-lang/rust/issues/132306
    sym::const_continue,
    sym::loop_match,

    // The `#[pin_v2]` attribute is part of the `pin_ergonomics` experiment
    // that allows structurally pinning, tracked in:
    //
    // - https://github.com/rust-lang/rust/issues/130494
    sym::pin_v2,

    // ==========================================================================
    // Internal attributes: Stability, deprecation, and unsafe:
    // ==========================================================================

    sym::feature,
    // DuplicatesOk since it has its own validation
    sym::stable,
    sym::unstable,
    sym::unstable_feature_bound,
    sym::unstable_removed,
    sym::rustc_const_unstable,
    sym::rustc_const_stable,
    sym::rustc_default_body_unstable,
    sym::allow_internal_unstable,
    sym::allow_internal_unsafe,
    sym::rustc_eii_foreign_item,
    sym::rustc_allowed_through_unstable_modules,
    sym::rustc_deprecated_safe_2024,
    sym::rustc_pub_transparent,


    // ==========================================================================
    // Internal attributes: Type system related:
    // ==========================================================================

    sym::fundamental,
    sym::may_dangle,

    sym::rustc_never_type_options,

    // ==========================================================================
    // Internal attributes: Runtime related:
    // ==========================================================================

    sym::rustc_allocator,
    sym::rustc_nounwind,
    sym::rustc_reallocator,
    sym::rustc_deallocator,
    sym::rustc_allocator_zeroed,
    sym::rustc_allocator_zeroed_variant,
    sym::default_lib_allocator,
    sym::needs_allocator,
    sym::panic_runtime,
    sym::needs_panic_runtime,
    sym::compiler_builtins,
    sym::profiler_runtime,

    // ==========================================================================
    // Internal attributes, Linkage:
    // ==========================================================================

    sym::linkage,
    sym::rustc_std_internal_symbol,
    sym::rustc_objc_class,
    sym::rustc_objc_selector,

    // ==========================================================================
    // Internal attributes, Macro related:
    // ==========================================================================

    sym::rustc_builtin_macro,
    sym::rustc_proc_macro_decls,
    sym::rustc_macro_transparency,
    sym::rustc_autodiff,
    sym::rustc_offload_kernel,
    // Traces that are left when `cfg` and `cfg_attr` attributes are expanded.
    // The attributes are not gated, to avoid stability errors, but they cannot be used in stable
    // or unstable code directly because `sym::cfg_(attr_)trace` are not valid identifiers, they
    // can only be generated by the compiler.
    sym::cfg_trace,
    sym::cfg_attr_trace,

    // ==========================================================================
    // Internal attributes, Diagnostics related:
    // ==========================================================================

    sym::rustc_on_unimplemented,
    sym::rustc_confusables,
    // Enumerates "identity-like" conversion methods to suggest on type mismatch.
    sym::rustc_conversion_suggestion,
    // Prevents field reads in the marked trait or method to be considered
    // during dead code analysis.
    sym::rustc_trivial_field_reads,
    // Used by the `rustc::potential_query_instability` lint to warn methods which
    // might not be stable during incremental compilation.
    sym::rustc_lint_query_instability,
    // Used by the `rustc::untracked_query_information` lint to warn methods which
    // might not be stable during incremental compilation.
    sym::rustc_lint_untracked_query_information,
    // Used by the `rustc::bad_opt_access` lint to identify `DebuggingOptions` and `CodegenOptions`
    // types (as well as any others in future).
    sym::rustc_lint_opt_ty,
    // Used by the `rustc::bad_opt_access` lint on fields
    // types (as well as any others in future).
    sym::rustc_lint_opt_deny_field_access,

    // ==========================================================================
    // Internal attributes, Const related:
    // ==========================================================================

    sym::rustc_promotable,
    sym::rustc_legacy_const_generics,
    // Do not const-check this function's body. It will always get replaced during CTFE via `hook_special_const_fn`.
    sym::rustc_do_not_const_check,
    sym::rustc_const_stable_indirect,
    sym::rustc_intrinsic_const_stable_indirect,
    sym::rustc_allow_const_fn_unstable,

    // ==========================================================================
    // Internal attributes, Layout related:
    // ==========================================================================

    sym::rustc_simd_monomorphize_lane_limit,
    sym::rustc_nonnull_optimization_guaranteed,

    // ==========================================================================
    // Internal attributes, Misc:
    // ==========================================================================
    sym::lang,
    sym::rustc_as_ptr,
    sym::rustc_should_not_be_called_on_const_items,
    sym::rustc_pass_by_value,
    sym::rustc_never_returns_null_ptr,
    sym::rustc_no_implicit_autorefs,
    sym::rustc_coherence_is_core,
    sym::rustc_coinductive,
    sym::rustc_comptime,
    sym::rustc_allow_incoherent_impl,
    sym::rustc_preserve_ub_checks,
    sym::rustc_deny_explicit_impl,
    sym::rustc_dyn_incompatible_trait,
    sym::rustc_has_incoherent_inherent_impls,
    sym::rustc_non_const_trait_method,

    sym::rustc_diagnostic_item,
    sym::prelude_import,
    sym::rustc_paren_sugar,
    sym::rustc_inherit_overflow_checks,
    sym::rustc_reservation_impl,
    sym::rustc_test_marker,
    sym::rustc_unsafe_specialization_marker,
    sym::rustc_specialization_trait,
    sym::rustc_main,
    sym::rustc_skip_during_method_dispatch,
    sym::rustc_must_implement_one_of,
    sym::rustc_doc_primitive,
    sym::rustc_intrinsic,
    sym::rustc_no_mir_inline,
    sym::rustc_force_inline,
    sym::rustc_scalable_vector,
    sym::rustc_must_match_exhaustively,
    sym::rustc_no_writable,

    // ==========================================================================
    // Internal attributes, Testing:
    // ==========================================================================

    sym::rustc_effective_visibility,
    sym::rustc_dump_inferred_outlives,
    sym::rustc_capture_analysis,
    sym::rustc_insignificant_dtor,
    sym::rustc_no_implicit_bounds,
    sym::rustc_strict_coherence,
    sym::rustc_dump_variances,
    sym::rustc_dump_variances_of_opaques,
    sym::rustc_dump_hidden_type_of_opaques,
    sym::rustc_dump_layout,
    sym::rustc_abi,
    sym::rustc_regions,
    sym::rustc_delayed_bug_from_inside_query,
    sym::rustc_dump_user_args,
    sym::rustc_evaluate_where_clauses,
    sym::rustc_if_this_changed,
    sym::rustc_then_this_would_need,
    sym::rustc_clean,
    sym::rustc_partition_reused,
    sym::rustc_partition_codegened,
    sym::rustc_expected_cgu_reuse,
    sym::rustc_dump_symbol_name,
    sym::rustc_dump_def_path,
    sym::rustc_mir,
    sym::custom_mir,
    sym::rustc_dump_item_bounds,
    sym::rustc_dump_predicates,
    sym::rustc_dump_def_parents,
    sym::rustc_dump_object_lifetime_defaults,
    sym::rustc_dump_vtable,
    sym::rustc_dummy,
    sym::pattern_complexity_limit,
];

pub fn is_builtin_attr_name(name: Symbol) -> bool {
    BUILTIN_ATTRIBUTE_MAP.get(&name).is_some()
}

pub static BUILTIN_ATTRIBUTE_MAP: LazyLock<FxHashSet<Symbol>> = LazyLock::new(|| {
    let mut map = FxHashSet::default();
    for attr in BUILTIN_ATTRIBUTES.iter() {
        if !map.insert(*attr) {
            panic!("duplicate builtin attribute `{}`", attr);
        }
    }
    map
});
