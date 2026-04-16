//! Built-in attributes and `cfg` flag gating.

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
        sym::target_has_atomic_equal_alignment,
        sym::cfg_target_has_atomic_equal_alignment,
        Features::cfg_target_has_atomic_equal_alignment,
    ),
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
    (sym::emscripten_wasm_eh, sym::cfg_emscripten_wasm_eh, Features::cfg_emscripten_wasm_eh),
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
pub enum AttributeGate {
    /// A gated attribute which requires a feature gate to be enabled.
    Gated { gated_attr: GatedAttribute, kind: GateKind },
    /// Ungated attribute, can be used on all release channels
    Ungated,
}

#[derive(Clone, Debug, Copy)]
pub enum GateKind {
    Error,
    Ignore,
}

#[derive(Clone, Debug, Copy)]
pub struct GatedAttribute {
    /// The feature gate, for example `#![feature(rustc_attrs)]` for rustc_* attributes.
    pub feature: Symbol,
    /// The error message displayed when an attempt is made to use the attribute without its feature gate.
    pub message: &'static str,
    /// Check function to be called during the `PostExpansionVisitor` pass.
    pub check: fn(&Features) -> bool,
    /// Notes to be displayed when an attempt is made to use the attribute without its feature gate.
    pub notes: &'static [&'static str],
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
        name: impl std::fmt::Display,
    ) -> Vec<String> {
        let (start, macro_call, end) = match style {
            AttrSuggestionStyle::Attribute(AttrStyle::Outer) => ("#[", "", "]"),
            AttrSuggestionStyle::Attribute(AttrStyle::Inner) => ("#![", "", "]"),
            AttrSuggestionStyle::Macro => ("", "!", ""),
            AttrSuggestionStyle::EmbeddedAttribute => ("", "", ""),
        };

        let mut suggestions = vec![];

        if self.word {
            debug_assert!(macro_call.is_empty(), "Macro suggestions use list style");
            suggestions.push(format!("{start}{name}{end}"));
        }
        if let Some(descr) = self.list {
            for descr in descr {
                suggestions.push(format!("{start}{name}{macro_call}({descr}){end}"));
            }
        }
        suggestions.extend(self.one_of.iter().map(|&word| format!("{start}{name}({word}){end}")));
        if let Some(descr) = self.name_value_str {
            debug_assert!(macro_call.is_empty(), "Macro suggestions use list style");
            for descr in descr {
                suggestions.push(format!("{start}{name} = \"{descr}\"{end}"));
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

pub struct BuiltinAttribute {
    pub name: Symbol,
    pub gate: AttributeGate,
}

/// Attributes that have a special meaning to rustc or rustdoc.
pub const BUILTIN_ATTRIBUTES: &[Symbol] = {
    use sym::*;
    &[
        // ==========================================================================
        // Stable attributes:
        // ==========================================================================

        // Conditional compilation:
        cfg,
        cfg_attr,
        // Testing:
        ignore,
        should_panic,
        // Macros:
        automatically_derived,
        macro_use,
        macro_escape, // Deprecated synonym for `macro_use`.
        macro_export,
        proc_macro,
        proc_macro_derive,
        proc_macro_attribute,
        // Lints:
        warn,
        allow,
        expect,
        forbid,
        deny,
        must_use,
        must_not_suspend,
        deprecated,
        // Crate properties:
        crate_name,
        crate_type,
        // ABI, linking, symbols, and FFI
        link,
        link_name,
        no_link,
        repr,
        // FIXME(#82232, #143834): temporarily renamed to mitigate `#[align]` nameres ambiguity
        rustc_align,
        rustc_align_static,
        export_name,
        link_section,
        no_mangle,
        used,
        link_ordinal,
        naked,
        export_name,
        link_section,
        no_mangle,
        used,
        link_ordinal,
        naked,
        // See `TyAndLayout::pass_indirectly_in_non_rustic_abis` for details.
        rustc_pass_indirectly_in_non_rustic_abis,
        // Limits:
        recursion_limit,
        type_length_limit,
        move_size_limit,
        // Entry point:
        no_main,
        // Modules, prelude, and resolution:
        path,
        no_std,
        no_implicit_prelude,
        non_exhaustive,
        // Runtime
        windows_subsystem,
        // RFC 2070
        panic_handler,
        // Code generation:
        inline,
        cold,
        no_builtins,
        target_feature,
        track_caller,
        instruction_set,
        force_target_feature,
        sanitize,
        coverage,
        doc,
        // Debugging
        debugger_visualizer,
        collapse_debuginfo,
        debugger_visualizer,
        collapse_debuginfo,
        // ==========================================================================
        // Unstable attributes:
        // ==========================================================================

        // Linking:
        export_stable,
        // Testing:
        test_runner,
        reexport_test_harness_main,
        // RFC #1268
        marker,
        thread_local,
        no_core,
        // RFC 2412
        optimize,
        ffi_pure,
        ffi_const,
        register_tool,
        // `#[cfi_encoding = ""]`
        cfi_encoding,
        // `#[coroutine]` attribute to be applied to closures to make them coroutines instead
        coroutine,
        // RFC 3543
        // `#[patchable_function_entry(prefix_nops = m, entry_nops = n)]`
        patchable_function_entry,
        // The `#[loop_match]` and `#[const_continue]` attributes are part of the
        // lang experiment for RFC 3720 tracked in:
        //
        // - https://github.com/rust-lang/rust/issues/132306
        const_continue,
        loop_match,
        // The `#[pin_v2]` attribute is part of the `pin_ergonomics` experiment
        // that allows structurally pinning, tracked in:
        //
        // - https://github.com/rust-lang/rust/issues/130494
        pin_v2,
        // ==========================================================================
        // Internal attributes: Stability, deprecation, and unsafe:
        // ==========================================================================
        feature,
        // DuplicatesOk since it has its own validation
        stable,
        unstable,
        unstable_feature_bound,
        unstable_removed,
        rustc_const_unstable,
        rustc_const_stable,
        rustc_default_body_unstable,
        allow_internal_unstable,
        allow_internal_unsafe,
        rustc_eii_foreign_item,
        rustc_allowed_through_unstable_modules,
        rustc_deprecated_safe_2024,
        rustc_pub_transparent,
        // ==========================================================================
        // Internal attributes: Type system related:
        // ==========================================================================
        fundamental,
        may_dangle,
        rustc_never_type_options,
        // ==========================================================================
        // Internal attributes: Runtime related:
        // ==========================================================================
        rustc_allocator,
        rustc_nounwind,
        rustc_reallocator,
        rustc_deallocator,
        rustc_allocator_zeroed,
        rustc_allocator_zeroed_variant,
        default_lib_allocator,
        needs_allocator,
        panic_runtime,
        needs_panic_runtime,
        compiler_builtins,
        profiler_runtime,
        // ==========================================================================
        // Internal attributes, Linkage:
        // ==========================================================================
        linkage,
        rustc_std_internal_symbol,
        rustc_objc_class,
        rustc_objc_selector,
        // ==========================================================================
        // Internal attributes, Macro related:
        // ==========================================================================
        rustc_builtin_macro,
        rustc_proc_macro_decls,
        rustc_macro_transparency,
        rustc_autodiff,
        rustc_offload_kernel,
        // Traces that are left when `cfg` and `cfg_attr` attributes are expanded.
        // The attributes are not gated, to avoid stability errors, but they cannot be used in stable
        // or unstable code directly because `sym::cfg_(attr_)trace` are not valid identifiers, they
        // can only be generated by the compiler.
        cfg_trace,
        cfg_attr_trace,
        // ==========================================================================
        // Internal attributes, Diagnostics related:
        // ==========================================================================
        rustc_on_unimplemented,
        rustc_confusables,
        // Enumerates "identity-like" conversion methods to suggest on type mismatch.
        rustc_conversion_suggestion,
        // Prevents field reads in the marked trait or method to be considered
        // during dead code analysis.
        rustc_trivial_field_reads,
        // Used by the `rustc::potential_query_instability` lint to warn methods which
        // might not be stable during incremental compilation.
        rustc_lint_query_instability,
        // Used by the `rustc::untracked_query_information` lint to warn methods which
        // might not be stable during incremental compilation.
        rustc_lint_untracked_query_information,
        // Used by the `rustc::bad_opt_access` lint to identify `DebuggingOptions` and `CodegenOptions`
        // types (as well as any others in future).
        rustc_lint_opt_ty,
        // Used by the `rustc::bad_opt_access` lint on fields
        // types (as well as any others in future).
        rustc_lint_opt_deny_field_access,
        // ==========================================================================
        // Internal attributes, Const related:
        // ==========================================================================
        rustc_promotable,
        rustc_legacy_const_generics,
        // Do not const-check this function's body. It will always get replaced during CTFE via `hook_special_const_fn`.
        rustc_do_not_const_check,
        rustc_const_stable_indirect,
        rustc_intrinsic_const_stable_indirect,
        rustc_allow_const_fn_unstable,
        // ==========================================================================
        // Internal attributes, Layout related:
        // ==========================================================================
        rustc_layout_scalar_valid_range_start,
        rustc_layout_scalar_valid_range_end,
        rustc_simd_monomorphize_lane_limit,
        rustc_nonnull_optimization_guaranteed,
        // ==========================================================================
        // Internal attributes, Misc:
        // ==========================================================================
        lang,
        rustc_as_ptr,
        rustc_should_not_be_called_on_const_items,
        rustc_pass_by_value,
        rustc_never_returns_null_ptr,
        rustc_no_implicit_autorefs,
        rustc_coherence_is_core,
        rustc_coinductive,
        rustc_allow_incoherent_impl,
        rustc_preserve_ub_checks,
        rustc_deny_explicit_impl,
        rustc_dyn_incompatible_trait,
        rustc_has_incoherent_inherent_impls,
        rustc_non_const_trait_method,
        rustc_diagnostic_item,
        prelude_import,
        rustc_paren_sugar,
        rustc_inherit_overflow_checks,
        rustc_reservation_impl,
        rustc_reservation_impl,
        rustc_test_marker,
        rustc_unsafe_specialization_marker,
        rustc_specialization_trait,
        rustc_main,
        rustc_main,
        rustc_skip_during_method_dispatch,
        rustc_must_implement_one_of,
        rustc_doc_primitive,
        rustc_intrinsic,
        rustc_no_mir_inline,
        rustc_force_inline,
        rustc_scalable_vector,
        rustc_must_match_exhaustively,
        // ==========================================================================
        // Internal attributes, Testing:
        // ==========================================================================
        rustc_effective_visibility,
        rustc_dump_inferred_outlives,
        rustc_capture_analysis,
        rustc_insignificant_dtor,
        rustc_no_implicit_bounds,
        rustc_strict_coherence,
        rustc_dump_variances,
        rustc_dump_variances_of_opaques,
        rustc_dump_hidden_type_of_opaques,
        rustc_dump_layout,
        rustc_abi,
        rustc_regions,
        rustc_delayed_bug_from_inside_query,
        rustc_dump_user_args,
        rustc_evaluate_where_clauses,
        rustc_if_this_changed,
        rustc_then_this_would_need,
        rustc_clean,
        rustc_expected_cgu_reuse,
        rustc_dump_symbol_name,
        rustc_dump_def_path,
        rustc_mir,
        custom_mir,
        rustc_dump_item_bounds,
        rustc_dump_predicates,
        rustc_dump_def_parents,
        rustc_dump_object_lifetime_defaults,
        rustc_dump_vtable,
        rustc_dummy,
        pattern_complexity_limit,
        // FIXME find correct place
        rustc_partition_reused,
        rustc_partition_codegened,
    ]
};

pub fn is_stable_diagnostic_attribute(sym: Symbol, features: &Features) -> bool {
    match sym {
        sym::on_unimplemented | sym::do_not_recommend => true,
        sym::on_const => features.diagnostic_on_const(),
        sym::on_move => features.diagnostic_on_move(),
        _ => false,
    }
}
