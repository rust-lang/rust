//! Built-in attributes and `cfg` flag gating.

use std::sync::LazyLock;

use rustc_data_structures::fx::FxHashSet;
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
        sym::target_has_reliable_f16b,
        sym::cfg_target_has_reliable_f16b,
        Features::cfg_target_has_reliable_f16b,
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
    (sym::target_has_threads, sym::cfg_target_has_threads, Features::cfg_target_has_threads),
    (sym::target_object_format, sym::cfg_target_object_format, Features::cfg_target_object_format),
];

/// Find a gated cfg matching `name`.
pub fn find_gated_cfg(name: Symbol) -> Option<&'static GatedCfg> {
    GATED_CFGS.iter().find(|(cfg_sym, ..)| name == *cfg_sym)
}

#[derive(Clone, Debug, Copy)]
pub enum AttributeStability {
    /// An attribute that is unstable behind a specified feature gate.
    Unstable {
        /// The feature gate, for example `rustc_attrs` for rustc_* attributes.
        gate_name: Symbol,
        /// Notes to be displayed when an attempt is made to use the attribute without its feature
        /// gate.
        notes: &'static [&'static str],
    },
    /// A stable attribute, can be used on all release channels
    Stable,
}

/// Enumerates all built-in attributes.
pub static BUILTIN_ATTRIBUTES: &[Symbol] = &[
    // tidy-alphabetical-start
    sym::allow,
    sym::allow_internal_unsafe,
    sym::allow_internal_unstable,
    sym::automatically_derived,
    sym::cfg,
    sym::cfg_attr,
    sym::cfi_encoding,
    sym::cold,
    sym::collapse_debuginfo,
    sym::compiler_builtins,
    sym::const_continue,
    sym::coroutine,
    sym::coverage,
    sym::crate_name,
    sym::crate_type,
    sym::custom_mir,
    sym::debugger_visualizer,
    sym::default_lib_allocator,
    sym::deny,
    sym::deprecated,
    sym::doc,
    sym::expect,
    sym::export_name,
    sym::export_stable,
    sym::feature,
    sym::ffi_const,
    sym::ffi_pure,
    sym::forbid,
    sym::force_target_feature,
    sym::fundamental,
    sym::ignore,
    sym::inline,
    sym::instruction_set,
    sym::instrument_fn,
    sym::lang,
    sym::link,
    sym::link_name,
    sym::link_ordinal,
    sym::link_section,
    sym::linkage,
    sym::loop_match,
    sym::macro_escape,
    sym::macro_export,
    sym::macro_use,
    sym::marker,
    sym::may_dangle,
    sym::move_size_limit,
    sym::must_not_suspend,
    sym::must_use,
    sym::naked,
    sym::needs_allocator,
    sym::needs_panic_runtime,
    sym::no_builtins,
    sym::no_core,
    sym::no_implicit_prelude,
    sym::no_link,
    sym::no_main,
    sym::no_mangle,
    sym::no_std,
    sym::non_exhaustive,
    sym::optimize,
    sym::panic_handler,
    sym::panic_runtime,
    sym::patchable_function_entry,
    sym::path,
    sym::pattern_complexity_limit,
    sym::pin_v2,
    sym::prelude_import,
    sym::proc_macro,
    sym::proc_macro_attribute,
    sym::proc_macro_derive,
    sym::profiler_runtime,
    sym::recursion_limit,
    sym::reexport_test_harness_main,
    sym::register_attribute_tool,
    sym::register_lint_tool,
    sym::register_tool,
    sym::repr,
    sym::rustc_abi,
    sym::rustc_align,
    sym::rustc_align_static,
    sym::rustc_allocator,
    sym::rustc_allocator_zeroed,
    sym::rustc_allocator_zeroed_variant,
    sym::rustc_allow_const_fn_unstable,
    sym::rustc_allow_incoherent_impl,
    sym::rustc_allow_lifetime_dependent_specialization,
    sym::rustc_allowed_through_unstable_modules,
    sym::rustc_always_gca,
    sym::rustc_assert_variance,
    sym::rustc_autodiff,
    sym::rustc_builtin_macro,
    sym::rustc_canonical_symbol,
    sym::rustc_capture_analysis,
    sym::rustc_clean,
    sym::rustc_coherence_is_core,
    sym::rustc_coinductive,
    sym::rustc_comptime,
    sym::rustc_confusables,
    sym::rustc_const_stable,
    sym::rustc_const_stable_indirect,
    sym::rustc_const_unstable,
    sym::rustc_conversion_suggestion,
    sym::rustc_deallocator,
    sym::rustc_default_body_unstable,
    sym::rustc_delayed_bug_from_inside_query,
    sym::rustc_deny_explicit_impl,
    sym::rustc_deprecated_safe_2024,
    sym::rustc_diagnostic_item,
    sym::rustc_diagnostic_opaque,
    sym::rustc_do_not_const_check,
    sym::rustc_doc_primitive,
    sym::rustc_dummy,
    sym::rustc_dump_clauses,
    sym::rustc_dump_def_parents,
    sym::rustc_dump_def_path,
    sym::rustc_dump_generics,
    sym::rustc_dump_hidden_type_of_opaques,
    sym::rustc_dump_inferred_outlives,
    sym::rustc_dump_item_bounds,
    sym::rustc_dump_layout,
    sym::rustc_dump_object_lifetime_defaults,
    sym::rustc_dump_symbol_name,
    sym::rustc_dump_user_args,
    sym::rustc_dump_variances,
    sym::rustc_dump_variances_of_opaques,
    sym::rustc_dump_vtable,
    sym::rustc_dyn_incompatible_trait,
    sym::rustc_effective_visibility,
    sym::rustc_eii_foreign_item,
    sym::rustc_evaluate_where_clauses,
    sym::rustc_expected_cgu_reuse,
    sym::rustc_force_inline,
    sym::rustc_has_incoherent_inherent_impls,
    sym::rustc_if_this_changed,
    sym::rustc_inherit_overflow_checks,
    sym::rustc_insignificant_dtor,
    sym::rustc_intrinsic,
    sym::rustc_intrinsic_const_stable_indirect,
    sym::rustc_legacy_const_generics,
    sym::rustc_lint_opt_deny_field_access,
    sym::rustc_lint_opt_ty,
    sym::rustc_lint_query_instability,
    sym::rustc_lint_untracked_query_information,
    sym::rustc_macro_transparency,
    sym::rustc_main,
    sym::rustc_mir,
    sym::rustc_must_implement_one_of,
    sym::rustc_must_match_exhaustively,
    sym::rustc_no_implicit_autorefs,
    sym::rustc_no_implicit_bounds,
    sym::rustc_no_mir_inline,
    sym::rustc_no_writable,
    sym::rustc_non_const_trait_method,
    sym::rustc_nonnull_optimization_guaranteed,
    sym::rustc_nounwind,
    sym::rustc_objc_class,
    sym::rustc_objc_selector,
    sym::rustc_offload_kernel,
    sym::rustc_on_unimplemented,
    sym::rustc_panics_when_zero,
    sym::rustc_paren_sugar,
    sym::rustc_partition_codegened,
    sym::rustc_partition_reused,
    sym::rustc_pass_by_value,
    sym::rustc_pass_indirectly_in_non_rustic_abis,
    sym::rustc_preserve_ub_checks,
    sym::rustc_proc_macro_decls,
    sym::rustc_promotable,
    sym::rustc_pub_transparent,
    sym::rustc_reallocator,
    sym::rustc_regions,
    sym::rustc_scalable_vector,
    sym::rustc_show_spans,
    sym::rustc_simd_monomorphize_lane_limit,
    sym::rustc_skip_during_method_dispatch,
    sym::rustc_specialization_trait,
    sym::rustc_splat,
    sym::rustc_std_internal_symbol,
    sym::rustc_strict_coherence,
    sym::rustc_test_marker,
    sym::rustc_then_this_would_need,
    sym::rustc_trivial_field_reads,
    sym::rustc_unchecked_type_alias,
    sym::rustc_unroll,
    sym::sanitize,
    sym::should_panic,
    sym::stable,
    sym::target_feature,
    sym::test_runner,
    sym::thread_local,
    sym::track_caller,
    sym::type_length_limit,
    sym::unstable,
    sym::unstable_feature_bound,
    sym::unstable_removed,
    sym::used,
    sym::warn,
    sym::windows_subsystem,
    // tidy-alphabetical-end
];

pub fn is_builtin_attr_name(name: Symbol) -> bool {
    BUILTIN_ATTRIBUTE_SET.contains(&name)
}

pub static BUILTIN_ATTRIBUTE_SET: LazyLock<FxHashSet<Symbol>> = LazyLock::new(|| {
    let mut set = FxHashSet::default();
    for attr in BUILTIN_ATTRIBUTES.iter() {
        if !set.insert(*attr) {
            panic!("duplicate builtin attribute `{}`", attr);
        }
    }
    set
});
