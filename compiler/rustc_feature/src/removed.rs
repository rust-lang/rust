//! List of the removed feature gates.

use rustc_span::sym;

use super::{Feature, to_nonzero};

pub struct RemovedFeature {
    pub feature: Feature,
    pub reason: Option<&'static str>,
}

macro_rules! declare_features {
    ($(
        $(#[doc = $doc:tt])* (removed, $feature:ident, $ver:expr, $issue:expr, $reason:expr),
    )+) => {
        /// Formerly unstable features that have now been removed.
        pub static REMOVED_LANG_FEATURES: &[RemovedFeature] = &[
            $(RemovedFeature {
                feature: Feature {
                    name: sym::$feature,
                    since: $ver,
                    issue: to_nonzero($issue),
                },
                reason: $reason
            }),+
        ];
    };
}

#[rustfmt::skip]
declare_features! (
    // -------------------------------------------------------------------------
    // feature-group-start: removed features
    // -------------------------------------------------------------------------

    // Note that the version indicates when it got *removed*.
    // When moving an unstable feature here, set the version number to
    // `CURRENT RUSTC VERSION` with ` ` replaced by `_`.
    // (But not all features below do this properly; many indicate the
    // version they got originally added in.)

    /// Allows using the `amdgpu-kernel` ABI.
    (removed, abi_amdgpu_kernel, "1.77.0", Some(51575), None),
    (removed, advanced_slice_patterns, "1.0.0", Some(62254),
     Some("merged into `#![feature(slice_patterns)]`")),
    (removed, allocator, "1.0.0", None, None),
    /// Allows a test to fail without failing the whole suite.
    (removed, allow_fail, "1.19.0", Some(46488), Some("removed due to no clear use cases")),
    (removed, await_macro, "1.38.0", Some(50547),
     Some("subsumed by `.await` syntax")),
    /// Allows using the `box $expr` syntax.
    (removed, box_syntax, "1.70.0", Some(49733), Some("replaced with `#[rustc_box]`")),
    /// Allows capturing disjoint fields in a closure/coroutine (RFC 2229).
    (removed, capture_disjoint_fields, "1.49.0", Some(53488), Some("stabilized in Rust 2021")),
    /// Allows comparing raw pointers during const eval.
    (removed, const_compare_raw_pointers, "1.46.0", Some(53020),
     Some("cannot be allowed in const eval in any meaningful way")),
    /// Allows limiting the evaluation steps of const expressions
    (removed, const_eval_limit, "1.43.0", Some(67217), Some("removed the limit entirely")),
    /// Allows non-trivial generic constants which have to be manually propagated upwards.
    (removed, const_evaluatable_checked, "1.48.0", Some(76560), Some("renamed to `generic_const_exprs`")),
    /// Allows the definition of `const` functions with some advanced features.
    (removed, const_fn, "1.54.0", Some(57563),
     Some("split into finer-grained feature gates")),
    /// Allows const generic types (e.g. `struct Foo<const N: usize>(...);`).
    (removed, const_generics, "1.34.0", Some(44580),
     Some("removed in favor of `#![feature(adt_const_params)]` and `#![feature(generic_const_exprs)]`")),
    /// Allows `[x; N]` where `x` is a constant (RFC 2203).
    (removed, const_in_array_repeat_expressions,  "1.37.0", Some(49147),
     Some("removed due to causing promotable bugs")),
    /// Allows casting raw pointers to `usize` during const eval.
    (removed, const_raw_ptr_to_usize_cast, "1.55.0", Some(51910),
     Some("at compile-time, pointers do not have an integer value, so these casts cannot be properly supported")),
    /// Allows `T: ?const Trait` syntax in bounds.
    (removed, const_trait_bound_opt_out, "1.42.0", Some(67794),
     Some("Removed in favor of `~const` bound in #![feature(const_trait_impl)]")),
    /// Allows using `crate` as visibility modifier, synonymous with `pub(crate)`.
    (removed, crate_visibility_modifier, "1.63.0", Some(53120), Some("removed in favor of `pub(crate)`")),
    /// Allows using custom attributes (RFC 572).
    (removed, custom_attribute, "1.0.0", Some(29642),
     Some("removed in favor of `#![register_tool]` and `#![register_attr]`")),
    /// Allows the use of `#[derive(Anything)]` as sugar for `#[derive_Anything]`.
    (removed, custom_derive, "1.32.0", Some(29644),
     Some("subsumed by `#[proc_macro_derive]`")),
    /// Allows default type parameters to influence type inference.
    (removed, default_type_parameter_fallback, "1.82.0", Some(27336),
     Some("never properly implemented; requires significant design work")),
    /// Allows deriving traits as per `SmartPointer` specification
    (removed, derive_smart_pointer, "1.79.0", Some(123430), Some("replaced by `CoercePointee`")),
    /// Allows using `#[doc(keyword = "...")]`.
    (removed, doc_keyword, "1.28.0", Some(51315),
     Some("merged into `#![feature(rustdoc_internals)]`")),
    /// Allows using `doc(primitive)` without a future-incompat warning.
    (removed, doc_primitive, "1.56.0", Some(88070),
     Some("merged into `#![feature(rustdoc_internals)]`")),
    /// Allows `#[doc(spotlight)]`.
    /// The attribute was renamed to `#[doc(notable_trait)]`
    /// and the feature to `doc_notable_trait`.
    (removed, doc_spotlight, "1.22.0", Some(45040),
     Some("renamed to `doc_notable_trait`")),
    /// Allows using `#[unsafe_destructor_blind_to_params]` (RFC 1238).
    (removed, dropck_parametricity, "1.38.0", Some(28498), None),
    /// Allows making `dyn Trait` well-formed even if `Trait` is not dyn compatible[^1].
    /// In that case, `dyn Trait: Trait` does not hold. Moreover, coercions and
    /// casts in safe Rust to `dyn Trait` for such a `Trait` is also forbidden.
    ///
    /// Renamed from `object_safe_for_dispatch`.
    ///
    /// [^1]: Formerly known as "object safe".
    (removed, dyn_compatible_for_dispatch, "1.83.0", Some(43561),
     Some("removed, not used heavily and represented additional complexity in dyn compatibility")),
    /// Uses generic effect parameters for ~const bounds
    (removed, effects, "1.84.0", Some(102090),
     Some("removed, redundant with `#![feature(const_trait_impl)]`")),
    /// Allows defining `existential type`s.
    (removed, existential_type, "1.38.0", Some(63063),
     Some("removed in favor of `#![feature(type_alias_impl_trait)]`")),
    /// Paths of the form: `extern::foo::bar`
    (removed, extern_in_paths, "1.33.0", Some(55600),
     Some("subsumed by `::foo::bar` paths")),
    /// Allows `#[doc(include = "some-file")]`.
    (removed, external_doc, "1.54.0", Some(44732),
     Some("use #[doc = include_str!(\"filename\")] instead, which handles macro invocations")),
    /// Allows using `#[ffi_returns_twice]` on foreign functions.
    (removed, ffi_returns_twice, "1.78.0", Some(58314),
     Some("being investigated by the ffi-unwind project group")),
    /// Allows generators to be cloned.
    (removed, generator_clone, "1.65.0", Some(95360), Some("renamed to `coroutine_clone`")),
    /// Allows defining generators.
    (removed, generators, "1.21.0", Some(43122), Some("renamed to `coroutines`")),
    /// An extension to the `generic_associated_types` feature, allowing incomplete features.
    (removed, generic_associated_types_extended, "1.85.0", Some(95451),
        Some(
            "feature needs overhaul and reimplementation pending \
            better implied higher-ranked implied bounds support"
        )
    ),
    (removed, import_shadowing, "1.0.0", None, None),
    /// Allows in-band quantification of lifetime bindings (e.g., `fn foo(x: &'a u8) -> &'a u8`).
    (removed, in_band_lifetimes, "1.23.0", Some(44524),
     Some("removed due to unsolved ergonomic questions and added lifetime resolution complexity")),
    /// Allows inferring `'static` outlives requirements (RFC 2093).
    (removed, infer_static_outlives_requirements, "1.63.0", Some(54185),
     Some("removed as it caused some confusion and discussion was inactive for years")),
    /// Allow anonymous constants from an inline `const` block in pattern position
    (removed, inline_const_pat, "1.88.0", Some(76001),
     Some("removed due to implementation concerns as it requires significant refactorings")),
    /// Lazily evaluate constants. This allows constants to depend on type parameters.
    (removed, lazy_normalization_consts, "1.46.0", Some(72219), Some("superseded by `generic_const_exprs`")),
    /// Changes `impl Trait` to capture all lifetimes in scope.
    (removed, lifetime_capture_rules_2024, "1.76.0", None, Some("unnecessary -- use edition 2024 instead")),
    /// Allows using the `#[link_args]` attribute.
    (removed, link_args, "1.53.0", Some(29596),
     Some("removed in favor of using `-C link-arg=ARG` on command line, \
           which is available from cargo build scripts with `cargo:rustc-link-arg` now")),
    (removed, macro_reexport, "1.0.0", Some(29638),
     Some("subsumed by `pub use`")),
    /// Allows using `#[main]` to replace the entrypoint `#[lang = "start"]` calls.
    (removed, main, "1.53.0", Some(29634), None),
    (removed, managed_boxes, "1.0.0", None, None),
    /// Allows the use of type alias impl trait in function return positions
    (removed, min_type_alias_impl_trait, "1.56.0", Some(63063),
     Some("removed in favor of full type_alias_impl_trait")),
    /// Make `mut` not reset the binding mode on edition >= 2024.
    (removed, mut_preserve_binding_mode_2024, "1.79.0", Some(123076), Some("superseded by `ref_pat_eat_one_layer_2024`")),
    (removed, needs_allocator, "1.4.0", Some(27389),
     Some("subsumed by `#![feature(allocator_internals)]`")),
    /// Allows use of unary negate on unsigned integers, e.g., -e for e: u8
    (removed, negate_unsigned, "1.0.0", Some(29645), None),
    /// Allows `#[no_coverage]` on functions.
    /// The feature was renamed to `coverage_attribute` and the attribute to `#[coverage(on|off)]`
    (removed, no_coverage, "1.74.0", Some(84605), Some("renamed to `coverage_attribute`")),
    /// Allows `#[no_debug]`.
    (removed, no_debug, "1.43.0", Some(29721), Some("removed due to lack of demand")),
    /// Note: this feature was previously recorded in a separate
    /// `STABLE_REMOVED` list because it, uniquely, was once stable but was
    /// then removed. But there was no utility storing it separately, so now
    /// it's in this list.
    (removed, no_stack_check, "1.0.0", None, None),
    /// Allows making `dyn Trait` well-formed even if `Trait` is not dyn compatible (object safe).
    /// Renamed to `dyn_compatible_for_dispatch`.
    (removed, object_safe_for_dispatch, "1.83.0", Some(43561),
     Some("renamed to `dyn_compatible_for_dispatch`")),
    /// Allows using `#[on_unimplemented(..)]` on traits.
    /// (Moved to `rustc_attrs`.)
    (removed, on_unimplemented, "1.40.0", None, None),
    /// A way to temporarily opt out of opt-in copy. This will *never* be accepted.
    (removed, opt_out_copy, "1.0.0", None, None),
    /// Allows features specific to OIBIT (now called auto traits).
    /// Renamed to `auto_traits`.
    (removed, optin_builtin_traits, "1.0.0", Some(13231),
     Some("renamed to `auto_traits`")),
    /// Allows overlapping impls of marker traits.
    (removed, overlapping_marker_traits, "1.42.0", Some(29864),
     Some("removed in favor of `#![feature(marker_trait_attr)]`")),
    (removed, panic_implementation, "1.28.0", Some(44489),
     Some("subsumed by `#[panic_handler]`")),
    /// Allows `extern "platform-intrinsic" { ... }`.
    (removed, platform_intrinsics, "1.4.0", Some(27731),
     Some("SIMD intrinsics use the regular intrinsics ABI now")),
    /// Allows using `#![plugin(myplugin)]`.
    (removed, plugin, "1.75.0", Some(29597),
     Some("plugins are no longer supported")),
    /// Allows using `#[plugin_registrar]` on functions.
    (removed, plugin_registrar, "1.54.0", Some(29597),
     Some("plugins are no longer supported")),
    /// Allows exhaustive integer pattern matching with `usize::MAX`/`isize::MIN`/`isize::MAX`.
    (removed, precise_pointer_size_matching, "1.32.0", Some(56354),
     Some("removed in favor of half-open ranges")),
    (removed, pref_align_of, "CURRENT_RUSTC_VERSION", Some(91971),
     Some("removed due to marginal use and inducing compiler complications")),
    (removed, proc_macro_expr, "1.27.0", Some(54727),
     Some("subsumed by `#![feature(proc_macro_hygiene)]`")),
    (removed, proc_macro_gen, "1.27.0", Some(54727),
     Some("subsumed by `#![feature(proc_macro_hygiene)]`")),
    (removed, proc_macro_mod, "1.27.0", Some(54727),
     Some("subsumed by `#![feature(proc_macro_hygiene)]`")),
    (removed, proc_macro_non_items, "1.27.0", Some(54727),
     Some("subsumed by `#![feature(proc_macro_hygiene)]`")),
    (removed, pub_macro_rules, "1.53.0", Some(78855),
     Some("removed due to being incomplete, in particular it does not work across crates")),
    (removed, pushpop_unsafe, "1.2.0", None, None),
    (removed, quad_precision_float, "1.0.0", None, None),
    (removed, quote, "1.33.0", Some(29601), None),
    (removed, ref_pat_everywhere, "1.79.0", Some(123076), Some("superseded by `ref_pat_eat_one_layer_2024")),
    (removed, reflect, "1.0.0", Some(27749), None),
    /// Allows using the `#[register_attr]` attribute.
    (removed, register_attr, "1.65.0", Some(66080),
     Some("removed in favor of `#![register_tool]`")),
    (removed, rust_2018_preview, "1.76.0", None,
     Some("2018 Edition preview is no longer relevant")),
    /// Allows using the macros:
    /// + `__diagnostic_used`
    /// + `__register_diagnostic`
    /// +`__build_diagnostic_array`
    (removed, rustc_diagnostic_macros, "1.38.0", None, None),
    /// Allows identifying crates that contain sanitizer runtimes.
    (removed, sanitizer_runtime, "1.17.0", None, None),
    (removed, simd, "1.0.0", Some(27731), Some("removed in favor of `#[repr(simd)]`")),
    /// Allows using `#[start]` on a function indicating that it is the program entrypoint.
    (removed, start, "1.0.0", Some(29633), Some("not portable enough and never RFC'd")),
    /// Allows `#[link(kind = "static-nobundle", ...)]`.
    (removed, static_nobundle, "1.16.0", Some(37403),
     Some(r#"subsumed by `#[link(kind = "static", modifiers = "-bundle", ...)]`"#)),
    (removed, struct_inherit, "1.0.0", None, None),
    (removed, test_removed_feature, "1.0.0", None, None),
    /// Allows using items which are missing stability attributes
    (removed, unmarked_api, "1.0.0", None, None),
    /// Allows unnamed fields of struct and union type
    (removed, unnamed_fields, "1.83.0", Some(49804), Some("feature needs redesign")),
    (removed, unsafe_no_drop_flag, "1.0.0", None, None),
    (removed, unsized_tuple_coercion, "1.87.0", Some(42877),
     Some("The feature restricts possible layouts for tuples, and this restriction is not worth it.")),
    /// Allows `union` fields that don't implement `Copy` as long as they don't have any drop glue.
    (removed, untagged_unions, "1.13.0", Some(55149),
     Some("unions with `Copy` and `ManuallyDrop` fields are stable; there is no intent to stabilize more")),
    /// Allows `#[unwind(..)]`.
    ///
    /// Permits specifying whether a function should permit unwinding or abort on unwind.
    (removed, unwind_attributes, "1.56.0", Some(58760), Some("use the C-unwind ABI instead")),
    (removed, visible_private_types, "1.0.0", None, None),
    /// Allows `extern "wasm" fn`
    (removed, wasm_abi, "1.81.0", Some(83788),
     Some("non-standard wasm ABI is no longer supported")),
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!
    // Features are listed in alphabetical order. Tidy will fail if you don't keep it this way.
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!

    // -------------------------------------------------------------------------
    // feature-group-end: removed features
    // -------------------------------------------------------------------------
);
