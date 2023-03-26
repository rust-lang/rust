//! List of the removed feature gates.

use super::{to_nonzero, Feature, State};
use rustc_span::symbol::sym;

macro_rules! declare_features {
    ($(
        $(#[doc = $doc:tt])* (removed, $feature:ident, $ver:expr, $issue:expr, None, $reason:expr),
    )+) => {
        /// Represents unstable features which have since been removed (it was once Active)
        pub const REMOVED_FEATURES: &[Feature] = &[
            $(
                Feature {
                    state: State::Removed { reason: $reason },
                    name: sym::$feature,
                    since: $ver,
                    issue: to_nonzero($issue),
                    edition: None,
                }
            ),+
        ];
    };

    ($(
        $(#[doc = $doc:tt])* (stable_removed, $feature:ident, $ver:expr, $issue:expr, None),
    )+) => {
        /// Represents stable features which have since been removed (it was once Accepted)
        pub const STABLE_REMOVED_FEATURES: &[Feature] = &[
            $(
                Feature {
                    state: State::Stabilized { reason: None },
                    name: sym::$feature,
                    since: $ver,
                    issue: to_nonzero($issue),
                    edition: None,
                }
            ),+
        ];
    };
}

#[rustfmt::skip]
declare_features! (
    // -------------------------------------------------------------------------
    // feature-group-start: removed features
    // -------------------------------------------------------------------------

    (removed, advanced_slice_patterns, "1.0.0", Some(62254), None,
     Some("merged into `#![feature(slice_patterns)]`")),
    (removed, allocator, "1.0.0", None, None, None),
    /// Allows a test to fail without failing the whole suite.
    (removed, allow_fail, "1.19.0", Some(46488), None, Some("removed due to no clear use cases")),
    (removed, await_macro, "1.38.0", Some(50547), None,
     Some("subsumed by `.await` syntax")),
    /// Allows using the `box $expr` syntax.
    (removed, box_syntax, "CURRENT_RUSTC_VERSION", Some(49733), None, Some("replaced with `#[rustc_box]`")),
    /// Allows capturing disjoint fields in a closure/generator (RFC 2229).
    (removed, capture_disjoint_fields, "1.49.0", Some(53488), None, Some("stabilized in Rust 2021")),
    /// Allows comparing raw pointers during const eval.
    (removed, const_compare_raw_pointers, "1.46.0", Some(53020), None,
     Some("cannot be allowed in const eval in any meaningful way")),
    /// Allows non-trivial generic constants which have to be manually propagated upwards.
     (removed, const_evaluatable_checked, "1.48.0", Some(76560), None, Some("renamed to `generic_const_exprs`")),
    /// Allows the definition of `const` functions with some advanced features.
    (removed, const_fn, "1.54.0", Some(57563), None,
     Some("split into finer-grained feature gates")),
    /// Allows const generic types (e.g. `struct Foo<const N: usize>(...);`).
    (removed, const_generics, "1.34.0", Some(44580), None,
     Some("removed in favor of `#![feature(adt_const_params)]` and `#![feature(generic_const_exprs)]`")),
    /// Allows `[x; N]` where `x` is a constant (RFC 2203).
    (removed, const_in_array_repeat_expressions,  "1.37.0", Some(49147), None,
     Some("removed due to causing promotable bugs")),
    /// Allows casting raw pointers to `usize` during const eval.
    (removed, const_raw_ptr_to_usize_cast, "1.55.0", Some(51910), None,
     Some("at compile-time, pointers do not have an integer value, so these casts cannot be properly supported")),
    /// Allows `T: ?const Trait` syntax in bounds.
    (removed, const_trait_bound_opt_out, "1.42.0", Some(67794), None,
     Some("Removed in favor of `~const` bound in #![feature(const_trait_impl)]")),
    /// Allows using `crate` as visibility modifier, synonymous with `pub(crate)`.
    (removed, crate_visibility_modifier, "1.63.0", Some(53120), None, Some("removed in favor of `pub(crate)`")),
    /// Allows using custom attributes (RFC 572).
    (removed, custom_attribute, "1.0.0", Some(29642), None,
     Some("removed in favor of `#![register_tool]` and `#![register_attr]`")),
    /// Allows the use of `#[derive(Anything)]` as sugar for `#[derive_Anything]`.
    (removed, custom_derive, "1.32.0", Some(29644), None,
     Some("subsumed by `#[proc_macro_derive]`")),
    /// Allows using `#[doc(keyword = "...")]`.
    (removed, doc_keyword, "1.28.0", Some(51315), None,
     Some("merged into `#![feature(rustdoc_internals)]`")),
    /// Allows using `doc(primitive)` without a future-incompat warning.
    (removed, doc_primitive, "1.56.0", Some(88070), None,
     Some("merged into `#![feature(rustdoc_internals)]`")),
    /// Allows `#[doc(spotlight)]`.
    /// The attribute was renamed to `#[doc(notable_trait)]`
    /// and the feature to `doc_notable_trait`.
    (removed, doc_spotlight, "1.22.0", Some(45040), None,
     Some("renamed to `doc_notable_trait`")),
    /// Allows using `#[unsafe_destructor_blind_to_params]` (RFC 1238).
    (removed, dropck_parametricity, "1.38.0", Some(28498), None, None),
    /// Allows defining `existential type`s.
    (removed, existential_type, "1.38.0", Some(63063), None,
     Some("removed in favor of `#![feature(type_alias_impl_trait)]`")),
    /// Paths of the form: `extern::foo::bar`
    (removed, extern_in_paths, "1.33.0", Some(55600), None,
     Some("subsumed by `::foo::bar` paths")),
    /// Allows `#[doc(include = "some-file")]`.
    (removed, external_doc, "1.54.0", Some(44732), None,
     Some("use #[doc = include_str!(\"filename\")] instead, which handles macro invocations")),
    /// Allows `impl Trait` in bindings (`let`, `const`, `static`).
    (removed, impl_trait_in_bindings, "1.55.0", Some(63065), None,
     Some("the implementation was not maintainable, the feature may get reintroduced once the current refactorings are done")),
    (removed, import_shadowing, "1.0.0", None, None, None),
    /// Allows in-band quantification of lifetime bindings (e.g., `fn foo(x: &'a u8) -> &'a u8`).
    (removed, in_band_lifetimes, "1.23.0", Some(44524), None,
     Some("removed due to unsolved ergonomic questions and added lifetime resolution complexity")),
    /// Allows inferring `'static` outlives requirements (RFC 2093).
    (removed, infer_static_outlives_requirements, "1.63.0", Some(54185), None,
     Some("removed as it caused some confusion and discussion was inactive for years")),
    /// Lazily evaluate constants. This allows constants to depend on type parameters.
    (removed, lazy_normalization_consts, "1.46.0", Some(72219), None, Some("superseded by `generic_const_exprs`")),
    /// Allows using the `#[link_args]` attribute.
    (removed, link_args, "1.53.0", Some(29596), None,
     Some("removed in favor of using `-C link-arg=ARG` on command line, \
           which is available from cargo build scripts with `cargo:rustc-link-arg` now")),
    (removed, macro_reexport, "1.0.0", Some(29638), None,
     Some("subsumed by `pub use`")),
    /// Allows using `#[main]` to replace the entrypoint `#[lang = "start"]` calls.
    (removed, main, "1.53.0", Some(29634), None, None),
    (removed, managed_boxes, "1.0.0", None, None, None),
    /// Allows the use of type alias impl trait in function return positions
    (removed, min_type_alias_impl_trait, "1.56.0", Some(63063), None,
     Some("removed in favor of full type_alias_impl_trait")),
    (removed, needs_allocator, "1.4.0", Some(27389), None,
     Some("subsumed by `#![feature(allocator_internals)]`")),
    /// Allows use of unary negate on unsigned integers, e.g., -e for e: u8
    (removed, negate_unsigned, "1.0.0", Some(29645), None, None),
    /// Allows `#[no_debug]`.
    (removed, no_debug, "1.43.0", Some(29721), None, Some("removed due to lack of demand")),
    /// Allows using `#[on_unimplemented(..)]` on traits.
    /// (Moved to `rustc_attrs`.)
    (removed, on_unimplemented, "1.40.0", None, None, None),
    /// A way to temporarily opt out of opt in copy. This will *never* be accepted.
    (removed, opt_out_copy, "1.0.0", None, None, None),
    /// Allows features specific to OIBIT (now called auto traits).
    /// Renamed to `auto_traits`.
    (removed, optin_builtin_traits, "1.0.0", Some(13231), None,
     Some("renamed to `auto_traits`")),
    /// Allows overlapping impls of marker traits.
    (removed, overlapping_marker_traits, "1.42.0", Some(29864), None,
     Some("removed in favor of `#![feature(marker_trait_attr)]`")),
    (removed, panic_implementation, "1.28.0", Some(44489), None,
     Some("subsumed by `#[panic_handler]`")),
    /// Allows using `#[plugin_registrar]` on functions.
    (removed, plugin_registrar, "1.54.0", Some(29597), None,
     Some("a __rustc_plugin_registrar symbol must now be defined instead")),
    (removed, proc_macro_expr, "1.27.0", Some(54727), None,
     Some("subsumed by `#![feature(proc_macro_hygiene)]`")),
    (removed, proc_macro_gen, "1.27.0", Some(54727), None,
     Some("subsumed by `#![feature(proc_macro_hygiene)]`")),
    (removed, proc_macro_mod, "1.27.0", Some(54727), None,
     Some("subsumed by `#![feature(proc_macro_hygiene)]`")),
    (removed, proc_macro_non_items, "1.27.0", Some(54727), None,
     Some("subsumed by `#![feature(proc_macro_hygiene)]`")),
    (removed, pub_macro_rules, "1.53.0", Some(78855), None,
     Some("removed due to being incomplete, in particular it does not work across crates")),
    (removed, pushpop_unsafe, "1.2.0", None, None, None),
    (removed, quad_precision_float, "1.0.0", None, None, None),
    (removed, quote, "1.33.0", Some(29601), None, None),
    (removed, reflect, "1.0.0", Some(27749), None, None),
    /// Allows using the `#[register_attr]` attribute.
    (removed, register_attr, "1.65.0", Some(66080), None,
     Some("removed in favor of `#![register_tool]`")),
    /// Allows using the macros:
    /// + `__diagnostic_used`
    /// + `__register_diagnostic`
    /// +`__build_diagnostic_array`
    (removed, rustc_diagnostic_macros, "1.38.0", None, None, None),
    /// Allows identifying crates that contain sanitizer runtimes.
    (removed, sanitizer_runtime, "1.17.0", None, None, None),
    (removed, simd, "1.0.0", Some(27731), None,
     Some("removed in favor of `#[repr(simd)]`")),
    /// Allows `#[link(kind = "static-nobundle", ...)]`.
    (removed, static_nobundle, "1.16.0", Some(37403), None,
     Some(r#"subsumed by `#[link(kind = "static", modifiers = "-bundle", ...)]`"#)),
    (removed, struct_inherit, "1.0.0", None, None, None),
    (removed, test_removed_feature, "1.0.0", None, None, None),
    /// Allows using items which are missing stability attributes
    (removed, unmarked_api, "1.0.0", None, None, None),
    (removed, unsafe_no_drop_flag, "1.0.0", None, None, None),
    /// Allows `union` fields that don't implement `Copy` as long as they don't have any drop glue.
    (removed, untagged_unions, "1.13.0", Some(55149), None,
     Some("unions with `Copy` and `ManuallyDrop` fields are stable; there is no intent to stabilize more")),
    /// Allows `#[unwind(..)]`.
    ///
    /// Permits specifying whether a function should permit unwinding or abort on unwind.
    (removed, unwind_attributes, "1.56.0", Some(58760), None, Some("use the C-unwind ABI instead")),
    (removed, visible_private_types, "1.0.0", None, None, None),
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!
    // Features are listed in alphabetical order. Tidy will fail if you don't keep it this way.
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!

    // -------------------------------------------------------------------------
    // feature-group-end: removed features
    // -------------------------------------------------------------------------
);

#[rustfmt::skip]
declare_features! (
    (stable_removed, no_stack_check, "1.0.0", None, None),
);
