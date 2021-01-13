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
                    description: concat!($($doc,)*),
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
                    description: concat!($($doc,)*),
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

    (removed, import_shadowing, "1.0.0", None, None, None),
    (removed, managed_boxes, "1.0.0", None, None, None),
    /// Allows use of unary negate on unsigned integers, e.g., -e for e: u8
    (removed, negate_unsigned, "1.0.0", Some(29645), None, None),
    (removed, reflect, "1.0.0", Some(27749), None, None),
    /// A way to temporarily opt out of opt in copy. This will *never* be accepted.
    (removed, opt_out_copy, "1.0.0", None, None, None),
    (removed, quad_precision_float, "1.0.0", None, None, None),
    (removed, struct_inherit, "1.0.0", None, None, None),
    (removed, test_removed_feature, "1.0.0", None, None, None),
    (removed, visible_private_types, "1.0.0", None, None, None),
    (removed, unsafe_no_drop_flag, "1.0.0", None, None, None),
    /// Allows using items which are missing stability attributes
    (removed, unmarked_api, "1.0.0", None, None, None),
    (removed, allocator, "1.0.0", None, None, None),
    (removed, simd, "1.0.0", Some(27731), None,
     Some("removed in favor of `#[repr(simd)]`")),
    (removed, advanced_slice_patterns, "1.0.0", Some(62254), None,
     Some("merged into `#![feature(slice_patterns)]`")),
    (removed, macro_reexport, "1.0.0", Some(29638), None,
     Some("subsumed by `pub use`")),
    /// Allows using custom attributes (RFC 572).
    (removed, custom_attribute, "1.0.0", Some(29642), None,
     Some("removed in favor of `#![register_tool]` and `#![register_attr]`")),
    /// Allows features specific to OIBIT (now called auto traits).
    /// Renamed to `auto_traits`.
    (removed, optin_builtin_traits, "1.0.0", Some(13231), None,
     Some("renamed to `auto_traits`")),
    (removed, pushpop_unsafe, "1.2.0", None, None, None),
    (removed, needs_allocator, "1.4.0", Some(27389), None,
     Some("subsumed by `#![feature(allocator_internals)]`")),
    /// Allows identifying crates that contain sanitizer runtimes.
    (removed, sanitizer_runtime, "1.17.0", None, None, None),
    (removed, proc_macro_mod, "1.27.0", Some(54727), None,
     Some("subsumed by `#![feature(proc_macro_hygiene)]`")),
    (removed, proc_macro_expr, "1.27.0", Some(54727), None,
     Some("subsumed by `#![feature(proc_macro_hygiene)]`")),
    (removed, proc_macro_non_items, "1.27.0", Some(54727), None,
     Some("subsumed by `#![feature(proc_macro_hygiene)]`")),
    (removed, proc_macro_gen, "1.27.0", Some(54727), None,
     Some("subsumed by `#![feature(proc_macro_hygiene)]`")),
    (removed, panic_implementation, "1.28.0", Some(44489), None,
     Some("subsumed by `#[panic_handler]`")),
    /// Allows the use of `#[derive(Anything)]` as sugar for `#[derive_Anything]`.
    (removed, custom_derive, "1.32.0", Some(29644), None,
     Some("subsumed by `#[proc_macro_derive]`")),
    /// Paths of the form: `extern::foo::bar`
    (removed, extern_in_paths, "1.33.0", Some(55600), None,
     Some("subsumed by `::foo::bar` paths")),
    (removed, quote, "1.33.0", Some(29601), None, None),
    /// Allows using `#[unsafe_destructor_blind_to_params]` (RFC 1238).
    (removed, dropck_parametricity, "1.38.0", Some(28498), None, None),
    (removed, await_macro, "1.38.0", Some(50547), None,
     Some("subsumed by `.await` syntax")),
    /// Allows defining `existential type`s.
    (removed, existential_type, "1.38.0", Some(63063), None,
     Some("removed in favor of `#![feature(type_alias_impl_trait)]`")),
    /// Allows using the macros:
    /// + `__diagnostic_used`
    /// + `__register_diagnostic`
    /// +`__build_diagnostic_array`
    (removed, rustc_diagnostic_macros, "1.38.0", None, None, None),
    /// Allows using `#[on_unimplemented(..)]` on traits.
    /// (Moved to `rustc_attrs`.)
    (removed, on_unimplemented, "1.40.0", None, None, None),
    /// Allows overlapping impls of marker traits.
    (removed, overlapping_marker_traits, "1.42.0", Some(29864), None,
     Some("removed in favor of `#![feature(marker_trait_attr)]`")),
    /// Allows `#[no_debug]`.
    (removed, no_debug, "1.43.0", Some(29721), None, Some("removed due to lack of demand")),
    /// Allows comparing raw pointers during const eval.
    (removed, const_compare_raw_pointers, "1.46.0", Some(53020), None,
     Some("cannot be allowed in const eval in any meaningful way")),

    // -------------------------------------------------------------------------
    // feature-group-end: removed features
    // -------------------------------------------------------------------------
);

#[rustfmt::skip]
declare_features! (
    (stable_removed, no_stack_check, "1.0.0", None, None),
);
