//! List of the renamed feature gates.

use std::num::NonZero;

use rustc_span::sym;

use super::{Feature, to_nonzero};
use crate::opt_nonzero_u32;

pub struct RenamedFeature {
    pub feature: Feature,
    pub new_name: &'static str,
    pub pull: Option<NonZero<u32>>,
}

macro_rules! declare_features {
    ($(
        $(#[doc = $doc:tt])* (renamed, $old_feature_name:ident => $new_feature_name:ident, $ver:expr, $issue:expr $(, $pull:expr)?),
    )+) => {
        /// Features that have been renamed.
        pub static RENAMED_LANG_FEATURES: &[RenamedFeature] = &[
            $(RenamedFeature {
                feature: Feature {
                    name: sym::$old_feature_name,
                    since: $ver,
                    issue: to_nonzero($issue),
                },
                new_name: stringify!($new_feature_name),
                pull:  opt_nonzero_u32!($($pull)?),
            }),+
        ];
    };
}

#[rustfmt::skip]
declare_features! {
    // -------------------------------------------------------------------------
    // feature-group-start: renamed features
    // -------------------------------------------------------------------------

    // Note that the version indicates when it got *renamed*.
    //
    // When renaming a feature, set the version number to
    // `CURRENT RUSTC VERSION` with ` ` replaced by `_`.

    (renamed, abi_c_cmse_nonsecure_call => abi_cmse_nonsecure_call, "1.90.0", Some(81391), 142146),
    /// Allows non-trivial generic constants which have to be manually propagated upwards.
    (renamed, const_evaluatable_checked => generic_const_exprs, "1.56.0", Some(76560), 88369),
    /// Allows `#[doc(spotlight)]`.
    /// The attribute was renamed to `#[doc(notable_trait)]`
    /// and the feature to `doc_notable_trait`.
    (renamed, doc_spotlight => doc_notable_trait, "1.53.0", Some(45040), 80965),
    /// Allows generators to be cloned.
    (renamed, generator_clone => coroutine_clone, "1.75.0", Some(95360), 116958),
    /// Allows defining generators.
    (renamed, generators => coroutines, "1.75.0", Some(43122), 116958),
    /// Allows `#[no_coverage]` on functions.
    /// The feature was renamed to `coverage_attribute` and the attribute to `#[coverage(on|off)]`
    (renamed, no_coverage => coverage_attribute, "1.74.0", Some(84605), 114656),
    // Allows the use of `no_sanitize` attribute.
    /// The feature was renamed to `sanitize` and the attribute to `#[sanitize(xyz = "on|off")]`
    (renamed, no_sanitize => sanitize, "1.91.0", Some(39699), 142681),
    /// Allows making `dyn Trait` well-formed even if `Trait` is not dyn compatible (object safe).
    /// Renamed to `dyn_compatible_for_dispatch`.
    (renamed, object_safe_for_dispatch => dyn_compatible_for_dispatch, "1.83.0", Some(43561), 131511),
    /// Allows features specific to OIBIT (now called auto traits).
    /// Renamed to `auto_traits`.
    (renamed, optin_builtin_traits => auto_traits, "1.50.0", Some(13231), 79336),

    // -------------------------------------------------------------------------
    // feature-group-end: renamed features
    // -------------------------------------------------------------------------


    // -------------------------------------------------------------------------
    // feature-group-start: renamed library features
    // -------------------------------------------------------------------------
    //
    // FIXME(#141617): we should have a better way to track renamed library features, but we reuse
    // the infrastructure here so users still get hints. The symbols used here can be remove from
    // `symbol.rs` when that happens.

    // -------------------------------------------------------------------------
    // feature-group-end: renamed library features
    // -------------------------------------------------------------------------
 }
