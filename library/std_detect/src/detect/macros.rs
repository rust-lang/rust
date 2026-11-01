/// Forward a literal to the relevant `__is_feature_detected` function.
#[macro_export]
#[allow_internal_unstable(stdarch_internal)]
#[unstable(feature = "stdarch_internal", issue = "none")]
macro_rules! detect_feature {
    ($feature:tt, $feature_lit:tt) => {
        $crate::detect_feature!($feature, $feature_lit : $feature_lit)
    };
    ($feature:tt, $feature_lit:tt : $($target_feature_lit:tt),*) => {
        // Emit `true` if enabled at compile time
        $(cfg!(target_feature = $target_feature_lit) ||)*
            $crate::detect::__is_feature_detected::$feature()
    };
    ($feature:tt, $feature_lit:tt, without cfg check: true) => {
        $crate::detect::__is_feature_detected::$feature()
    };
}

/// Verify that all features are known by `cfg!`.
#[allow(unused_macros, reason = "it's used in the features! macro below")]
macro_rules! check_cfg_feature {
    ($feature:tt, $feature_lit:tt) => {
        check_cfg_feature!($feature, $feature_lit : $feature_lit)
    };
    ($feature:tt, $feature_lit:tt : $($target_feature_lit:tt),*) => {
        $(cfg!(target_feature = $target_feature_lit);)*
    };
    ($feature:tt, $feature_lit:tt, without cfg check: $feature_cfg_check:literal) => {
        #[allow(unexpected_cfgs, reason = $feature_lit)]
        { cfg!(target_feature = $feature_lit) }
    };
}

/// Create the `is_*_feature_detected` macro for an architecture.
// Make sure any additions to this macro are tested in
// `tests/ui/std/is_arch_feature_detected.AARCH64.stderr`.
#[allow(unused)]
macro_rules! features {
    (
        // Target or target family name, for use in diagnostic messages
        @TARGET: $target:ident;
        // Cfg for which this macro should be allowed
        @CFG: $cfg:meta;
        // Name like `is_*_feature_detected`
        @MACRO_NAME: $macro_name:ident;
        // Docs and stability attributes
        @MACRO_ATTRS: $(#[$macro_attrs:meta])*
        // Synonyms, with possible deprecation
        $(@BIND_FEATURE_NAME: $bind_feature:tt; $feature_impl:tt; $(#[$deprecate_attr:meta];)?)*
        // Features that exist but can't be detected at runtime, so emit an error.
        $(@NO_RUNTIME_DETECTION: $nort_feature:tt; )*
        // List of features
        $(@FEATURE: #[$stability_attr:meta] $feature:ident: $feature_lit:tt;
            // Skip verifying that the feature is known by `cfg!`, which also skips emitting `true`
            // if the feature is enabled at compile time.
            $(without cfg check: $feature_cfg_check:tt;)?
            // If this feature is always enabled by some other feature, we can avoid the cost of
            // checking them separately by instead only checking the root features.
            $(implied by target_features: [$($target_feature_lit:tt),*];)?
            // Doc comment for the feature.
            $(#[$feature_comment:meta])*)*
    ) => {
        // The current arch's version actually does the feature detection.
        #[macro_export]
        $(#[$macro_attrs])*
        #[allow_internal_unstable(stdarch_internal)]
        #[cfg($cfg)]
        #[doc(cfg($cfg))]
        macro_rules! $macro_name {
            // Emit all known features
            $(
                ($feature_lit) => {
                    $crate::detect_feature!(
                        $feature,
                        $feature_lit
                        $(, without cfg check: $feature_cfg_check)?
                        $(: $($target_feature_lit),*)?
                    )
                };
            )*
            // Emit all aliases, with a deprecation notice if applicable
            $(
                ($bind_feature) => {
                    {
                        $(
                            #[$deprecate_attr] macro_rules! deprecated_feature { {} => {}; }
                            deprecated_feature! {};
                        )?
                        $crate::$macro_name!($feature_impl)
                    }
                };
            )*
            // Emit an error for features that can't be detected at runtime
            $(
                ($nort_feature) => {
                    compile_error!(
                        concat!(
                            stringify!($nort_feature),
                            " feature cannot be detected at run-time"
                        )
                    )
                };
            )*
            // Allow a trailing comma
            ($t:tt,) => {
                $crate::$macro_name!($t)
            };
            // Fallback, error for unknown target features
            ($t:tt) => {
                compile_error!(
                    concat!("unknown ", stringify!($target), " target feature: ", $t)
                )
            };
        }

        // All other arches still validate the features, but also
        $(#[$macro_attrs])*
        #[macro_export]
        #[cfg(not($cfg))]
        #[doc(cfg($cfg))]
        macro_rules! $macro_name {
            // All known features get an error
            $(
                ($feature_lit) => {
                    compile_error!(
                        concat!(
                            "This macro cannot be used on the current target. You can prevent\n\
                            it from being used in other architectures by guarding it behind\n\
                            a `cfg(", stringify!($cfg), ")`."
                        )
                    )
                };
            )*
            // Forward deprecated features
            $(
                ($bind_feature) => { $crate::$macro_name!($feature_impl) };
            )*
            // Emit an error for features that can't be detected at runtime
            $(
                ($nort_feature) => {
                    compile_error!(
                        concat!(
                            stringify!($nort_feature),
                            " feature cannot be detected at run-time"
                        )
                    )
                };
            )*
            // Allow a trailing comma
            ($t:tt,) => {
                    $crate::$macro_name!($t);
            };
            // Fallback, error for unknown target features
            ($t:tt) => {
                compile_error!(
                    concat!("unknown ", stringify!($target), " target feature: ", $t)
                )
            };
        }

        #[deny(unexpected_cfgs)]
        #[deny(unfulfilled_lint_expectations)]
        const _: () = {
            $(
                check_cfg_feature!(
                    $feature,
                    $feature_lit
                    $(, without cfg check: $feature_cfg_check)?
                    $(: $($target_feature_lit),*)?)
                ;
            )*
        };

        /// Each variant denotes a position in a bitset for a particular feature.
        ///
        /// PLEASE: do not use this, it is an implementation detail subject
        /// to change.
        #[doc(hidden)]
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone)]
        #[repr(u8)]
        #[unstable(feature = "stdarch_internal", issue = "none")]
        #[cfg($cfg)]
        pub(crate) enum Feature {
            $(
                $(#[$feature_comment])*
                $feature,
            )*
        }

        #[cfg($cfg)]
        impl Feature {
            pub(crate) const ALL: &[Feature] = &[
                $(Feature::$feature,)*
            ];

            pub(crate) fn to_str(self) -> &'static str {
                match self {
                    $(Feature::$feature => $feature_lit,)*
                }
            }

            #[cfg(feature = "std_detect_env_override")]
            pub(crate) fn from_str(s: &str) -> Result<Feature, ()> {
                match s {
                    $($feature_lit => Ok(Feature::$feature),)*
                    _ => Err(())
                }
            }
        }

        /// Each function performs run-time feature detection for a single
        /// feature. This allow us to use stability attributes on a per feature
        /// basis.
        ///
        /// PLEASE: do not use this, it is an implementation detail subject
        /// to change.
        #[doc(hidden)]
        #[cfg($cfg)]
        pub mod __is_feature_detected {
            $(

                /// PLEASE: do not use this, it is an implementation detail
                /// subject to change.
                #[inline]
                #[doc(hidden)]
                #[$stability_attr]
                pub fn $feature() -> bool {
                    $crate::detect::check_for($crate::detect::Feature::$feature)
                }
            )*
        }
    };
}
