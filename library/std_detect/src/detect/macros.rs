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
    (@cached $cache:expr; $feature:tt, $feature_lit:tt) => {
        $crate::detect_feature!(@cached $cache; $feature, $feature_lit : $feature_lit)
    };
    (@cached $cache:expr; $feature:tt, $feature_lit:tt : $($target_feature_lit:tt),*) => {
        // Emit `true` if enabled at compile time
        ($(cfg!(target_feature = $target_feature_lit) ||)*
            $crate::detect::__is_feature_detected::cached::$feature($cache))
    };
    (@cached $cache:expr; $feature:tt, $feature_lit:tt, without cfg check: true) => {
        $crate::detect::__is_feature_detected::cached::$feature($cache)
    };
}

/// Parse a boolean expression of target features for `is_*_feature_detected!`.
#[doc(hidden)]
#[macro_export]
#[allow_internal_unstable(stdarch_internal, feature_detection_boolean_syntax)]
#[unstable(feature = "feature_detection_boolean_syntax", issue = "164651")]
macro_rules! __detect_feature_bool {
    (@expr $macro_name:ident, $cache:ident, [$($acc:tt)*] ($($inner:tt)+) $($rest:tt)*) => {
        $crate::__detect_feature_bool!(
            @op $macro_name, $cache,
            [$($acc)* ($crate::__detect_feature_bool!(@expr $macro_name, $cache, [] $($inner)+))]
            $($rest)*
        )
    };
    (@expr $macro_name:ident, $cache:ident, [$($acc:tt)*] () $($rest:tt)*) => {
        compile_error!("expected target feature in parentheses")
    };
    (@expr $macro_name:ident, $cache:ident, [$($acc:tt)*] && $($rest:tt)*) => {
        compile_error!("expected target feature before `&&`")
    };
    (@expr $macro_name:ident, $cache:ident, [$($acc:tt)*] || $($rest:tt)*) => {
        compile_error!("expected target feature before `||`")
    };
    (@expr $macro_name:ident, $cache:ident, [$($acc:tt)*] $lit:tt $($rest:tt)*) => {
        $crate::__detect_feature_bool!(
            @op $macro_name, $cache,
            [$($acc)* $crate::$macro_name!(@cached $cache, $lit)]
            $($rest)*
        )
    };
    (@op $macro_name:ident, $cache:ident, [$($acc:tt)*] $(,)?) => {
        $($acc)*
    };
    (@op $macro_name:ident, $cache:ident, [$($acc:tt)*] && $($rest:tt)+) => {
        $crate::__detect_feature_bool!(@expr $macro_name, $cache, [$($acc)* &&] $($rest)+)
    };
    (@op $macro_name:ident, $cache:ident, [$($acc:tt)*] || $($rest:tt)+) => {
        $crate::__detect_feature_bool!(@expr $macro_name, $cache, [$($acc)* ||] $($rest)+)
    };
    (@op $macro_name:ident, $cache:ident, [$($acc:tt)*] &&) => {
        compile_error!("expected target feature after `&&`")
    };
    (@op $macro_name:ident, $cache:ident, [$($acc:tt)*] ||) => {
        compile_error!("expected target feature after `||`")
    };
    (@op $macro_name:ident, $cache:ident, [$($acc:tt)*] $($rest:tt)+) => {
        compile_error!("expected `&&` or `||` between target features")
    };
    ($macro_name:ident; $($t:tt)+) => {{
        #[allow(unused_mut, unused_variables)]
        let mut cache = $crate::detect::LocalCache::uninitialized();
        $crate::__detect_feature_bool!(@expr $macro_name, cache, [] $($t)+)
    }};
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
        macro_rules! with_dollar {
            ($d:tt) => {
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
                                    #[$deprecate_attr]
                                    macro_rules! deprecated_feature { {} => {}; }
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
                    // Internal rules for checking a single feature with a shared `LocalCache`.
                    $(
                        (@cached $cache:ident, $feature_lit) => {
                            $crate::detect_feature!(
                                @cached &mut $cache;
                                $feature,
                                $feature_lit
                                $(, without cfg check: $feature_cfg_check)?
                                $(: $($target_feature_lit),*)?
                            )
                        };
                    )*
                    $(
                        (@cached $cache:ident, $bind_feature) => {
                            {
                                $(
                                    #[$deprecate_attr]
                                    macro_rules! deprecated_feature { {} => {}; }
                                    deprecated_feature! {};
                                )?
                                $crate::$macro_name!(@cached $cache, $feature_impl)
                            }
                        };
                    )*
                    $(
                        (@cached $cache:ident, $nort_feature) => {
                            compile_error!(
                                concat!(
                                    stringify!($nort_feature),
                                    " feature cannot be detected at run-time"
                                )
                            )
                        };
                    )*
                    (@cached $cache:ident, $t:tt) => {
                        compile_error!(
                            concat!("unknown ", stringify!($target), " target feature: ", $t)
                        )
                    };
                    // Allow a trailing comma
                    ($t:tt,) => {
                        $crate::$macro_name!($t)
                    };
                    (($d($inner:tt)*)) => {
                        $crate::__detect_feature_bool!($macro_name; ($d($inner)*))
                    };
                    // Fallback, error for unknown target features
                    ($t:tt) => {
                        compile_error!(
                            concat!("unknown ", stringify!($target), " target feature: ", $t)
                        )
                    };
                    ($d($t:tt)+) => {
                        $crate::__detect_feature_bool!($macro_name; $d($t)+)
                    };
                }

                // All other arches still validate the features, but also
                $(#[$macro_attrs])*
                #[macro_export]
                #[allow_internal_unstable(stdarch_internal)]
                #[cfg(not($cfg))]
                #[doc(cfg($cfg))]
                macro_rules! $macro_name {
                    // All known features get an error
                    $(
                        ($feature_lit) => {
                            compile_error!(
                                concat!(
                                    "This macro cannot be used on the current target. \
                                    You can prevent\nit from being used in other \
                                    architectures by guarding it behind\na `cfg(",
                                    stringify!($cfg),
                                    ")`."
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
                    (@cached $cache:ident, $t:tt) => {{
                        $crate::$macro_name!($t);
                        false
                    }};
                    // Allow a trailing comma
                    ($t:tt,) => {
                        $crate::$macro_name!($t);
                    };
                    (($d($inner:tt)*)) => {
                        $crate::__detect_feature_bool!($macro_name; ($d($inner)*))
                    };
                    // Fallback, error for unknown target features
                    ($t:tt) => {
                        compile_error!(
                            concat!("unknown ", stringify!($target), " target feature: ", $t)
                        )
                    };
                    ($d($t:tt)+) => {
                        $crate::__detect_feature_bool!($macro_name; $d($t)+)
                    };
                }
            };
        }
        with_dollar!($);

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

            #[doc(hidden)]
            #[unstable(feature = "stdarch_internal", issue = "none")]
            pub mod cached {
                $(
                    /// PLEASE: do not use this, it is an implementation detail
                    /// subject to change.
                    #[inline]
                    #[doc(hidden)]
                    #[$stability_attr]
                    pub fn $feature(cache: &mut $crate::detect::LocalCache) -> bool {
                        $crate::detect::check_for_cached(cache, $crate::detect::Feature::$feature)
                    }
                )*
            }
        }
    };
}
