#[allow(unused)]
macro_rules! features {
    (
      @TARGET: $target:ident;
      @MACRO_NAME: $macro_name:ident;
      @MACRO_ATTRS: $(#[$macro_attrs:meta])*
      $(@BIND_FEATURE_NAME: $bind_feature:tt; $feature_impl:tt; )*
      $(@NO_RUNTIME_DETECTION: $nort_feature:tt; )*
      $(@FEATURE: #[$stability_attr:meta] $feature:ident: $feature_lit:tt; $(#[$feature_comment:meta])*)*
    ) => {
        #[macro_export]
        $(#[$macro_attrs])*
        #[allow_internal_unstable(stdsimd_internal)]
        macro_rules! $macro_name {
            $(
                ($feature_lit) => {
                    $crate::detect::__is_feature_detected::$feature()
                };
            )*
            $(
                ($bind_feature) => { $macro_name!($feature_impl) };
            )*
            $(
                ($nort_feature) => {
                    compile_error!(
                        concat!(
                            stringify!(nort_feature),
                            " feature cannot be detected at run-time"
                        )
                    )
                };
            )*
            ($t:tt,) => {
                    $macro_name!($t);
            };
            ($t:tt) => {
                compile_error!(
                    concat!(
                        concat!("unknown ", stringify!($target)),
                        concat!(" target feature: ", $t)
                    )
                )
            };
        }

        /// Each variant denotes a position in a bitset for a particular feature.
        ///
        /// PLEASE: do not use this, it is an implementation detail subject
        /// to change.
        #[doc(hidden)]
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone)]
        #[repr(u8)]
        #[unstable(feature = "stdsimd_internal", issue = "none")]
        pub(crate) enum Feature {
            $(
                $(#[$feature_comment])*
                $feature,
            )*

            // Do not add variants after last:
            _last
        }

        impl Feature {
            pub(crate) fn to_str(self) -> &'static str {
                match self {
                    $(Feature::$feature => $feature_lit,)*
                    Feature::_last => unreachable!(),
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
        pub mod __is_feature_detected {
            $(

                /// PLEASE: do not use this, it is an implementation detail
                /// subject to change.
                #[inline]
                #[doc(hidden)]
                #[$stability_attr]
                pub fn $feature() -> bool {
                    cfg!(target_feature = $feature_lit) ||
                        $crate::detect::check_for($crate::detect::Feature::$feature)
                }
            )*
        }
    };
}
