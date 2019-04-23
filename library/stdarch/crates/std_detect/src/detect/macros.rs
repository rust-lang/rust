macro_rules! features {
    (
      @TARGET: $target:ident;
      @MACRO_NAME: $macro_name:ident;
      @MACRO_ATTRS: $(#[$macro_attrs:meta])*
      $(@BIND_FEATURE_NAME: $bind_feature:tt; $feature_impl:tt; )*
      $(@NO_RUNTIME_DETECTION: $nort_feature:tt; )*
      $(@FEATURE: $feature:ident: $feature_lit:tt; $(#[$feature_comment:meta])*)*
    ) => {
        #[macro_export]
        $(#[$macro_attrs])*
        #[allow_internal_unstable(stdsimd_internal,stdsimd)]
        macro_rules! $macro_name {
            $(
                ($feature_lit) => {
                    cfg!(target_feature = $feature_lit) ||
                        $crate::detect::check_for($crate::detect::Feature::$feature)
                };
            )*
            $(
                ($bind_feature) => { $macro_name!($feature_impl); };
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
        #[unstable(feature = "stdsimd_internal", issue = "0")]
        pub enum Feature {
            $(
                $(#[$feature_comment])*
                $feature,
            )*

            // Do not add variants after last:
            _last
        }

        impl Feature {
            pub fn to_str(self) -> &'static str {
                match self {
                    $(Feature::$feature => $feature_lit,)*
                    Feature::_last => unreachable!(),
                }
            }
        }
    };
}
