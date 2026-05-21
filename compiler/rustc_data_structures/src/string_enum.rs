/// Like an enum, but the variants are tied to a string representation.
///
/// Generates, for each variant, the mapping between the variant and its string
/// form, plus a `VARIANTS` and `STR_VARIANTS` array, an inherent `to_str`
/// method, and `Display`/`FromStr` impls. Intended for use with command-line
/// flag values where the set of valid strings is fixed and known at compile
/// time.
///
/// `FromStr::Err` is `()` because diagnostic emission is handled by the caller.
#[macro_export]
macro_rules! string_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident => $repr:expr,
            )*
        }
    ) => {
        $(#[$meta])*
        $vis enum $name {
            $(
                $(#[$variant_meta])*
                $variant,
            )*
        }

        impl $name {
            #[allow(dead_code)]
            $vis const VARIANTS: &'static [Self] = &[
                $( Self::$variant, )*
            ];
            #[allow(dead_code)]
            $vis const STR_VARIANTS: &'static [&'static str] = &[
                $( Self::$variant.to_str(), )*
            ];

            $vis const fn to_str(&self) -> &'static str {
                match self {
                    $( Self::$variant => $repr, )*
                }
            }
        }

        impl ::std::fmt::Display for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                ::std::fmt::Display::fmt(self.to_str(), f)
            }
        }

        impl ::std::str::FromStr for $name {
            type Err = ();

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $( $repr => Ok(Self::$variant), )*
                    _ => Err(()),
                }
            }
        }
    }
}
