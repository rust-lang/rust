/// Like an enum, but the variants are tied to a string representation.
///
/// Each variant is declared as `Variant => "primary"` or, when multiple
/// strings should parse to the same variant, `Variant => "primary" | "alias1" | "alias2"`.
/// The first string is the canonical form returned by `to_str` and `Display`;
/// all forms are accepted by `FromStr`.
///
/// Generates:
/// * `VARIANTS` — every variant, in declaration order.
/// * `STR_VARIANTS` — each variant's canonical string, in declaration order.
/// * `ALL_STR_VARIANTS` — every accepted string (canonical + aliases) in
///   declaration order. Use this when help text should list all accepted forms.
/// * `to_str()`, `Display`, `FromStr`. `FromStr::Err` is `()` because diagnostic
///   emission is handled by the caller.
#[macro_export]
macro_rules! string_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident => $repr:literal $( | $alias:literal )* ,
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
            #[allow(dead_code)]
            $vis const ALL_STR_VARIANTS: &'static [&'static str] = &[
                $( $repr, $( $alias, )* )*
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
                    $( $repr $( | $alias )* => Ok(Self::$variant), )*
                    _ => Err(()),
                }
            }
        }
    }
}
