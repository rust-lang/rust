/// Like an enum, but the variants are tied to a string representation.
///
/// Each variant is declared in one of three forms:
/// * `Variant => "primary"` — single canonical CLI string.
/// * `Variant => "primary" | "alias1" | "alias2"` — canonical string plus
///   one or more aliases that also parse to this variant. `to_str` and
///   `Display` return the canonical form.
/// * `Variant` (no `=>`) — the variant exists in the enum but has no CLI
///   string representation. Reachable only by code that produces the value
///   directly (e.g. a parser handling no-value or boolean fallthrough).
///   Calling `to_str` or `Display` on such a variant panics, and `FromStr`
///   will not produce it.
///
/// Generates:
/// * `VARIANTS` — every variant, in declaration order.
/// * `STR_VARIANTS` — canonical string of each variant that has one, in
///   declaration order.
/// * `ALL_STR_VARIANTS` — every accepted string (canonical + aliases) in
///   declaration order. Use this when help text should list all accepted
///   forms.
/// * `to_str()`, `Display`, `FromStr`. `FromStr::Err` is `()` because
///   diagnostic emission is handled by the caller.
#[macro_export]
macro_rules! string_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident $( => $repr:literal $( | $alias:literal )* )? ,
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
                $( $( $repr, )? )*
            ];
            #[allow(dead_code)]
            $vis const ALL_STR_VARIANTS: &'static [&'static str] = &[
                $( $( $repr, $( $alias, )* )? )*
            ];

            #[allow(unreachable_patterns)]
            $vis const fn to_str(&self) -> &'static str {
                match self {
                    $( $( Self::$variant => $repr, )? )*
                    _ => panic!("variant has no CLI string representation"),
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
                    $( $( $repr $( | $alias )* => Ok(Self::$variant), )? )*
                    _ => Err(()),
                }
            }
        }
    }
}
