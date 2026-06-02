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
/// Any variant may also carry an explicit discriminant
/// (`Variant = N` or `Variant = N => "primary"`), forwarded verbatim to
/// the generated enum. Use this when the discriminant values are
/// load-bearing (e.g. encoded on the wire or stable-hashed).
///
/// Variants with a CLI string may also be marked `@no_from_str`
/// (`Variant => "primary" @no_from_str`), which makes the string
/// available to `to_str`/`Display` but excludes it from `FromStr`.
/// Use this for variants that have a textual identity for display
/// purposes but should not be constructible from untrusted user input
/// (e.g. variants that require out-of-band context to build correctly).
/// Such variants still appear in `STR_VARIANTS`/`ALL_STR_VARIANTS`;
/// callers that want only the strings the user is allowed to supply
/// should use `FROM_STR_VARIANTS` instead.
///
/// Generates:
/// * `VARIANTS` — every variant, in declaration order.
/// * `STR_VARIANTS` — canonical string of each variant that has one, in
///   declaration order.
/// * `ALL_STR_VARIANTS` — every accepted string (canonical + aliases) in
///   declaration order. Use this when help text should list all accepted
///   forms.
/// * `FROM_STR_VARIANTS` — canonical string of each variant whose canonical
///   string is accepted by `FromStr` (i.e. `STR_VARIANTS` minus any variant
///   marked `@no_from_str`). Use this in diagnostics that list the inputs
///   the user is actually allowed to supply.
/// * `to_str()`, `Display`, `FromStr`. `FromStr::Err` is `()` because
///   diagnostic emission is handled by the caller.
#[macro_export]
macro_rules! string_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident $( = $disc:expr )?
                    $( => $repr:literal $( | $alias:literal )*
                       $( @ $no_from_str:ident )? )? ,
            )*
        }
    ) => {
        $(#[$meta])*
        $vis enum $name {
            $(
                $(#[$variant_meta])*
                $variant $( = $disc )?,
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
            #[allow(dead_code)]
            $vis const FROM_STR_VARIANTS: &'static [&'static str] =
                $crate::__string_enum_from_str_arr!(
                    @collect []
                    $( $( $repr $( @ $no_from_str )? , )? )*
                );

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

            #[allow(unreachable_code)]
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $( $( $repr $( | $alias )* => {
                        $(
                            $crate::__string_enum_check_no_from_str!($no_from_str);
                            return Err(());
                        )?
                        Ok(Self::$variant)
                    }, )? )*
                    _ => Err(()),
                }
            }
        }
    }
}

/// Validates that a `string_enum!` variant's `@`-marker is spelled exactly
/// `no_from_str`. Used internally by [`string_enum!`]; not part of the public
/// surface.
#[doc(hidden)]
#[macro_export]
macro_rules! __string_enum_check_no_from_str {
    (no_from_str) => {};
}

/// Builds a `&[&str]` literal containing only the canonical strings of
/// variants accepted by `FromStr`. Token-tree munches a comma-terminated
/// stream of `$repr` (or `$repr @ no_from_str`) entries into an accumulator,
/// then emits the full slice literal in one go — needed because in
/// expression position a macro must expand to a single expression. Used
/// internally by [`string_enum!`]; not part of the public surface.
#[doc(hidden)]
#[macro_export]
macro_rules! __string_enum_from_str_arr {
    (@collect [$($acc:literal,)*]) => {
        &[ $($acc,)* ]
    };
    (@collect [$($acc:literal,)*] $repr:literal @ no_from_str , $($rest:tt)*) => {
        $crate::__string_enum_from_str_arr!(@collect [$($acc,)*] $($rest)*)
    };
    (@collect [$($acc:literal,)*] $repr:literal , $($rest:tt)*) => {
        $crate::__string_enum_from_str_arr!(@collect [$($acc,)* $repr,] $($rest)*)
    };
}
