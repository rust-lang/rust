//! Convenience macros.

/// Appends formatted string to a `String`.
#[macro_export]
macro_rules! format_to {
    ($buf:expr) => ();
    ($buf:expr, $lit:literal $($arg:tt)*) => {
        {
            use ::std::fmt::Write as _;
            // We can't do ::std::fmt::Write::write_fmt($buf, format_args!($lit $($arg)*))
            // unfortunately, as that loses out on autoref behavior.
            _ = $buf.write_fmt(format_args!($lit $($arg)*))
        }
    };
}

/// Appends formatted string to a `String` and returns the `String`.
///
/// Useful for folding iterators into a `String`.
#[macro_export]
macro_rules! format_to_acc {
    ($buf:expr, $lit:literal $($arg:tt)*) => {
        {
            use ::std::fmt::Write as _;
            // We can't do ::std::fmt::Write::write_fmt($buf, format_args!($lit $($arg)*))
            // unfortunately, as that loses out on autoref behavior.
            _ = $buf.write_fmt(format_args!($lit $($arg)*));
            $buf
        }
    };
}

/// Generates `From` impls that wrap values in, or map variants between, enums.
///
/// Enum mappings with `Source => Target` rename the variant and convert its payload with `.into()`.
///
/// # Examples
///
/// ```ignore
/// impl_from!(Struct, Union, Enum for Adt);
/// impl_from!(impl<'db> Item, InternedItem<'db> for ItemId<'db>);
/// impl_from!(AssocItem { Function, Const, TypeAlias } for Item);
/// impl_from!(AdtId { StructId => Struct, EnumId => Enum } for Adt);
/// ```
#[macro_export]
macro_rules! impl_from {
    (
        impl<$lifetime:lifetime>
        $source:ident $(<$source_lifetime:lifetime>)?
        { $($source_variant:ident $(=> $target_variant:ident)?),* $(,)? }
        for $target:ident<$target_lifetime:lifetime>
    ) => {
        impl<$lifetime> From<$source$(<$source_lifetime>)?> for $target<$target_lifetime> {
            fn from(value: $source$(<$source_lifetime>)?) -> Self {
                match value {
                    $(
                        $source::$source_variant(value) => $crate::impl_from!(
                            @map_enum_variant $target, value, $source_variant
                            $(=> $target_variant)?
                        ),
                    )*
                }
            }
        }
    };
    (
        $source:ident
        { $($source_variant:ident $(=> $target_variant:ident)?),* $(,)? }
        for $target:ident
    ) => {
        impl From<$source> for $target {
            fn from(value: $source) -> Self {
                match value {
                    $(
                        $source::$source_variant(value) => $crate::impl_from!(
                            @map_enum_variant $target, value, $source_variant
                            $(=> $target_variant)?
                        ),
                    )*
                }
            }
        }
    };
    (@map_enum_variant $target:ident, $value:ident, $variant:ident) => {
        $target::$variant($value)
    };
    (
        @map_enum_variant $target:ident, $value:ident,
        $source_variant:ident => $target_variant:ident
    ) => {
        $target::$target_variant($value.into())
    };
    (
        impl<$lifetime:lifetime>
        $(
            $variant:ident $(<$variant_lifetime:lifetime>)?
            $(($($sub_variant:ident $(<$sub_variant_lifetime:lifetime>)?),*))?
        ),*
        for $enum:ident<$enum_lifetime:lifetime>
    ) => {
        $(
            impl<$lifetime> From<$variant$(<$variant_lifetime>)?> for $enum<$enum_lifetime> {
                fn from(it: $variant$(<$variant_lifetime>)?) -> $enum<$enum_lifetime> {
                    $enum::$variant(it)
                }
            }
            $($(
                impl<$lifetime> From<$sub_variant$(<$sub_variant_lifetime>)?>
                    for $enum<$enum_lifetime>
                {
                    fn from(
                        it: $sub_variant$(<$sub_variant_lifetime>)?,
                    ) -> $enum<$enum_lifetime> {
                        $enum::$variant($variant::$sub_variant(it))
                    }
                }
            )*)?
        )*
    };
    ($($variant:ident $(($($sub_variant:ident),*))?),* for $enum:ident) => {
        $(
            impl From<$variant> for $enum {
                fn from(it: $variant) -> $enum {
                    $enum::$variant(it)
                }
            }
            $($(
                impl From<$sub_variant> for $enum {
                    fn from(it: $sub_variant) -> $enum {
                        $enum::$variant($variant::$sub_variant(it))
                    }
                }
            )*)?
        )*
    };
    ($($variant:ident$(<$V:ident>)?),* for $enum:ident) => {
        $(
            impl$(<$V>)? From<$variant$(<$V>)?> for $enum$(<$V>)? {
                fn from(it: $variant$(<$V>)?) -> $enum$(<$V>)? {
                    $enum::$variant(it)
                }
            }
        )*
    }
}
