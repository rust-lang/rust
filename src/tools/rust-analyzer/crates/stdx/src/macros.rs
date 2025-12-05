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

/// Generates `From` impls for `Enum E { Foo(Foo), Bar(Bar) }` enums
///
/// # Example
///
/// ```ignore
/// impl_from!(Struct, Union, Enum for Adt);
/// ```
#[macro_export]
macro_rules! impl_from {
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
