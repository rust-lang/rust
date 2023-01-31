//! Convenience macros.

#[macro_export]
macro_rules! eprintln {
    ($($tt:tt)*) => {{
        if $crate::is_ci() {
            panic!("Forgot to remove debug-print?")
        }
        std::eprintln!($($tt)*)
    }}
}

/// Appends formatted string to a `String`.
#[macro_export]
macro_rules! format_to {
    ($buf:expr) => ();
    ($buf:expr, $lit:literal $($arg:tt)*) => {
        { use ::std::fmt::Write as _; let _ = ::std::write!($buf, $lit $($arg)*); }
    };
}

/// Generates `From` impls for `Enum E { Foo(Foo), Bar(Bar) }` enums
///
/// # Example
///
/// ```rust
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
