//! Convenience macros.

use std::{
    fmt, mem, panic,
    sync::atomic::{AtomicUsize, Ordering::SeqCst},
};
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
    }
}

/// A version of `assert!` macro which allows to handle an assertion failure.
///
/// In release mode, it returns the condition and logs an error.
///
/// ```
/// if assert_never!(impossible) {
///     // Heh, this shouldn't have happened, but lets try to soldier on...
///     return None;
/// }
/// ```
///
/// Rust analyzer is a long-running process, and crashing really isn't an option.
///
/// Shamelessly stolen from: https://www.sqlite.org/assert.html
#[macro_export]
macro_rules! assert_never {
    ($cond:expr) => { $crate::assert_always!($cond, "") };
    ($cond:expr, $($fmt:tt)*) => {{
        let value = $cond;
        if value {
            $crate::on_assert_failure(
                format_args!($($fmt)*)
            );
        }
        value
    }};
}

type AssertHook = fn(&panic::Location<'_>, fmt::Arguments<'_>);
static HOOK: AtomicUsize = AtomicUsize::new(0);

pub fn set_assert_hook(hook: AssertHook) {
    HOOK.store(hook as usize, SeqCst);
}

#[cold]
#[track_caller]
pub fn on_assert_failure(args: fmt::Arguments) {
    let hook: usize = HOOK.load(SeqCst);
    if hook == 0 {
        panic!("\n  assertion failed: {}\n", args);
    }

    let hook: AssertHook = unsafe { mem::transmute::<usize, AssertHook>(hook) };
    hook(panic::Location::caller(), args)
}
