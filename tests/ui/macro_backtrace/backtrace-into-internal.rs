//! We try to hide the internals of builtin-like macros from error messages,
//! but *not* if `-Z macro-backtrace` is enabled

//@ revisions: default with
//@[with] compile-flags: -Z macro-backtrace
//@[with] error-pattern: in this expansion of `macro_two!`

#![feature(allow_internal_unstable)]

#[allow_internal_unstable(hint_must_use, liballoc_internals)]
macro_rules! macro_one {
    ($($arg:tt)*) => {
        std::convert::identity(macro_two!($($arg)*))
    }
}

#[allow_internal_unstable(fmt_internals)]
macro_rules! macro_two {
    ($($arg:tt)*) => {
        { $($arg)* + 1 }
        //~^ERROR cannot add `{integer}` to `&str`
    }
}

fn main(){
    let _ = macro_one!("boo");
}
