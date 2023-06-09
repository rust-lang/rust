// This test is similar to `basic.rs`, but with macros defining local items.

// run-pass
// edition:2018

#![allow(non_camel_case_types)]

// Test that ambiguity errors are not emitted between `self::test` and
// `::test`, assuming the latter (crate) is not in `extern_prelude`.
macro_rules! m1 {
    () => {
        mod test {
            pub struct Foo(pub ());
        }
    }
}
use test::Foo;
m1!();

// Test that qualified paths can refer to both the external crate and local item.
macro_rules! m2 {
    () => {
        mod std {
            pub struct io(pub ());
        }
    }
}
use ::std::io as std_io;
use self::std::io as local_io;
m2!();

fn main() {
    Foo(());
    let _ = std_io::stdout();
    local_io(());
}
