// run-pass
// edition:2018

#![allow(unused_imports)]
#![allow(non_camel_case_types)]

// Test that ambiguity errors are not emitted between `self::test` and
// `::test`, assuming the latter (crate) is not in `extern_prelude`.
mod test {
    pub struct Foo(pub ());
}
use test::Foo;

// Test that qualified paths can refer to both the external crate and local item.
mod std {
    pub struct io(pub ());
}
use ::std::io as std_io;
use self::std::io as local_io;

fn main() {
    Foo(());
    let _ = std_io::stdout();
    local_io(());

    {
        // Test that having `std_io` in a module scope and a non-module
        // scope is allowed, when both resolve to the same definition.
        use ::std::io as std_io;
        use std_io::stdout;
        let _ = stdout();
    }
}
