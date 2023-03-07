// This test is similar to `basic.rs`, but nested in modules.

// run-pass
// edition:2018

#![feature(decl_macro)]

#![allow(unused_imports)]
#![allow(non_camel_case_types)]

mod foo {
    // Test that ambiguity errors are not emitted between `self::test` and
    // `::test`, assuming the latter (crate) is not in `extern_prelude`.
    mod test {
        pub struct Foo(pub ());
    }
    pub use test::Foo;

    // Test that qualified paths can refer to both the external crate and local item.
    mod std {
        pub struct io(pub ());
    }
    pub use ::std::io as std_io;
    pub use self::std::io as local_io;
}

// Test that we can refer to the external crate unqualified
// (when there isn't a local item with the same name).
use std::io;

mod bar {
    // Also test the unqualified external crate import in a nested module,
    // to show that the above import doesn't resolve through a local `std`
    // item, e.g., the automatically injected `extern crate std;`, which in
    // the Rust 2018 should no longer be visible through `crate::std`.
    pub use std::io;

    // Also test that items named `std` in other namespaces don't
    // cause ambiguity errors for the import from `std` above.
    pub fn std() {}
    pub macro std() {}
}


fn main() {
    foo::Foo(());
    let _ = foo::std_io::stdout();
    foo::local_io(());
    let _ = io::stdout();
    let _ = bar::io::stdout();
    bar::std();
    bar::std!();

    {
        // Test that having `io` in a module scope and a non-module
        // scope is allowed, when both resolve to the same definition.
        use std::io;
        use io::stdout;
        let _ = stdout();
    }
}
