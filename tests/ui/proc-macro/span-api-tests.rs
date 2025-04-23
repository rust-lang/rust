//@ run-pass
//@ proc-macro: span-api-tests.rs
//@ aux-build:span-test-macros.rs
//@ compile-flags: -Ztranslate-remapped-path-to-local-path=yes

#[macro_use]
extern crate span_test_macros;

extern crate span_api_tests;

use span_api_tests::{reemit, assert_local_file, macro_stringify};

macro_rules! say_hello {
    ($macname:ident) => ( $macname! { "Hello, world!" })
}

assert_local_file! { "Hello, world!" }

say_hello! { assert_local_file }

reemit_legacy! {
    assert_local_file! { "Hello, world!" }
}

say_hello_extern! { assert_local_file }

reemit! {
    assert_local_file! { "Hello, world!" }
}

fn main() {
    let s = macro_stringify!(Hello, world!);
    assert_eq!(s, "Hello, world!");
    assert_eq!(macro_stringify!(Hello, world!), "Hello, world!");
    assert_eq!(reemit_legacy!(macro_stringify!(Hello, world!)), "Hello, world!");
    reemit_legacy!(assert_eq!(macro_stringify!(Hello, world!), "Hello, world!"));
    // reemit change the span to be that of the call site
    assert_eq!(
        reemit!(macro_stringify!(Hello, world!)),
        "reemit!(macro_stringify!(Hello, world!))"
    );
    let r = "reemit!(assert_eq!(macro_stringify!(Hello, world!), r))";
    reemit!(assert_eq!(macro_stringify!(Hello, world!), r));

    assert_eq!(macro_stringify!(
        Hello,
        world!
    ), "Hello,\n        world!");

    assert_eq!(macro_stringify!(Hello, /*world */ !), "Hello, /*world */ !");
        assert_eq!(macro_stringify!(
        Hello,
        // comment
        world!
    ), "Hello,\n        // comment\n        world!");

    assert_eq!(say_hello! { macro_stringify }, "\"Hello, world!\"");
    assert_eq!(say_hello_extern! { macro_stringify }, "\"Hello, world!\"");
}
