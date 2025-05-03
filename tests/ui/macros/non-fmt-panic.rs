//! The non_fmt_panics lint detects panic!(..) invocations where
//! the first argument is not a formatting string.
//!
//! Also, this test checks that this is not emitted if it originates
//! in an external macro.

//@ run-rustfix
//@ rustfix-only-machine-applicable
//@ build-pass (FIXME(62277): should be check-pass)
//@ aux-build:fancy-panic.rs

extern crate fancy_panic;

const C: &str = "abc {}";
static S: &str = "{bla}";

#[allow(unreachable_code)]
fn main() {
    panic!("here's a brace: {"); //~ WARN panic message contains a brace
    unreachable!("here's a brace: {"); //~ WARN panic message contains a brace
    std::panic!("another one: }"); //~ WARN panic message contains a brace
    core::panic!("Hello {}"); //~ WARN panic message contains an unused formatting placeholder
    assert!(false, "{:03x} {test} bla");
    //~^ WARN panic message contains unused formatting placeholders
    assert!(false, S);
    //~^ WARN panic message is not a string literal
    assert!(false, 123);
    //~^ WARN panic message is not a string literal
    assert!(false, Some(123));
    //~^ WARN panic message is not a string literal
    debug_assert!(false, "{{}} bla"); //~ WARN panic message contains braces
    panic!(C); //~ WARN panic message is not a string literal
    panic!(S); //~ WARN panic message is not a string literal
    unreachable!(S); //~ WARN panic message is not a string literal
    unreachable!(S); //~ WARN panic message is not a string literal
    std::panic!(123); //~ WARN panic message is not a string literal
    core::panic!(&*"abc"); //~ WARN panic message is not a string literal
    panic!(Some(123)); //~ WARN panic message is not a string literal
    panic!(concat!("{", "}")); //~ WARN panic message contains an unused formatting placeholder
    panic!(concat!("{", "{")); //~ WARN panic message contains braces

    fancy_panic::fancy_panic!("test {} 123");
    //~^ WARN panic message contains an unused formatting placeholder

    fancy_panic::fancy_panic!(); // OK
    fancy_panic::fancy_panic!(S); // OK

    macro_rules! a {
        () => { 123 };
    }

    panic!(a!()); //~ WARN panic message is not a string literal
    unreachable!(a!()); //~ WARN panic message is not a string literal

    panic!(format!("{}", 1)); //~ WARN panic message is not a string literal
    unreachable!(format!("{}", 1)); //~ WARN panic message is not a string literal
    assert!(false, format!("{}", 1)); //~ WARN panic message is not a string literal
    debug_assert!(false, format!("{}", 1)); //~ WARN panic message is not a string literal

    panic![123]; //~ WARN panic message is not a string literal
    panic!{123}; //~ WARN panic message is not a string literal

    // Check that the lint only triggers for std::panic and core::panic,
    // not any panic macro:
    macro_rules! panic {
        ($e:expr) => ();
    }
    panic!("{}"); // OK
    panic!(S); // OK

    a(1);
    b(1);
    c(1);
    d(1);
}

fn a<T: Send + 'static>(v: T) {
    panic!(v); //~ WARN panic message is not a string literal
    assert!(false, v); //~ WARN panic message is not a string literal
}

fn b<T: std::fmt::Debug + Send + 'static>(v: T) {
    panic!(v); //~ WARN panic message is not a string literal
    assert!(false, v); //~ WARN panic message is not a string literal
}

fn c<T: std::fmt::Display + Send + 'static>(v: T) {
    panic!(v); //~ WARN panic message is not a string literal
    assert!(false, v); //~ WARN panic message is not a string literal
}

fn d<T: std::fmt::Display + std::fmt::Debug + Send + 'static>(v: T) {
    panic!(v); //~ WARN panic message is not a string literal
    assert!(false, v); //~ WARN panic message is not a string literal
}
