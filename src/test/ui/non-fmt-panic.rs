// build-pass (FIXME(62277): should be check-pass)
// aux-build:fancy-panic.rs

extern crate fancy_panic;

const C: &str = "abc {}";
static S: &str = "{bla}";

#[allow(unreachable_code)]
fn main() {
    panic!("here's a brace: {"); //~ WARN panic message contains a brace
    std::panic!("another one: }"); //~ WARN panic message contains a brace
    core::panic!("Hello {}"); //~ WARN panic message contains an unused formatting placeholder
    assert!(false, "{:03x} {test} bla");
    //~^ WARN panic message contains unused formatting placeholders
    assert!(false, S);
    //~^ WARN panic message is not a string literal
    debug_assert!(false, "{{}} bla"); //~ WARN panic message contains braces
    panic!(C); //~ WARN panic message is not a string literal
    panic!(S); //~ WARN panic message is not a string literal
    std::panic!(123); //~ WARN panic message is not a string literal
    core::panic!(&*"abc"); //~ WARN panic message is not a string literal
    panic!(concat!("{", "}")); //~ WARN panic message contains an unused formatting placeholder
    panic!(concat!("{", "{")); //~ WARN panic message contains braces

    fancy_panic::fancy_panic!("test {} 123");
    //~^ WARN panic message contains an unused formatting placeholder

    fancy_panic::fancy_panic!(S);
    //~^ WARN panic message is not a string literal

    macro_rules! a {
        () => { 123 };
    }

    panic!(a!()); //~ WARN panic message is not a string literal

    panic!(format!("{}", 1)); //~ WARN panic message is not a string literal
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
}
