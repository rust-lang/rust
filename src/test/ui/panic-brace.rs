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
    debug_assert!(false, "{{}} bla"); //~ WARN panic message contains braces
    panic!(C); // No warning (yet)
    panic!(S); // No warning (yet)
    panic!(concat!("{", "}")); //~ WARN panic message contains an unused formatting placeholder
    panic!(concat!("{", "{")); //~ WARN panic message contains braces

    fancy_panic::fancy_panic!("test {} 123");
    //~^ WARN panic message contains an unused formatting placeholder

    // Check that the lint only triggers for std::panic and core::panic,
    // not any panic macro:
    macro_rules! panic {
        ($e:expr) => ();
    }
    panic!("{}"); // OK
}
