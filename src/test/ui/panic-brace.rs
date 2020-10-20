// build-pass (FIXME(62277): should be check-pass)

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
}
