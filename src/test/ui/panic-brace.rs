// build-pass (FIXME(62277): should be check-pass)

#[allow(unreachable_code)]
fn main() {
    panic!("here's a brace: {"); //~ WARN panic message contains a brace
    std::panic!("another one: }"); //~ WARN panic message contains a brace
    core::panic!("Hello {}"); //~ WARN panic message contains an unused formatting placeholder
    assert!(false, "{:03x} bla"); //~ WARN panic message contains an unused formatting placeholder
    debug_assert!(false, "{{}} bla"); //~ WARN panic message contains a brace
}
