// build-pass (FIXME(62277): should be check-pass)

#[allow(unreachable_code)]
fn main() {
    panic!("here's a brace: {"); //~ WARN Panic message contains a brace
    std::panic!("another one: }"); //~ WARN Panic message contains a brace
    core::panic!("Hello { { {"); //~ WARN Panic message contains a brace
    assert!(false, "} } }..."); //~ WARN Panic message contains a brace
}
