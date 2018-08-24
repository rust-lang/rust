// ignore-wasm32-bare compiled with panic=abort by default
// compile-flags: --test
#[test]
#[should_panic(expected = "foo")]
pub fn test_foo() {
    panic!("foo bar")
}

#[test]
#[should_panic(expected = "foo")]
pub fn test_foo_dynamic() {
    panic!("{} bar", "foo")
}
