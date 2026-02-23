//! regression test for <https://github.com/rust-lang/rust/issues/18352>
//@ run-pass

const X: &'static str = "12345";

fn test(s: String) -> bool {
    match &*s {
        X => true,
        _ => false,
    }
}

fn main() {
    assert!(test("12345".to_string()));
}
