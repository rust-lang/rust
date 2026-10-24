//! Regression test for <https://github.com/rust-lang/rust/issues/3574>.
//! Test pattern-matching on `&str` doesn't ICE.
//@ run-pass

fn compare(x: &str, y: &str) -> bool {
    match x {
        "foo" => y == "foo",
        _ => y == "bar",
    }
}

pub fn main() {
    assert!(compare("foo", "foo"));
}
