//! Regression test for https://github.com/rust-lang/rust/issues/10683

//@ run-pass

static NAME: &'static str = "hello world";

fn main() {
    match &*NAME.to_ascii_lowercase() {
        "foo" => {}
        _ => {}
    }
}
