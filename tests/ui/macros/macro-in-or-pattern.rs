// Regression test for https://github.com/rust-lang/rust/issues/23891
// Tests that a macro expanding to a pattern works correctly inside of or patterns

//@ run-pass
macro_rules! id {
    ($s: pat) => ($s);
}

fn main() {
    match (Some(123), Some(456)) {
        (id!(Some(a)), _) | (_, id!(Some(a))) => println!("{}", a),
        _ => (),
    }
}
