//@ check-pass

// Regression test for https://github.com/rust-lang/rust/issues/152938
// The irrefutable `let...else` diagnostic should explain that the pattern
// always matches and point at the `else` block for removal.

pub fn say_hello(name: Option<String>) {
    let name_str = Some(name) else { return; };
    //~^ WARN unreachable `else` clause
    drop(name_str);
}

fn main() {}
