//@ check-pass

// Check that we use subtyping when reifying a closure into a function pointer.

fn foo(x: &str) {}

fn main() {
    let c = |_: &str| {};
    let x = c as fn(&'static str);
}
