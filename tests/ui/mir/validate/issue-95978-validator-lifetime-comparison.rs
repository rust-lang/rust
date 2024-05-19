//@ check-pass
//@ compile-flags: -Zvalidate-mir

fn foo(_a: &str) {}

fn main() {
    let x = foo as fn(&'static str);

    let _ = x == foo;
}
