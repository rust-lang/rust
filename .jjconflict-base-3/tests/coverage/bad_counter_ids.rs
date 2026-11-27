#![feature(coverage_attribute)]
//@ edition: 2021
//@ compile-flags: -Copt-level=0 -Zmir-opt-level=3

// Regression test for <https://github.com/rust-lang/rust/issues/117012>.
//
// If some coverage counters were removed by MIR optimizations, we need to take
// care not to refer to those counter IDs in coverage mappings, and instead
// replace them with a constant zero value. If we don't, `llvm-cov` might see
// a too-large counter ID and silently discard the entire function from its
// coverage reports.

#[derive(Debug, PartialEq, Eq)]
struct Foo(u32);

fn eq_good() {
    println!("a");
    assert_eq!(Foo(1), Foo(1));
}

fn eq_good_message() {
    println!("b");
    assert_eq!(Foo(1), Foo(1), "message b");
}

fn ne_good() {
    println!("c");
    assert_ne!(Foo(1), Foo(3));
}

fn ne_good_message() {
    println!("d");
    assert_ne!(Foo(1), Foo(3), "message d");
}

fn eq_bad() {
    println!("e");
    assert_eq!(Foo(1), Foo(3));
}

fn eq_bad_message() {
    println!("f");
    assert_eq!(Foo(1), Foo(3), "message f");
}

fn ne_bad() {
    println!("g");
    assert_ne!(Foo(1), Foo(1));
}

fn ne_bad_message() {
    println!("h");
    assert_ne!(Foo(1), Foo(1), "message h");
}

#[coverage(off)]
fn main() {
    eq_good();
    eq_good_message();
    ne_good();
    ne_good_message();

    assert!(std::panic::catch_unwind(eq_bad).is_err());
    assert!(std::panic::catch_unwind(eq_bad_message).is_err());
    assert!(std::panic::catch_unwind(ne_bad).is_err());
    assert!(std::panic::catch_unwind(ne_bad_message).is_err());
}
