//@ run-pass

#![deny(rust_2021_incompatible_closure_captures)]

#[derive(Debug)]
struct Foo(i32);
impl Drop for Foo {
    fn drop(&mut self) {
        println!("{:?} dropped", self.0);
    }
}

struct ConstainsDropField(Foo, Foo);

// Test that if all paths starting at root variable that implement Drop are captured
// then it doesn't trigger the lint.
fn test_precise_analysis_simple_1() {
    let t = (Foo(10), Foo(20), Foo(30));

    let c = || {
        let _t = t.0;
        let _t = t.1;
        let _t = t.2;
    };

    c();
}

// Test that if all paths starting at root variable that implement Drop are captured
// then it doesn't trigger the lint.
fn test_precise_analysis_simple_2() {
    let t = ConstainsDropField(Foo(10), Foo(20));

    let c = || {
        let _t = t.0;
        let _t = t.1;
    };

    c();
}

#[derive(Debug)]
struct ContainsAndImplsDrop(Foo);
impl Drop for ContainsAndImplsDrop {
    fn drop(&mut self) {
        println!("{:?} dropped", self.0);
    }
}

// If a path isn't directly captured but requires Drop, then this tests that migrations aren't
// needed if the parent to that path is captured.
fn test_precise_analysis_parent_captured_1() {
    let t = ConstainsDropField(Foo(10), Foo(20));

    let c = || {
        let _t = t;
    };

    c();
}

// If a path isn't directly captured but requires Drop, then this tests that migrations aren't
// needed if the parent to that path is captured.
fn test_precise_analysis_parent_captured_2() {
    let t = ContainsAndImplsDrop(Foo(10));

    let c = || {
        let _t = t;
    };

    c();
}

struct S;
impl Drop for S {
    fn drop(&mut self) {}
}

struct T(S, S);
struct U(T, T);

// Test that if the path is longer than just one element, precise analysis works correctly.
fn test_precise_analysis_long_path() {
    let u = U(T(S, S), T(S, S));

    let c = || {
        let _x = u.0.0;
        let _x = u.0.1;
        let _x = u.1.0;
        let _x = u.1.1;
    };

    c();
}

fn main() {
    test_precise_analysis_simple_1();
    test_precise_analysis_simple_2();

    test_precise_analysis_parent_captured_1();
    test_precise_analysis_parent_captured_2();

    test_precise_analysis_long_path();
}
