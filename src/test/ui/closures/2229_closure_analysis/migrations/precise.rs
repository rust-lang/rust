#![deny(disjoint_capture_drop_reorder)]
//~^ NOTE: the lint level is defined here

#[derive(Debug)]
struct Foo(i32);
impl Drop for Foo {
    fn drop(&mut self) {
        println!("{:?} dropped", self.0);
    }
}

struct ConstainsDropField(Foo, Foo);

#[derive(Debug)]
struct ContainsAndImplsDrop(Foo);
impl Drop for ContainsAndImplsDrop {
    fn drop(&mut self) {
        println!("{:?} dropped", self.0);
    }
}

// Test that even if all paths starting at root variable that implement Drop are captured,
// the lint is triggered if the root variable implements drop and isn't captured.
fn test_precise_analysis_parent_root_impl_drop_not_captured() {
    let t = ContainsAndImplsDrop(Foo(10));

    let c = || {
    //~^ERROR: drop order affected for closure because of `capture_disjoint_fields`
    //~| NOTE: drop(&(t));
        let _t = t.0;
    };

    c();
}

// Test that lint is triggered if a path that implements Drop is not captured by move
fn test_precise_analysis_drop_paths_not_captured_by_move() {
    let t = ConstainsDropField(Foo(10), Foo(20));

    let c = || {
    //~^ERROR: drop order affected for closure because of `capture_disjoint_fields`
    //~| NOTE: drop(&(t));
        let _t = t.0;
        let _t = &t.1;
    };

    c();
}

struct S;
impl Drop for S {
    fn drop(&mut self) {
    }
}

struct T(S, S);
struct U(T, T);

// Test precise analysis for the lint works with paths longer than one.
fn test_precise_analysis_long_path_missing() {
    let u = U(T(S, S), T(S, S));

    let c = || {
    //~^ERROR: drop order affected for closure because of `capture_disjoint_fields`
    //~| NOTE: drop(&(u));
        let _x = u.0.0;
        let _x = u.0.1;
        let _x = u.1.0;
    };

    c();
}

fn main() {
    test_precise_analysis_parent_root_impl_drop_not_captured();
    test_precise_analysis_drop_paths_not_captured_by_move();
    test_precise_analysis_long_path_missing();
}
