//@ run-rustfix

#![deny(rust_2021_incompatible_closure_captures)]
//~^ NOTE: the lint level is defined here

#[derive(Debug)]
struct Foo(i32);
impl Drop for Foo {
    fn drop(&mut self) {
        println!("{:?} dropped", self.0);
    }
}

struct ConstainsDropField(Foo, Foo);

// Test that lint is triggered if a path that implements Drop is not captured by move
fn test_precise_analysis_drop_paths_not_captured_by_move() {
    let t = ConstainsDropField(Foo(10), Foo(20));

    let c = || {
        //~^ ERROR: drop order
        //~| NOTE: for more information, see
        //~| HELP: add a dummy let to cause `t` to be fully captured
        let _t = t.0;
        //~^ NOTE: in Rust 2018, this closure captures all of `t`, but in Rust 2021, it will only capture `t.0`
        let _t = &t.1;
    };

    c();
}
//~^ NOTE: in Rust 2018, `t` is dropped here, but in Rust 2021, only `t.0` will be dropped here as part of the closure

struct S;
impl Drop for S {
    fn drop(&mut self) {}
}

struct T(S, S);
struct U(T, T);

// Test precise analysis for the lint works with paths longer than one.
fn test_precise_analysis_long_path_missing() {
    let u = U(T(S, S), T(S, S));

    let c = || {
        //~^ ERROR: drop order
        //~| NOTE: for more information, see
        //~| HELP: add a dummy let to cause `u` to be fully captured
        let _x = u.0.0;
        //~^ NOTE: in Rust 2018, this closure captures all of `u`, but in Rust 2021, it will only capture `u.0.0`
        let _x = u.0.1;
        //~^ NOTE: in Rust 2018, this closure captures all of `u`, but in Rust 2021, it will only capture `u.0.1`
        let _x = u.1.0;
        //~^ NOTE: in Rust 2018, this closure captures all of `u`, but in Rust 2021, it will only capture `u.1.0`
    };

    c();
}
//~^ NOTE: in Rust 2018, `u` is dropped here, but in Rust 2021, only `u.0.0` will be dropped here as part of the closure
//~| NOTE: in Rust 2018, `u` is dropped here, but in Rust 2021, only `u.0.1` will be dropped here as part of the closure
//~| NOTE: in Rust 2018, `u` is dropped here, but in Rust 2021, only `u.1.0` will be dropped here as part of the closure


fn main() {
    test_precise_analysis_drop_paths_not_captured_by_move();
    test_precise_analysis_long_path_missing();
}
