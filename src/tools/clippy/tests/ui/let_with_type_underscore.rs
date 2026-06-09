//@aux-build: proc_macros.rs
#![allow(unused)]
#![warn(clippy::let_with_type_underscore)]
#![allow(clippy::let_unit_value, clippy::needless_late_init)]

extern crate proc_macros;

fn func() -> &'static str {
    ""
}

#[rustfmt::skip]
fn main() {
    // Will lint
    let x: _ = 1;
    //~^ let_with_type_underscore
    let _: _ = 2;
    //~^ let_with_type_underscore
    let x: _ = func();
    //~^ let_with_type_underscore
    let x: _;
    //~^ let_with_type_underscore
    x = ();

    let x = 1; // Will not lint, Rust infers this to an integer before Clippy
    let x = func();
    let x: Vec<_> = Vec::<u32>::new();
    let x: [_; 1] = [1];
    let x : _ = 1;
    //~^ let_with_type_underscore

    // Do not lint from procedural macros
    proc_macros::with_span! {
        span
        let x: _ = ();
        // Late initialization
        let x: _;
        x = ();
        // Ensure weird formatting will not break it (hopefully)
        let x : _ = 1;
        let x
: _ = 1;
        let                   x :              
        _;
        x = ();
    };
}

fn issue15377() {
    let (a): _ = 0;
    //~^ let_with_type_underscore
    let ((a)): _ = 0;
    //~^ let_with_type_underscore
    let ((a,)): _ = (0,);
    //~^ let_with_type_underscore
    #[rustfmt::skip]
    let (   (a   )   ):  _ = 0;
    //~^ let_with_type_underscore
}
