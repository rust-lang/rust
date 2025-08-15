//@ revisions: old-solver next-solver
//@[next-solver] compile-flags: -Znext-solver
//@ run-pass
//@ run-rustfix

fn foo(_: impl Into<f32>) {}

fn main() {
    foo(1.0);
    //~^ WARN falling back to `f32`
    //~| WARN this was previously accepted
    foo(-(2.5));
    //~^ WARN falling back to `f32`
    //~| WARN this was previously accepted
    foo(1e5);
    //~^ WARN falling back to `f32`
    //~| WARN this was previously accepted
    foo(4f32); // no warning
    let x = -4.0;
    //~^ WARN falling back to `f32`
    //~| WARN this was previously accepted
    foo(x);
}
