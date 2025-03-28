//@ revisions: old-solver next-solver
//@[next-solver] compile-flags: -Znext-solver
//@ run-pass

fn foo(_: impl Into<f32>) {}

fn main() {
    foo(1.0);
}
