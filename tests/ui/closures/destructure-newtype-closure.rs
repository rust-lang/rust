//! Regression test for https://github.com/rust-lang/rust/issues/20174

//@ check-pass
struct GradFn<F: Fn() -> usize>(F);

fn main() {
    let GradFn(x_squared) : GradFn<_> = GradFn(|| -> usize { 2 });
    let _  = x_squared();
}
