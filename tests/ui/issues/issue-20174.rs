//@ run-pass
struct GradFn<F: Fn() -> usize>(F);

fn main() {
    let GradFn(x_squared) : GradFn<_> = GradFn(|| -> usize { 2 });
    let _  = x_squared();
}
