trait Trait {
    fn method(&self) {}
}

impl<F: Fn() -> T, T> Trait for F {}

impl Trait for f32 {}

fn main() {
    || 10. .method();
    || .. .method();
    || 1.. .method();
}
