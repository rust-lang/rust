// compile-flags: -Z chalk

trait Foo { }

impl Foo for i32 { }

impl Foo for u32 { }

fn gimme<F: Foo>() { }

// Note: this also tests that `std::process::Termination` is implemented for `()`.
fn main() {
    gimme::<i32>();
    gimme::<u32>();
    gimme::<f32>(); //~ERROR the trait bound `f32: Foo` is not satisfied
}
