// Checks that we only suggest borrowing if &T actually implements the trait.

trait Tr {}
impl Tr for &f32 {}
fn bar<T: Tr>(t: T) {}

fn main() {
    let a = 0i32;
    let b = 0.0f32;
    bar(a); //~ ERROR: the trait bound `i32: Tr` is not satisfied [E0277]
    bar(b); //~ ERROR: the trait bound `f32: Tr` is not satisfied [E0277]
}
