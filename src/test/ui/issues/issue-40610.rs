fn f(_: &[f32]) {}

fn main() {
    () + f(&[1.0]);
    //~^ ERROR binary operation `+` cannot be applied to type `()`
}
