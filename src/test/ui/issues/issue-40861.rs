fn f(_: &[f32]) {}

fn main() {
    ()[f(&[1.0])];
    //~^ ERROR cannot index into a value of type `()`
}
