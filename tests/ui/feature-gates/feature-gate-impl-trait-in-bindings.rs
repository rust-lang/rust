fn main() {
    let x: impl Sized = ();
    //~^ ERROR `impl Trait` is not allowed in the type of variable bindings
}
