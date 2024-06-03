trait Trait: Sized {
    fn function() {}
}
fn main() {
    <dyn Trait>::function();
    //~^ ERROR trait `Trait` cannot be made into an object
    //~| ERROR the size for values of type `dyn Trait` cannot be known at compilation time
}
