fn main() {
    let x : (impl Copy,) = (true,);
    //~^ ERROR `impl Trait` is not allowed in the type of variable bindings
}
