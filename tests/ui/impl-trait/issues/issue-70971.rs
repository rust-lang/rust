fn main() {
    let x : (impl Copy,) = (true,);
    //~^ `impl Trait` is not allowed in the type of variable bindings
}
