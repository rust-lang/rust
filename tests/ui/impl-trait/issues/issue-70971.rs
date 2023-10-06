fn main() {
    let x : (impl Copy,) = (true,);
    //~^ `impl Trait` only allowed in function and inherent method argument and return types
}
