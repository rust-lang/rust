fn main() {
    let x : (impl Copy,) = (true,);
    //~^ `impl Trait` not allowed outside of function and inherent method return types
}
