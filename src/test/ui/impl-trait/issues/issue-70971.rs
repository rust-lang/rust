fn main() {
    let x : (impl Copy,) = (true,);
    //~^ `impl Trait` not allowed within variable binding
}
