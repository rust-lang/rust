fn main() {
    let x : (impl Copy,) = (true,);
    //~^ `impl Trait` isn't allowed within variable binding
}
