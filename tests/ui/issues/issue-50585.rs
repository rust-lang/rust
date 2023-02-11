fn main() {
    |y: Vec<[(); for x in 0..2 {}]>| {};
    //~^ ERROR mismatched types
    //~| ERROR `for` is not allowed in a `const`
}
