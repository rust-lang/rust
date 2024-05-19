fn main() {
    Vec::<[(); 1 + for x in 0..1 {}]>::new();
    //~^ ERROR cannot add
    //~| ERROR `for` is not allowed in a `const`
}
