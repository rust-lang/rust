fn main() {

    let f;

    f = Box::new(f);
    //~^ ERROR mismatched types
    //~| cyclic type of infinite size
}
