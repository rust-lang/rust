//@ edition:2015..2021
fn main() {
    panic!(std::default::Default::default());
    //~^ ERROR type annotations needed
}
