fn main() {
    0.clone::<'a>(); //~ ERROR use of undeclared lifetime name `'a`
    //~^ WARNING cannot specify lifetime arguments
    //~| WARNING this was previously accepted
}
