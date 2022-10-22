fn main() {
    0.clone::<'a>();
    //~^ ERROR use of undeclared lifetime name `'a`
    //~| ERROR cannot specify lifetime arguments explicitly if late bound
}
