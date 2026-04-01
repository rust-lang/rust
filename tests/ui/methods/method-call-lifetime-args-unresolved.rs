fn main() {
    0.clone::<'a>();
    //~^ ERROR use of undeclared lifetime name `'a`
    //~| WARN cannot specify lifetime arguments explicitly if late bound
    //~| WARN this was previously accepted by the compiler
}
