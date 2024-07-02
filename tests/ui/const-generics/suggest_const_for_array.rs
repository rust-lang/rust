#![crate_type = "lib"]

fn example<const N: usize>() {}

fn other() {
    example::<[usize; 3]>();
    //~^ ERROR type provided when a const
    //~| ERROR type annotations needed
    example::<[usize; 4 + 5]>();
    //~^ ERROR type provided when a const
    //~| ERROR type annotations needed
}
