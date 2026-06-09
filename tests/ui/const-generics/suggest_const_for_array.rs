#![crate_type = "lib"]

fn example<const N: usize>() {}

fn other() {
    example::<[usize; 3]>();
    //~^ ERROR type provided when a const
    example::<[usize; 4 + 5]>();
    //~^ ERROR type provided when a const
}
