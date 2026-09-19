//@revisions: min_generic_const_args generic_const_args regular
#![crate_type = "lib"]
#![cfg_attr(min_generic_const_args, feature(min_generic_const_args))]
#![cfg_attr(generic_const_args, feature(min_generic_const_args))]

fn example<const N: usize>() {}

fn other() {
    example::<[usize; 3]>();
    //~^ ERROR type provided when a const
    example::<[usize; 4 + 5]>();
    //~^ ERROR type provided when a const
}
