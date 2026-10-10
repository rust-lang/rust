//@revisions: gca_min_const_items gca_const_items regular
#![crate_type = "lib"]
#![cfg_attr(gca_min_const_items, feature(gca_min_const_items))]
#![cfg_attr(gca_const_items, feature(gca_min_const_items))]

fn example<const N: usize>() {}

fn other() {
    example::<[usize; 3]>();
    //~^ ERROR type provided when a const
    example::<[usize; 4 + 5]>();
    //~^ ERROR type provided when a const
}
