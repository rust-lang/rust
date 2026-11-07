use std::clone::UseCloned;
//~^ ERROR use of unstable library feature `ergonomic_clones` [E0658]

fn ergonomic_clone(x: i32) -> i32 {
    x.use
    //~^ ERROR ergonomic clones are experimental [E0658]
}

#[derive(Clone)]
struct Foo;

fn foo<T: UseCloned>() {}
//~^ ERROR use of unstable library feature `ergonomic_clones` [E0658]

impl UseCloned for Foo {}
//~^ ERROR use of unstable library feature `ergonomic_clones` [E0658]

fn main() {}
