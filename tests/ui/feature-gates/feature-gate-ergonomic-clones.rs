use std::clone::UseCloned;
//~^ ERROR use of unstable library feature `ergonomic_clones` [E0658]

fn ergonomic_clone(x: i32) -> i32 {
    x.use
    //~^ ERROR `.use` calls are experimental [E0658]
}

#[derive(Clone)]
struct Foo;

impl UseCloned for Foo {}
//~^ ERROR use of unstable library feature `ergonomic_clones` [E0658]

fn ergonomic_closure_clone() {
    let f1 = Foo;

    let f2 = use || {
        //~^ ERROR `.use` calls are experimental [E0658]
        f1
    };

    let f3 = use || {
        //~^ ERROR `.use` calls are experimental [E0658]
        f1
    };
}

fn main() {}
