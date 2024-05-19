mod foo {
    use self::{self};
    //~^ ERROR unresolved import `self` [E0432]
    //~| no `self` in the root

    use super::{self};
    //~^ ERROR unresolved import `super` [E0432]
    //~| no `super` in the root
}

fn main() {}
