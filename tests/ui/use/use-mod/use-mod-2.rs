mod foo {
    use self::{self};
    //~^ ERROR unresolved import `self` [E0432]
    //~| NOTE no `self` in the root

    use super::{self};
    //~^ ERROR unresolved import `super` [E0432]
    //~| NOTE no `super` in the root
}

fn main() {}
