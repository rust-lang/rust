mod foo {
    use self::{self};
    //~^ ERROR imports need to be explicitly named

    use super::{self};
    //~^ ERROR imports need to be explicitly named
}

fn main() {}
