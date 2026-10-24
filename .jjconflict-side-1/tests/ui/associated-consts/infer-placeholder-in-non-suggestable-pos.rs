trait Trait {
    const ASSOC: i32;
}

impl Trait for () {
    const ASSOC: &dyn Fn(_) = 1i32;
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for associated constants
}

fn main() {}
