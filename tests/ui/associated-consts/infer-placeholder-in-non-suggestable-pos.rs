trait Trait {
    const ASSOC: i32;
}

impl Trait for () {
    const ASSOC: &dyn Fn(_) = 1i32;
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for associated constants
    //~| WARN `&` without an explicit lifetime name cannot be used here
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

fn main() {}
