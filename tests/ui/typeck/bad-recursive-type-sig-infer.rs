fn a() -> _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    &a
}

fn b() -> _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    &a
}

fn main() {}
