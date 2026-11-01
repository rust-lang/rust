// Unresolved fields are not copy, but also shouldn't report an extra E0740.

pub union Foo {
    x: *const Missing,
    //~^ ERROR cannot find type `Missing` in this scope
}

fn main() {}
