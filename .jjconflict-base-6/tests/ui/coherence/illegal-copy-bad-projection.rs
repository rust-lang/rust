trait AsPtr {
    type Ptr;
}

impl AsPtr for () {
    type Ptr = *const void;
    //~^ ERROR cannot find type `void` in this scope
}

#[derive(Copy, Clone)]
struct Foo {
    p: <() as AsPtr>::Ptr,
    // Do not report a "`Copy` cannot be implemented" here.
}

fn main() {}
