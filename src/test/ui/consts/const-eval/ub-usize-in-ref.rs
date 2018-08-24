// compile-pass

union Foo {
    a: &'static u8,
    b: usize,
}

// This might point to an invalid address, but that's the user's problem
const USIZE_AS_STATIC_REF: &'static u8 = unsafe { Foo { b: 1337 }.a};

fn main() {
}
