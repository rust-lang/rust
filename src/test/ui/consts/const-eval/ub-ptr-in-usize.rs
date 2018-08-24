// compile-pass

union Foo {
    a: &'static u8,
    b: usize,
}

// a usize's value may be a pointer, that's fine
const PTR_AS_USIZE: usize = unsafe { Foo { a: &1 }.b};

fn main() {
}
