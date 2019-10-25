// Test that an `impl Trait` type that expands to itself is an error.

fn test() -> impl Sized { //~ ERROR E0720
    test()
}

fn main() {}
