trait Foo {
    fn test() -> impl Sized;
}

impl<'a, T> Foo for T {
    //~^ ERROR the lifetime parameter `'a` is not constrained by the impl trait, self type, or predicates

    fn test() -> &'a () {
        //~^ WARN: impl trait in impl method signature does not match trait method signature
        &()
    }
}

fn main() {}
