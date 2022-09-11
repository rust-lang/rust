trait Foo {
    fn bar() -> impl Sized; //~ ERROR `impl Trait` only allowed in function and inherent method return types, not in trait method return
}

fn main() {}
