// Regression test for issue #80062

fn foo<Foo>() -> Foo { todo!() }

fn bar<T>() {
    let _: [u8; foo::<T>()];
    //~^   ERROR the size for values of type `T` cannot be known at compilation time
    //~^^  ERROR mismatched types
    //~^^^ ERROR the size for values of type `T` cannot be known at compilation time
}

fn main() {}
