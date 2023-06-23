// compile-flags: -Ztrait-solver=next

// Makes sure that alias bounds are not unsound!

#![feature(trivial_bounds)]

trait Foo {
    type Item: Copy
    where
        <Self as Foo>::Item: Copy;

    fn copy_me(x: &Self::Item) -> Self::Item {
        *x
    }
}

impl Foo for () {
    type Item = String where String: Copy;
}

fn main() {
    let x = String::from("hello, world");
    drop(<() as Foo>::copy_me(&x));
    //~^ ERROR `<() as Foo>::Item: Copy` is not satisfied
    //~| ERROR `<() as Foo>::Item` is not well-formed
    //~| ERROR `<() as Foo>::Item` is not well-formed
    //~| ERROR `<() as Foo>::Item` is not well-formed
    println!("{x}");
}
