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
    //~^ ERROR overflow evaluating the requirement `<() as Foo>::Item: Copy`
}

fn main() {
    let x = String::from("hello, world");
    drop(<() as Foo>::copy_me(&x));
    //~^ ERROR overflow evaluating the requirement `<() as Foo>::Item: Sized`
    //~| ERROR overflow evaluating the requirement `<() as Foo>::Item == _`
    //~| ERROR overflow evaluating the requirement `<() as Foo>::Item well-formed`
    //~| ERROR overflow evaluating the requirement `String <: <() as Foo>::Item`
    //~| ERROR overflow evaluating the requirement `&<() as Foo>::Item well-formed`
    //~| ERROR type annotations needed
    println!("{x}");
}
