// ignore-tidy-linelength

fn foo<T>() {
    struct Foo {
        x: T, //~ ERROR can't use generic parameters from outer function
    }

    impl<T> Drop for Foo<T> {
        //~^ ERROR wrong number of type arguments
        //~| ERROR the type parameter `T` is not constrained by the impl trait, self type, or predicates
        fn drop(&mut self) {}
    }
}
fn main() { }
