fn foo<T>() {
    struct Foo {
        x: T, //~ ERROR can't use generic parameters from outer function
    }

    impl<T> Drop for Foo<T> {
        //~^ ERROR wrong number of type arguments
        fn drop(&mut self) {}
    }
}
fn main() { }
