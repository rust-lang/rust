fn foo<T>() {
    struct foo {
        x: T, //~ ERROR can't use type parameters from outer function
    }

    impl<T> Drop for foo<T> {
        //~^ ERROR wrong number of type arguments
        fn drop(&mut self) {}
    }
}
fn main() { }
