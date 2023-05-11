fn foo<T>() {
    struct Foo {
        x: T, //~ ERROR can't use generic parameters from outer function
    }

    impl<T> Drop for Foo<T> {
        //~^ ERROR struct takes 0 generic arguments but 1 generic argument
        fn drop(&mut self) {}
    }
}
fn main() {}
