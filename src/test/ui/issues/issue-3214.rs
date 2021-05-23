fn foo<T>() {
    struct Foo {
        x: T, //~ ERROR can't use generic parameters from outer function
    }

    impl<T> Drop for Foo<T> {
        //~^ ERROR this struct takes 0 generic arguments but 1 generic argument
        //~| ERROR the type parameter `T` is not constrained by the impl trait, self type, or predicates
        fn drop(&mut self) {}
    }
}
fn main() { }
