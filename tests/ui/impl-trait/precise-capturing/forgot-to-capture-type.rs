fn type_param<T>() -> impl Sized + use<> {}
//~^ ERROR `impl Trait` must mention all type parameters in scope

trait Foo {
    fn bar() -> impl Sized + use<>;
    //~^ ERROR `impl Trait` must mention the `Self` type of the trait
}

fn main() {}
