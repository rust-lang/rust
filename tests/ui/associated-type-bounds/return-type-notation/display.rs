#![feature(return_type_notation)]

trait Trait {}
fn needs_trait(_: impl Trait) {}

trait Assoc {
    fn method() -> impl Sized;
    fn method_with_lt() -> impl Sized;
    fn method_with_ty<T>() -> impl Sized;
    fn method_with_ct<const N: usize>() -> impl Sized;
}

fn foo<T: Assoc>(t: T) {
    needs_trait(T::method());
    //~^ ERROR the trait bound
    needs_trait(T::method_with_lt());
    //~^ ERROR the trait bound
    needs_trait(T::method_with_ty());
    //~^ ERROR the trait bound
    needs_trait(T::method_with_ct());
    //~^ ERROR the trait bound
}

fn main() {}
