trait WorksOnDefault {
    fn do_something() {}
}

impl<T: Default> WorksOnDefault for T {}
//~^ NOTE the following trait bounds were not satisfied
//~| NOTE unsatisfied trait bound introduced here

trait Foo {}

trait WorksOnFoo {
    fn do_be_do() {}
}

impl<T: Foo> WorksOnFoo for T {}
//~^ NOTE the following trait bounds were not satisfied
//~| NOTE unsatisfied trait bound introduced here

impl<A: Foo, B: Foo, C: Foo> Foo for (A, B, C) {}
//~^ NOTE `Foo` is implemented for `(i32, u32, String)`
impl Foo for i32 {}
impl Foo for &i32 {}
impl Foo for u32 {}
impl Foo for String {}

fn main() {
    let _success = <(i32, u32, String)>::do_something();
    let _failure = <(i32, &u32, String)>::do_something(); //~ ERROR E0599
    //~^ NOTE `Default` is implemented for `(i32, u32, String)`
    //~| NOTE function or associated item cannot be called on
    let _success = <(i32, u32, String)>::do_be_do();
    let _failure = <(i32, &u32, String)>::do_be_do(); //~ ERROR E0599
    //~^ NOTE function or associated item cannot be called on
    let _success = <(i32, u32, String)>::default();
    let _failure = <(i32, &u32, String)>::default(); //~ ERROR E0599
    //~^ NOTE `Default` is implemented for `(i32, u32, String)`
    //~| NOTE function or associated item cannot be called on
    //~| NOTE the following trait bounds were not satisfied
}
