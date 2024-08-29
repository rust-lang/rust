// Regression test for ICE #122989
trait Foo<const N: Bar<2>> {
    //~^ WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    //~| ERROR cycle detected when computing type of `Foo::N`
    //~| ERROR the trait `Foo` cannot be made into an object
    //~| ERROR `(dyn Bar<2> + 'static)` is forbidden as the type of a const generic parameter
    fn func() {}
}

trait Bar<const M: Foo<2>> {}
//~^ WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
//~| ERROR the trait `Foo` cannot be made into an object
//~| ERROR `(dyn Foo<2> + 'static)` is forbidden as the type of a const generic parameter

fn main() {}
