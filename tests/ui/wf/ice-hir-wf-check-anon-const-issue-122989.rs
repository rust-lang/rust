// Regression test for ICE #122989
trait Foo<const N: Bar<2>> {
    //~^ WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    //~| ERROR cycle detected when computing type of `Foo::N`
    //~| ERROR cycle detected when computing type of `Foo::N`
    fn func() {}
}

trait Bar<const M: Foo<2>> {}
//~^ WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!

fn main() {}
