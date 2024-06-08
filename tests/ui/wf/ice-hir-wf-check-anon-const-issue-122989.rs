// Regression test for ICE #122989
trait Foo<const N: Bar<2>> {
    //~^ ERROR cycle detected when computing predicates of `Foo`
    fn func() {
    }
}

trait Bar<const M: Foo<2>> {}

fn main() {}
