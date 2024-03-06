//@ revisions: current next
//@[next] compile-flags: -Znext-solver

struct Wrapper<T: ?Sized>(T);

trait A: B {}
trait B {}

fn test<'a>(x: Box<Wrapper<dyn A + 'a>>) -> Box<Wrapper<dyn B + 'a>> {
    x
    //~^ ERROR cannot cast `dyn A` to `dyn B`, trait upcasting coercion is experimental
}

fn main() {}
