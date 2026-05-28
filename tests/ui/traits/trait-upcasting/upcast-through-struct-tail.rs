//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

struct Wrapper<T: ?Sized>(T);

trait A: B {}
trait B {}

fn test<'a>(x: Box<Wrapper<dyn A + 'a>>) -> Box<Wrapper<dyn B + 'a>> {
    x
}

fn main() {}
