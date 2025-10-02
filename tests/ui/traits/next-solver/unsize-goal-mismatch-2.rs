//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Super<T> {}
trait Trait<T>: Super<T> + for<'hr> Super<&'hr ()> {}

fn foo<'a>(x: Box<dyn Trait<&'a ()>>) -> Box<dyn Super<&'a ()>> {
    x
}

fn main() {}
