//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[current] check-pass

trait Super<T> {}
trait Trait<T>: Super<T> + for<'hr> Super<&'hr ()> {}

fn foo<'a>(x: Box<dyn Trait<&'a ()>>) -> Box<dyn Super<&'a ()>> {
    x
    //[next]~^ ERROR type annotations needed: cannot satisfy `dyn Trait<&()>: Unsize<dyn Super<&()>>`
}

fn main() {}
