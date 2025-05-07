//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Foo {
    type Bar<'a>
    where
        Self: Sized;

    fn test(&self);
}

impl Foo for () {
    type Bar<'a> = () where Self: Sized;

    fn test(&self) {}
}

fn test(x: &dyn Foo) {
    x.test();
}

fn main() {
    test(&());
}
