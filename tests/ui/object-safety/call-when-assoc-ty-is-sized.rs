// check-pass
// revisions: current next
//[next] compile-flags: -Ztrait-solver=next

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
