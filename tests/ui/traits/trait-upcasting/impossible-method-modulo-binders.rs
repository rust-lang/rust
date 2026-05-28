//@ build-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Foo {}
impl Foo for fn(&'static ()) {}

trait Bar {
    type Assoc: Default;
}
impl<T: Foo> Bar for T {
    type Assoc = usize;
}
impl Bar for fn(&()) {
    type Assoc = ();
}

fn needs_foo<T: Foo>() -> usize {
    needs_bar::<T>()
}

fn needs_bar<T: Bar>() -> <T as Bar>::Assoc {
    Default::default()
}

trait Evil<T> {
    fn bad(&self)
    where
        T: Foo,
    {
        needs_foo::<T>();
    }
}

impl Evil<fn(&())> for () {}

fn main() {
    let x: &dyn Evil<fn(&())> = &();
}
