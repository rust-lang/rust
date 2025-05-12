//@ edition:2015
//@ check-pass
// issue: 114664

fn ice() -> impl AsRef<Fn(&())> {
    //~^ WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    Foo
}

struct Foo;
impl AsRef<dyn Fn(&())> for Foo {
    fn as_ref(&self) -> &(dyn for<'a> Fn(&'a ()) + 'static) {
        todo!()
    }
}

pub fn main() {}
