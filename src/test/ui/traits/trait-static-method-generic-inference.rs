// Issue #3902. We are (at least currently) unable to infer `Self`
// based on `T`, even though there is only a single impl, because of
// the possibility of associated types and other things (basically: no
// constraints on `Self` here at all).

mod base {
    pub trait HasNew<T> {
        fn new() -> T;
        fn dummy(&self) { }
    }

    pub struct Foo {
        dummy: (),
    }

    impl HasNew<Foo> for Foo {
        fn new() -> Foo {
            Foo { dummy: () }
        }
    }
}

pub fn foo() {
    let _f: base::Foo = base::HasNew::new();
    //~^ ERROR type annotations required
}

fn main() { }
