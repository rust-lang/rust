//@ known-bug: rust-lang/rust#125874
pub trait A {}

pub trait Mirror {
    type Assoc: ?Sized;
}
impl<T: ?Sized> Mirror for dyn A {
    type Assoc = T;
}

struct Bar {
    foo: <dyn A + 'static as Mirror>::Assoc,
}

pub fn main() {
    let strct = Bar { foo: 3 };

    match strct {
        Bar { foo: 1, .. } => {}
        _ => (),
    };
}
