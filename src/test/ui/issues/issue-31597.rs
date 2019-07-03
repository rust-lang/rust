// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
trait Make {
    type Out;

    fn make() -> Self::Out;
}

impl Make for () {
    type Out = ();

    fn make() -> Self::Out {}
}

// Also make sure we don't hit an ICE when the projection can't be known
fn f<T: Make>() -> <T as Make>::Out { loop {} }

// ...and that it works with a blanket impl
trait Tr {
    type Assoc;
}

impl<T: Make> Tr for T {
    type Assoc = ();
}

fn g<T: Make>() -> <T as Tr>::Assoc { }

fn main() {}
