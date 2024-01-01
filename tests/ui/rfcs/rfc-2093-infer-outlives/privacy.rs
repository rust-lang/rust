// Test that we do not get a privacy error here.  Initially, we did,
// because we inferred an outlives predciate of `<Foo<'a> as
// Private>::Out: 'a`, but the private trait is -- well -- private,
// and hence it was not something that a pub trait could refer to.
//
// build-pass



pub struct Foo<'a> {
    field: Option<&'a <Foo<'a> as Private>::Out>
}

trait Private {
    type Out: ?Sized;
}

impl<T: ?Sized> Private for T { type Out = Self; }

fn main() { }
