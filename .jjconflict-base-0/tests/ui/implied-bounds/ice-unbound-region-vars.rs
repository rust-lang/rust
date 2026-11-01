// Because of #109628, we can have unbounded region vars in implied bounds.
// Make sure we don't ICE in this case!
//
//@ check-pass

pub trait MapAccess {
    type Error;
    fn next_key_seed(&mut self) -> Option<Self::Error>;
}

struct Access<'a> {
    _marker: std::marker::PhantomData<&'a ()>,
}

// implied_bounds(Option<Self::Error>) = ['?1: 'a, ]
// where '?1 is a fresh region var.
impl<'a, 'b: 'a> MapAccess for Access<'a> {
    type Error = ();
    fn next_key_seed(&mut self) -> Option<Self::Error> {
        unimplemented!()
    }
}

fn main() {}
