//@ check-pass
#![allow(unused_variables)]
// Issue #12856: a lifetime formal binding introduced by a generic fn
// should not upset the variance inference for actual occurrences of
// that lifetime in type expressions.


pub trait HasLife<'a> {
    fn dummy(&'a self) { } // just to induce a variance on 'a
}

trait UseLife01 {
    fn refs<'a, H: HasLife<'a>>(&'a self) -> H;
}

trait UseLife02 {
    fn refs<'a, T: 'a, H: HasType<&'a T>>(&'a self) -> H;
}


pub trait HasType<T>
{
    fn dummy(&self, t: T) -> T { panic!() }
}


trait UseLife03<T> {
    fn refs<'a, H: HasType<&'a T>>(&'a self) -> H where T: 'a;
}


// (The functions below were not actually a problem observed during
// fixing of #12856; they just seem like natural tests to put in to
// cover a couple more points in the testing space)

pub fn top_refs_1<'a, H: HasLife<'a>>(_s: &'a ()) -> H {
    unimplemented!()
}

pub fn top_refs_2<'a, T: 'a, H: HasType<&'a T>>(_s: &'a ()) -> H {
    unimplemented!()
}

pub fn main() {}
