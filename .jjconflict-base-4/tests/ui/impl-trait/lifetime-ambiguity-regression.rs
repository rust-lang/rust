//! This test shows a situation where through subtle compiler changes we can
//! suddenly infer a different lifetime in the hidden type, and thus not meet
//! the opaque type bounds anymore. In this case `'a` and `'b` are equal, so
//! picking either is fine, but then we'll fail an identity check of the hidden
//! type and the expected hidden type.

//@ check-pass

fn test<'a: 'b, 'b: 'a>() -> impl IntoIterator<Item = (&'a u8, impl Into<(&'b u8, &'a u8)>)> {
    None::<(_, (_, _))>
}

fn main() {}
