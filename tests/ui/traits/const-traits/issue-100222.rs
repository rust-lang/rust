//@ revisions: nn ny yn yy
//@ compile-flags: -Znext-solver
//@ check-pass

#![allow(incomplete_features)]
#![feature(const_trait_impl, associated_type_defaults)]

#[cfg(any(yn, yy))] pub const trait Index { type Output; }
#[cfg(not(any(yn, yy)))] pub trait Index { type Output; }

#[cfg(any(ny, yy))]
pub const trait IndexMut
where
    Self: Index,
{
    const C: <Self as Index>::Output;
    type Assoc = <Self as Index>::Output;
    fn foo(&mut self, x: <Self as Index>::Output) -> <Self as Index>::Output;
}

#[cfg(not(any(ny, yy)))]
pub trait IndexMut
where
    Self: Index,
{
    const C: <Self as Index>::Output;
    type Assoc = <Self as Index>::Output;
    fn foo(&mut self, x: <Self as Index>::Output) -> <Self as Index>::Output;
}

impl Index for () {
    type Output = ();
}

#[cfg(not(any(nn, yn)))]
impl const IndexMut for <() as Index>::Output {
    const C: <Self as Index>::Output = ();
    type Assoc = <Self as Index>::Output;
    fn foo(&mut self, x: <Self as Index>::Output) -> <Self as Index>::Output
    where
        <Self as Index>::Output:,
    {
    }
}

#[cfg(any(nn, yn))]
impl IndexMut for <() as Index>::Output {
    const C: <Self as Index>::Output = ();
    type Assoc = <Self as Index>::Output;
    fn foo(&mut self, x: <Self as Index>::Output) -> <Self as Index>::Output
    where
        <Self as Index>::Output:,
    {
    }
}

const C: <() as Index>::Output = ();

fn main() {}
