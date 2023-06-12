// revisions: nn ny yn yy
// check-pass
#![feature(const_trait_impl, associated_type_defaults, const_mut_refs)]

#[cfg_attr(any(yn, yy), const_trait)]
pub trait Index {
    type Output;
}

#[cfg_attr(any(ny, yy), const_trait)]
pub trait IndexMut where Self: Index {
    const C: <Self as Index>::Output;
    type Assoc = <Self as Index>::Output;
    fn foo(&mut self, x: <Self as Index>::Output) -> <Self as Index>::Output;
}

impl Index for () { type Output = (); }

#[cfg(not(any(nn, yn)))]
impl const IndexMut for <() as Index>::Output {
    const C: <Self as Index>::Output = ();
    type Assoc = <Self as Index>::Output;
    fn foo(&mut self, x: <Self as Index>::Output) -> <Self as Index>::Output
        where <Self as Index>::Output:,
    {}
}

#[cfg(any(nn, yn))]
impl IndexMut for <() as Index>::Output {
    const C: <Self as Index>::Output = ();
    type Assoc = <Self as Index>::Output;
    fn foo(&mut self, x: <Self as Index>::Output) -> <Self as Index>::Output
        where <Self as Index>::Output:,
    {}
}

const C: <() as Index>::Output = ();

fn main() {}
