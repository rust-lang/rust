// revisions: nn ny yn yy
// check-pass
#![feature(const_trait_impl, associated_type_defaults, const_mut_refs)]
#![feature(effects)]

#[cfg_attr(any(yn, yy), const_trait)]
pub trait Index {
    type Output;
}

#[cfg(yy)]
#[const_trait]
pub trait IndexMut: ~const Index {
    const C: <Self as Index>::Output;
    type Assoc = <Self as Index>::Output;
    fn foo(&mut self, x: <Self as Index>::Output) -> <Self as Index>::Output;
}

#[cfg(not(yy))]
#[cfg_attr(ny, const_trait)]
pub trait IndexMut: Index {
    const C: <Self as Index>::Output;
    type Assoc = <Self as Index>::Output;
    fn foo(&mut self, x: <Self as Index>::Output) -> <Self as Index>::Output;
}

#[cfg(yy)]
// FIXME: once we have `<Type as ~const Index>` syntax, remove this impl
// and merge the `IndexMut` declarations to only use the one without a `~const Index` bound.
impl const Index for () { type Output = (); }
#[cfg(not(yy))]
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
