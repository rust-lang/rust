//@ check-pass
//
// Regression test from crater run for
// <https://github.com/rust-lang/rust/pull/98109>.


pub trait ElementLike {}

pub struct Located<T> where T: ElementLike {
    inner: T,
}

pub struct BlockElement<'a>(&'a str);

impl ElementLike for BlockElement<'_> {}


pub struct Page<'a> {
    /// Comprised of the elements within a page
    pub elements: Vec<Located<BlockElement<'a>>>,
}

impl<'a, __IdxT> std::ops::Index<__IdxT> for Page<'a> where
    Vec<Located<BlockElement<'a>>>: std::ops::Index<__IdxT>
{
    type Output =
        <Vec<Located<BlockElement<'a>>> as
        std::ops::Index<__IdxT>>::Output;

    #[inline]
    fn index(&self, idx: __IdxT) -> &Self::Output {
        <Vec<Located<BlockElement<'a>>> as
                std::ops::Index<__IdxT>>::index(&self.elements, idx)
    }
}

fn main() {}
