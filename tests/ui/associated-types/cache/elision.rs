// Check that you are allowed to implement using elision but write
// trait without elision (a bug in this cropped up during
// bootstrapping, so this is a regression test).

//@ check-pass

pub struct SplitWhitespace<'a> {
    x: &'a u8
}

pub trait UnicodeStr {
    fn split_whitespace<'a>(&'a self) -> SplitWhitespace<'a>;
}

impl UnicodeStr for str {
    #[inline]
    fn split_whitespace(&self) -> SplitWhitespace<'_> {
        unimplemented!()
    }
}


fn main() { }
