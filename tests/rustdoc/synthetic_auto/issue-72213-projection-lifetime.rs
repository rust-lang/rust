// Regression test for issue #72213
// Tests that we don't ICE when we have projection predicates
// in our initial ParamEnv

pub struct Lines<'a, L>
where
    L: Iterator<Item = &'a ()>,
{
    words: std::iter::Peekable<Words<'a, L>>,
}

pub struct Words<'a, L> {
    _m: std::marker::PhantomData<&'a L>,
}

impl<'a, L> Iterator for Words<'a, L>
where
    L: Iterator<Item = &'a ()>,
{
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        unimplemented!()
    }
}
