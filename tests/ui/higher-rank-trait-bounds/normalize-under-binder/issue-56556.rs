// check-pass

fn foo<T>(t: T) -> usize
where
    for<'a> &'a T: IntoIterator,
    for<'a> <&'a T as IntoIterator>::IntoIter: ExactSizeIterator,
{
    t.into_iter().len()
}

fn main() {
    foo::<Vec<u32>>(vec![]);
}

mod another {
    use std::ops::Deref;

    fn test<T, TDeref>()
    where
        T: Deref<Target = TDeref>,
        TDeref: ?Sized,
        for<'a> &'a TDeref: IntoIterator,
        for<'a> <&'a TDeref as IntoIterator>::IntoIter: Clone,
    {
    }

    fn main() {
        test::<Vec<u8>, _>();
    }
}
