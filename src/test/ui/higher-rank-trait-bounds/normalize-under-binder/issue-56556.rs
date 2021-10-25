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
