//@ run-pass
//@ run-rustfix
//@ rustfix-only-machine-applicable

#[allow(unused_must_use, unused_allocation)]
fn main() {
    let small = [1, 2];
    let big = [0u8; 33];

    // Expressions that should trigger the lint
    small.into_iter();
    //~^ WARNING this method call resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this changes meaning
    [1, 2].into_iter();
    //~^ WARNING this method call resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this changes meaning
    big.into_iter();
    //~^ WARNING this method call resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this changes meaning
    [0u8; 33].into_iter();
    //~^ WARNING this method call resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this changes meaning

    Box::new(small).into_iter();
    //~^ WARNING this method call resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this changes meaning
    Box::new([1, 2]).into_iter();
    //~^ WARNING this method call resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this changes meaning
    Box::new(big).into_iter();
    //~^ WARNING this method call resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this changes meaning
    Box::new([0u8; 33]).into_iter();
    //~^ WARNING this method call resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this changes meaning

    Box::new(Box::new(small)).into_iter();
    //~^ WARNING this method call resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this changes meaning
    Box::new(Box::new([1, 2])).into_iter();
    //~^ WARNING this method call resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this changes meaning
    Box::new(Box::new(big)).into_iter();
    //~^ WARNING this method call resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this changes meaning
    Box::new(Box::new([0u8; 33])).into_iter();
    //~^ WARNING this method call resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this changes meaning

    // Expressions that should not
    (&[1, 2]).into_iter();
    (&small).into_iter();
    (&[0u8; 33]).into_iter();
    (&big).into_iter();

    for _ in &[1, 2] {}
    (&small as &[_]).into_iter();
    small[..].into_iter();
    std::iter::IntoIterator::into_iter(&[1, 2]);

    #[allow(array_into_iter)]
    [0, 1].into_iter();
}
