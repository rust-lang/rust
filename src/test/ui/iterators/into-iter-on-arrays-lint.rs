// run-pass
// run-rustfix

fn main() {
    let small = [1, 2];
    let big = [0u8; 33];

    // Expressions that should trigger the lint
    small.into_iter();
    //~^ WARNING this method call currently resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this was previously accepted by the compiler but is being phased out
    [1, 2].into_iter();
    //~^ WARNING this method call currently resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this was previously accepted by the compiler but is being phased out
    big.into_iter();
    //~^ WARNING this method call currently resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this was previously accepted by the compiler but is being phased out
    [0u8; 33].into_iter();
    //~^ WARNING this method call currently resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this was previously accepted by the compiler but is being phased out

    Box::new(small).into_iter();
    //~^ WARNING this method call currently resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this was previously accepted by the compiler but is being phased out
    Box::new([1, 2]).into_iter();
    //~^ WARNING this method call currently resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this was previously accepted by the compiler but is being phased out
    Box::new(big).into_iter();
    //~^ WARNING this method call currently resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this was previously accepted by the compiler but is being phased out
    Box::new([0u8; 33]).into_iter();
    //~^ WARNING this method call currently resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this was previously accepted by the compiler but is being phased out

    Box::new(Box::new(small)).into_iter();
    //~^ WARNING this method call currently resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this was previously accepted by the compiler but is being phased out
    Box::new(Box::new([1, 2])).into_iter();
    //~^ WARNING this method call currently resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this was previously accepted by the compiler but is being phased out
    Box::new(Box::new(big)).into_iter();
    //~^ WARNING this method call currently resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this was previously accepted by the compiler but is being phased out
    Box::new(Box::new([0u8; 33])).into_iter();
    //~^ WARNING this method call currently resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this was previously accepted by the compiler but is being phased out

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
