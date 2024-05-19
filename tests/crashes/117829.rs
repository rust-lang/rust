//@ known-bug: #117829
auto trait Z<'a, T: ?Sized>
where
    T: Z<'a, u16>,

    for<'b> <T as Z<'b, u16>>::W: Clone,
{
    type W: ?Sized;
}
