fn for_trait<F>()
where
    for<'a> F: Fn(&'a str),
{
}
fn for_ref<F>()
where
    for<'a> &'a F: Debug,
{
}
fn for_parens<F>()
where
    for<'a> (&'a F): Fn(&'a str),
{
}
fn for_slice<F>()
where
    for<'a> [&'a F]: Eq,
{
}
fn for_qpath<T>(_t: &T)
where
    for<'a> <&'a T as Baz>::Foo: Iterator,
{
}
fn for_for_fn<T>()
where
    for<'a> for<'b> fn(&'a T, &'b T): Copy,
{
}
