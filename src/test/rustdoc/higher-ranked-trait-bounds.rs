#![crate_name = "foo"]

trait A<'x> {}

// @has foo/fn.test1.html
// @has - '//pre' "pub fn test1<T>() where for<'a> &'a T: Iterator,"
pub fn test1<T>()
where
    for<'a> &'a T: Iterator,
{
}

// @has foo/fn.test2.html
// @has - '//pre' "pub fn test2<T>() where for<'a, 'b> &'a T: A<'b>,"
pub fn test2<T>()
where
    for<'a, 'b> &'a T: A<'b>,
{
}

// @has foo/fn.test3.html
// @has - '//pre' "pub fn test3<F>() where F: for<'a, 'b> Fn(&'a u8, &'b u8),"
pub fn test3<F>()
where
    F: for<'a, 'b> Fn(&'a u8, &'b u8),
{
}

// @has foo/struct.Foo.html
pub struct Foo<'a> {
    _x: &'a u8,
}

impl<'a> Foo<'a> {
    // @has - '//code' "pub fn bar<T>() where T: A<'a>,"
    pub fn bar<T>()
    where
        T: A<'a>,
    {
    }
}
