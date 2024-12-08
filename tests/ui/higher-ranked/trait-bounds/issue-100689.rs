//@ check-pass

struct Foo<'a> {
    foo: &'a mut usize,
}

trait Bar<'a> {
    type FooRef<'b>
    where
        'a: 'b;
    fn uwu(foo: Foo<'a>, f: impl for<'b> FnMut(Self::FooRef<'b>));
}
impl<'a> Bar<'a> for () {
    type FooRef<'b>
    =
        &'b Foo<'a>
    where
        'a : 'b,
    ;

    fn uwu(
        foo: Foo<'a>,
        mut f: impl for<'b> FnMut(&'b Foo<'a>), //relevant part
    ) {
        f(&foo);
    }
}

fn main() {}
