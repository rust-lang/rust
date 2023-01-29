mod foo {
    pub trait MyTrait<'a> {
        type MyItem: ?Sized;
    }

    pub struct Inner<'a, Q, R: ?Sized> {
        field: Q,
        field3: &'a u8,
        my_foo: Foo<Q>,
        field2: R,
    }

    pub struct Outer<'a, T, K: ?Sized> {
        my_inner: Inner<'a, T, K>,
    }

    pub struct Foo<T> {
        myfield: T,
    }
}

// @has complex/struct.NotOuter.html
// @has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// "impl<'a, T, K: ?Sized> Send for Outer<'a, T, K>where K: for<'b> Fn((&'b bool, &'a u8)) \
// -> &'b i8, T: MyTrait<'a>, <T as MyTrait<'a>>::MyItem: Copy, 'a: 'static"

pub use foo::{Foo, Inner as NotInner, MyTrait as NotMyTrait, Outer as NotOuter};

unsafe impl<T> Send for Foo<T>
where
    T: NotMyTrait<'static>,
{
}

unsafe impl<'a, Q, R: ?Sized> Send for NotInner<'a, Q, R>
where
    Q: NotMyTrait<'a>,
    <Q as NotMyTrait<'a>>::MyItem: Copy,
    R: for<'b> Fn((&'b bool, &'a u8)) -> &'b i8,
    Foo<Q>: Send,
{
}
