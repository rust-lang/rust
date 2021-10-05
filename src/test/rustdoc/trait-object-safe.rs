#![crate_name = "foo"]

// @has 'foo/trait.Safe.html'
// @has - '//*[@class="obj-info"]' 'This trait is object safe.'
pub trait Safe {
    fn foo(&self);
}

// @has 'foo/trait.Unsafe.html'
// @has - '//*[@class="obj-info"]' 'This trait is not object safe.'
pub trait Unsafe {
    fn foo() -> Self;
}

// @has 'foo/trait.Unsafe2.html'
// @has - '//*[@class="obj-info"]' 'This trait is not object safe.'
pub trait Unsafe2<T> {
    fn foo(i: T);
}

// @has 'foo/struct.Foo.html'
// @!has - '//*[@class="obj-info"]'
pub struct Foo;
