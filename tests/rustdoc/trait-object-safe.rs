#![crate_name = "foo"]

// @has 'foo/trait.Unsafe.html'
// @has - '//*[@class="object-safety-info"]' 'This trait is not object safe.'
// @has - '//*[@id="object-safety"]' 'Object Safety'
pub trait Unsafe {
    fn foo() -> Self;
}

// @has 'foo/trait.Unsafe2.html'
// @has - '//*[@class="object-safety-info"]' 'This trait is not object safe.'
// @has - '//*[@id="object-safety"]' 'Object Safety'
pub trait Unsafe2<T> {
    fn foo(i: T);
}

// @has 'foo/trait.Safe.html'
// @!has - '//*[@class="object-safety-info"]' ''
// @!has - '//*[@id="object-safety"]' ''
pub trait Safe {
    fn foo(&self);
}

// @has 'foo/struct.Foo.html'
// @!has - '//*[@class="object-safety-info"]' ''
// @!has - '//*[@id="object-safety"]' ''
pub struct Foo;
