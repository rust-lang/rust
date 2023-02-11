#![crate_name = "foo"]

#![feature(rustdoc_internals)]

// @has foo/index.html
// @has - '//h2[@id="primitives"]' 'Primitive Types'
// @has - '//a[@href="primitive.reference.html"]' 'reference'
// @has - '//div[@class="sidebar-elems"]//li/a' 'Primitive Types'
// @has - '//div[@class="sidebar-elems"]//li/a/@href' '#primitives'
// @has foo/primitive.reference.html
// @has - '//a[@class="primitive"]' 'reference'
// @has - '//h1' 'Primitive Type reference'
// @has - '//section[@id="main-content"]//div[@class="docblock"]//p' 'this is a test!'

// There should be only one implementation listed.
// @count - '//*[@class="impl"]' 1
// @has - '//*[@id="impl-Foo%3C%26A%3E-for-%26B"]/*[@class="code-header"]' \
//        'impl<A, B> Foo<&A> for &B'
#[doc(primitive = "reference")]
/// this is a test!
mod reference {}

pub struct Bar;

// This implementation should **not** show up.
impl<T> From<&T> for Bar {
    fn from(s: &T) -> Self {
        Bar
    }
}

pub trait Foo<T> {
    fn stuff(&self, other: &T) {}
}

// This implementation should show up.
impl<A, B> Foo<&A> for &B {}
