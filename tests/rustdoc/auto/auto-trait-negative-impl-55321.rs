// https://github.com/rust-lang/rust/issues/55321
#![crate_name="foo"]

#![feature(negative_impls)]

//@ has foo/struct.A.html
//@ has - '//*[@id="trait-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// "impl !Send for A"
//@ has - '//*[@id="trait-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// "impl !Sync for A"
pub struct A();

impl !Send for A {}
impl !Sync for A {}

//@ has foo/struct.B.html
//@ has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// "impl<T> !Send for B<T>"
//@ has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// "impl<T> !Sync for B<T>"
pub struct B<T: ?Sized>(A, Box<T>);
