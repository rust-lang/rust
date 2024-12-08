// https://github.com/rust-lang/rust/issues/51236
#![crate_name="foo"]

use std::marker::PhantomData;

pub mod traits {
    pub trait Owned<'a> {
        type Reader;
    }
}

//@ has foo/struct.Owned.html
//@ has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// "impl<T> Send for Owned<T>where <T as Owned<'static>>::Reader: Send"
pub struct Owned<T> where T: for<'a> ::traits::Owned<'a> {
    marker: PhantomData<<T as ::traits::Owned<'static>>::Reader>,
}
