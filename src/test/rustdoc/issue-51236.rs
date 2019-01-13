use std::marker::PhantomData;

pub mod traits {
    pub trait Owned<'a> {
        type Reader;
    }
}

// @has issue_51236/struct.Owned.html
// @has - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]//code' "impl<T> Send for \
// Owned<T> where <T as Owned<'static>>::Reader: Send"
pub struct Owned<T> where T: for<'a> ::traits::Owned<'a> {
    marker: PhantomData<<T as ::traits::Owned<'static>>::Reader>,
}
