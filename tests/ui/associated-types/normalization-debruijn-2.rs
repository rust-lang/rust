//@ build-pass
//@ edition:2018

// Regression test to ensure we handle debruijn indices correctly in projection
// normalization under binders. Found in crater run for #85499

use std::future::Future;
use std::pin::Pin;
pub enum Outcome<S, E> {
    Success(S),
    Failure(E),
}
pub struct Request<'r> {
    _marker: std::marker::PhantomData<&'r ()>,
}
pub trait FromRequest<'r>: Sized {
    type Error;
    fn from_request<'life0>(
        request: &'r Request<'life0>,
    ) -> Pin<Box<dyn Future<Output = Outcome<Self, Self::Error>>>>;
}
pub struct S<T> {
    _marker: std::marker::PhantomData<T>,
}
impl<'r, T: FromRequest<'r>> S<T> {
    pub async fn from_request(request: &'r Request<'_>) {
        let _ = T::from_request(request).await;
    }
}

fn main() {}
