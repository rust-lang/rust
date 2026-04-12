// Regression test for #135287.
// Changing a struct to an enum with incremental compilation previously caused
// an ICE: `adt_sized_constraint called on non-struct type`.

//@ revisions: cfail1 cfail2
//@ edition: 2021
//@ check-pass

#![allow(dead_code)]

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll};

#[cfg(cfail1)]
struct SourceDocument {}

#[cfg(cfail2)]
enum SourceDocument {}

trait Loader {
    type Value;
    fn load(&self) -> impl Future<Output = Self::Value>;
}

struct SourceDocumentLoader;
impl Loader for SourceDocumentLoader {
    type Value = SourceDocument;
    async fn load(&self) -> Self::Value {
        todo!()
    }
}

struct ManualSend<T>(T);
unsafe impl<T: Send> Send for ManualSend<T> {}

struct PendingButCovariant<T>(PhantomData<T>);
impl<T> Future for PendingButCovariant<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

struct DataLoader<T>(T);
impl<T> DataLoader<T> {
    async fn load_one(&self) -> ManualSend<T::Value>
    where
        T: Loader,
    {
        PendingButCovariant(PhantomData).await
    }
}

trait ContainerType {
    fn resolve_field(&self) -> impl Future<Output = ()> + Send;
}
impl ContainerType for () {
    async fn resolve_field(&self) {
        let loader = DataLoader(SourceDocumentLoader);
        loader.load_one().await;
    }
}

fn main() {}
