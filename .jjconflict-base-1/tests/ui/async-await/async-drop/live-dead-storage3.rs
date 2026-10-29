// ex-ice: #141761
//@ compile-flags: -Zlint-mir --crate-type lib
//@ edition:2024
//@ check-pass

#![feature(async_drop)]
#![allow(incomplete_features)]

type BoxFuture<T> = std::pin::Pin<Box<dyn Future<Output = T>>>;
fn main() {}
async fn f() {
    run("").await
}
struct InMemoryStorage;
struct User<'dep> {
    dep: &'dep str,
}
impl<'a> StorageRequest<InMemoryStorage> for SaveUser<'a> {
    fn execute(&self) -> BoxFuture<Result<(), String>> {
        todo!()
    }
}
trait Storage {
    type Error;
}
impl Storage for InMemoryStorage {
    type Error = String;
}
trait StorageRequestReturnType {
    type Output;
}
trait StorageRequest<S: Storage>: StorageRequestReturnType {
    fn execute(&self) -> BoxFuture<Result<<Self>::Output, S::Error>>;
}
struct SaveUser<'a> {
    name: &'a str,
}
impl<'a> StorageRequestReturnType for SaveUser<'a> {
    type Output = ();
}
impl<'dep> User<'dep> {
    async fn save<S>(self)
    where
        S: Storage,
        for<'a> SaveUser<'a>: StorageRequest<S>,
    {
        SaveUser { name: "" }.execute().await;
    }
}
async fn run<S>(dep: &str)
where
    S: Storage,
    for<'a> SaveUser<'a>: StorageRequest<S>,
{
    User { dep }.save().await
}
