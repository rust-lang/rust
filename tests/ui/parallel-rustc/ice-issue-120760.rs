// Test for #120760, which causes an ice bug
//
//@ parallel-front-end-robustness
//@ compile-flags: -Z threads=45

type BoxFuture<T> = std::pin::Pin<Box<dyn std::future::Future<Output = T>>>;

fn main() {
    let _ = f();
}

async fn f() {
    run("dependency").await;
}

struct InMemoryStorage;

pub struct User<'dep> {
    pub name: &'a str,
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
    fn execute(
        &self,
    ) -> BoxFuture<Result<<SaveUser as StorageRequestReturnType>::Output, <S as Storage>::Error>>;
}

pub struct SaveUser<'a> {
    pub name: &'a str,
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
        let _ = run("dependency").await;
    }
}

async fn execute<S>(dep: &str)
where
    S: Storage,
    for<'a> SaveUser<'a>: StorageRequest<S>,
{
    User { dep }.save().await;
}
