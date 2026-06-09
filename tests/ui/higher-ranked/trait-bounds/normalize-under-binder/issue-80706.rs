//@ build-pass
//@ edition:2018

type BoxFuture<T> = std::pin::Pin<Box<dyn std::future::Future<Output=T>>>;

fn main() {
    f();
}

async fn f() {
    run("dependency").await;
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
    fn execute(
        &self,
    ) -> BoxFuture<Result<<Self as StorageRequestReturnType>::Output, <S as Storage>::Error>>;
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
        SaveUser { name: "Joe" }
            .execute()
            .await;
    }
}

async fn run<S>(dep: &str)
where
    S: Storage,
    for<'a> SaveUser<'a>: StorageRequest<S>,
    for<'a> SaveUser<'a>: StorageRequestReturnType,
{
    User { dep }.save().await;
}
