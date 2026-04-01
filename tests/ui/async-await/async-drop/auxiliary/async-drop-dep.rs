//@ edition:2021

#![feature(async_drop)]
#![allow(incomplete_features)]

pub struct HasDrop;
impl Drop for HasDrop{
    fn drop(&mut self) {
        println!("Sync drop");
    }
}

pub struct MongoDrop;
impl MongoDrop {
    pub async fn new() -> Result<Self, HasDrop> {
        Ok(Self)
    }
}
impl Drop for MongoDrop{
    fn drop(&mut self) {
        println!("Sync drop");
    }
}
impl std::future::AsyncDrop for MongoDrop {
    async fn drop(self: std::pin::Pin<&mut Self>) {
        println!("Async drop");
    }
}
