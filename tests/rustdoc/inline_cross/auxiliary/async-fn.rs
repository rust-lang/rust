//@ edition: 2021

pub async fn load() -> i32 {
    0
}

pub trait Load {
    async fn run(&self) -> i32;
}

pub struct Loader;

impl Load for Loader {
    async fn run(&self) -> i32 {
        1
    }
}
