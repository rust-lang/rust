//@ edition: 2021

#![deny(let_underscore_drop)]
fn main() {
    let _ = foo(); //~ ERROR non-binding let on a type that has a destructor
}

async fn from_config(_: Config) {}

async fn foo() {
    from_config(Config {
        nickname: None,
        ..Default::default()
    })
    .await;
}

#[derive(Default)]
struct Config {
    nickname: Option<Box<u8>>,
}
