//@ build-pass
//@ edition:2018

fn main() {
    let _ = foo();
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
