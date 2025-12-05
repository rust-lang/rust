#![warn(clippy::large_futures)]

fn main() {}

pub async fn should_warn() {
    let x = [0u8; 1024];
    async {}.await;
    dbg!(x);
}

pub async fn should_not_warn() {
    let x = [0u8; 1020];
    async {}.await;
    dbg!(x);
}

pub async fn bar() {
    should_warn().await;
    //~^ large_futures

    async {
        let x = [0u8; 1024];
        dbg!(x);
    }
    .await;

    should_not_warn().await;
}
