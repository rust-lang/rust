//@ edition:2021
//@ aux-build:must-use-foreign.rs
//@ check-pass

extern crate must_use_foreign;

use must_use_foreign::Manager;

async fn async_main() {
    Manager::new().await.1.await;
}

fn main() {
    let _ = async_main();
}
