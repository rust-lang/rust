// edition:2018
#![feature(must_not_suspend)]
#![deny(must_not_suspend)]

async fn other() {}

pub async fn uhoh(m: std::sync::Mutex<()>) {
    let _guard = m.lock().unwrap(); //~ ERROR `MutexGuard` held across
    other().await;
}

fn main() {
}
