//@ check-pass

//@ edition:2018
#![deny(must_not_suspend)]
//~^ WARNING unknown lint: `must_not_suspend`

async fn other() {}

pub async fn uhoh(m: std::sync::Mutex<()>) {
    let _guard = m.lock().unwrap();
    other().await;
}

fn main() {
}
