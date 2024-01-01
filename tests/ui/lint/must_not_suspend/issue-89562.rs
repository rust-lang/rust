// edition:2018
// build-pass

use std::sync::Mutex;

// Copied from the issue. Allow-by-default for now, so build-pass
pub async fn foo() {
    let foo = Mutex::new(1);
    let lock = foo.lock().unwrap();

    // Prevent mutex lock being held across `.await` point.
    drop(lock);

    bar().await;
}

async fn bar() {}

fn main() {}
