// run-pass

use std::sync::{Arc, Mutex};

fn main() {
    let data = Arc::new(Mutex::new(0));
    let _ = data.lock().unwrap(); //~WARNING non-binding let on a synchronization lock
}
