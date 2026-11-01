//@ run-pass

#![feature(share_trait)]

use std::clone::Share;
use std::sync::{Arc, Mutex};

trait Value {
    fn get(&self) -> i32;
}

impl Value for Mutex<i32> {
    fn get(&self) -> i32 {
        *self.lock().unwrap()
    }
}

fn main() {
    let value = Arc::new(Mutex::new(1));
    let shared = value.share();

    assert!(Arc::ptr_eq(&value, &shared));
    *shared.lock().unwrap() = 2;
    assert_eq!(*value.lock().unwrap(), 2);

    let dyn_value: Arc<dyn Value + Send + Sync> = Arc::new(Mutex::new(3));
    let shared_dyn_value = dyn_value.share();

    assert!(Arc::ptr_eq(&dyn_value, &shared_dyn_value));
    assert_eq!(shared_dyn_value.get(), 3);
}
