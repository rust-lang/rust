// ignore-windows

//! This test just calls the relevant APIs to check if Miri crashes.

use std::sync::{Arc, Mutex};
use std::thread;

fn main() {

    let data = Arc::new(Mutex::new(0));
    let mut threads = Vec::new();

    for _ in 0..3 {
        let data = Arc::clone(&data);
        let thread = thread::spawn(move || {
            let mut data = data.lock().unwrap();
            *data += 1;
        });
        threads.push(thread);
    }

    for thread in threads {
        thread.join().unwrap();
    }

    assert!(data.try_lock().is_ok());

    let data = Arc::try_unwrap(data).unwrap().into_inner().unwrap();
    assert_eq!(data, 3);

}