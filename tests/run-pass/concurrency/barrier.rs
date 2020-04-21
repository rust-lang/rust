// ignore-windows: Concurrency on Windows is not supported yet.

//! Check if Rust barriers are working.

use std::sync::{Arc, Barrier};
use std::thread;


/// This test is taken from the Rust documentation.
fn main() {
    let mut handles = Vec::with_capacity(10);
    let barrier = Arc::new(Barrier::new(10));
    for _ in 0..10 {
        let c = barrier.clone();
        // The same messages will be printed together.
        // You will NOT see any interleaving.
        handles.push(thread::spawn(move|| {
            println!("before wait");
            c.wait();
            println!("after wait");
        }));
    }
    // Wait for other threads to finish.
    for handle in handles {
        handle.join().unwrap();
    }
}