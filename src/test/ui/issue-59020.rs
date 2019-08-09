// edition:2018
// run-pass
// ignore-emscripten no threads support
// ignore-sgx no thread sleep support

use std::thread;
use std::time::Duration;

fn main() {
    let t1 = thread::spawn(|| {
        let sleep = Duration::new(0,100_000);
        for _ in 0..100 {
            println!("Parking1");
            thread::park_timeout(sleep);
        }
    });

    let t2 = thread::spawn(|| {
        let sleep = Duration::new(0,100_000);
        for _ in 0..100 {
            println!("Parking2");
            thread::park_timeout(sleep);
        }
    });

    t1.join().expect("Couldn't join thread 1");
    t2.join().expect("Couldn't join thread 2");
}
