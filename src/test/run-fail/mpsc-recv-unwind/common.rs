// ignore-test

// Common code used by the other tests in this directory

extern crate libc;

use std::sync::{Arc, Barrier, mpsc};
use std::time::Duration;
use std::{env, thread, process};

pub fn panic_inside_mpsc_recv(r: mpsc::Receiver<()>) {
    env::set_var("LIBC_FATAL_STDERR_", "1");
    let barrier = Arc::new(Barrier::new(2));
    let main_thread = unsafe { libc::pthread_self() };
    let barrier2 = barrier.clone();
    thread::spawn(move || {
        barrier2.wait();
        // try to make sure main thread proceeds into recv
        thread::sleep(Duration::from_millis(100));
        unsafe { libc::pthread_cancel(main_thread); }
        thread::sleep(Duration::from_millis(2000));
        println!("Deadlock detected");
        process::exit(1);
    });
    barrier.wait();
    r.recv().unwrap()
}
