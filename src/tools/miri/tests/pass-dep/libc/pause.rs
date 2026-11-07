//@ignore-target: windows # only very limited libc on Windows
//@compile-flags: -Zmiri-ignore-leaks
//@run-native

use std::{thread, time};

fn main() {
    thread::spawn(|| unsafe {
        libc::pause(); // should never return
        unreachable!();
    });
    thread::sleep(time::Duration::from_millis(100));
}
