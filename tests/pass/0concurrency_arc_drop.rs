// ignore-windows: Concurrency on Windows is not supported yet.
use std::sync::Arc;
use std::thread;

/// Test for Arc::drop bug (https://github.com/rust-lang/rust/issues/55005)
fn main() {
    // The bug seems to take up to 700 iterations to reproduce with most seeds (tested 0-9).
    for _ in 0..700 {
        let arc_1 = Arc::new(());
        let arc_2 = arc_1.clone();
        let thread = thread::spawn(|| drop(arc_2));
        let mut i = 0;
        while i < 256 {
            i += 1;
        }
        drop(arc_1);
        thread.join().unwrap();
    }
}
