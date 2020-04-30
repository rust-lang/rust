// ignore-linux: Only Windows is not supported.
// ignore-macos: Only Windows is not supported.

use std::thread;

// error-pattern: Miri does not support threading

fn main() {
    thread::spawn(|| {});
}
