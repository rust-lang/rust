// ignore-linux: Only Windows is not supported.
// ignore-macos: Only Windows is not supported.

use std::thread;

// error-pattern: Miri does not support concurrency on Windows

fn main() {
    thread::spawn(|| {});
}
