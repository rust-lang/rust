use std::thread;

// error-pattern: Miri does not support threading

fn main() {
    thread::spawn(|| {});
}
