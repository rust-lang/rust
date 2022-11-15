//@ignore-target-apple: park_timeout on macOS uses the system clock
use std::thread;
use std::time::{Duration, Instant};

fn main() {
    let start = Instant::now();

    thread::park_timeout(Duration::from_millis(200));

    // Thanks to deterministic execution, this will wiat *exactly* 200ms (rounded to 1ms).
    assert!((200..201).contains(&start.elapsed().as_millis()));
}
