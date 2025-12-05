//@ignore-target: apple # park_timeout on macOS uses the system clock
use std::thread;
use std::time::{Duration, Instant};

fn main() {
    let start = Instant::now();

    thread::park_timeout(Duration::from_millis(200));

    // Thanks to deterministic execution, this will wait *exactly* 200ms, plus the time for the surrounding code.
    assert!((200..210).contains(&start.elapsed().as_millis()), "{}", start.elapsed().as_millis());
}
