//@compile-flags: -Zmiri-permissive-provenance -Zmiri-backtrace=full
//@only-target-x86_64-unknown-linux: support for tokio only on linux and x86
//@error-pattern: returning ready events from epoll_wait is not yet implemented
//@normalize-stderr-test: " += note:.*\n" -> ""

use tokio::time::{sleep, Duration, Instant};

#[tokio::main]
async fn main() {
    let start = Instant::now();
    sleep(Duration::from_secs(1)).await;
    let time_elapsed = &start.elapsed().as_millis();
    assert!((1000..1100).contains(time_elapsed), "{}", time_elapsed);
}
