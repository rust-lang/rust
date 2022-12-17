//@compile-flags: -Zmiri-disable-isolation -Zmiri-permissive-provenance -Zmiri-backtrace=full
//@only-target-x86_64-unknown-linux: support for tokio only on linux and x86

use tokio::time::{sleep, Duration, Instant};

#[tokio::main]
async fn main() {
    let start = Instant::now();
    sleep(Duration::from_secs(1)).await;
    // It takes 96 millisecond to sleep for 1 millisecond
    // It takes 1025 millisecond to sleep for 1 second
    let time_elapsed = &start.elapsed().as_millis();
    assert!(time_elapsed > &1000, "{}", time_elapsed);
}
