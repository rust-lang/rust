//@only-target: linux # We only support tokio on Linux

use tokio::time::{Duration, Instant, sleep};

#[tokio::main]
async fn main() {
    let start = Instant::now();
    sleep(Duration::from_millis(100)).await;
    let time_elapsed = &start.elapsed().as_millis();
    assert!((100..1000).contains(time_elapsed), "{}", time_elapsed);
}
