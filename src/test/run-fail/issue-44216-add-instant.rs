// error-pattern:overflow

use std::time::{Instant, Duration};

fn main() {
    let now = Instant::now();
    let _ = now + Duration::from_secs(u64::max_value());
}
