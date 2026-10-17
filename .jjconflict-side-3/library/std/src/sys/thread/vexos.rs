use crate::time::{Duration, Instant};

pub fn yield_now() {
    unsafe {
        vex_sdk::vexTasksRun();
    }
}

pub fn sleep(dur: Duration) {
    let start = Instant::now();

    while start.elapsed() < dur {
        yield_now();
    }
}

pub fn sleep_until(deadline: Instant) {
    while Instant::now() < deadline {
        yield_now();
    }
}
