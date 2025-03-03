//! Like `std::time::Instant`, but also measures memory & CPU cycles.

#![allow(clippy::print_stderr)]

use std::{
    fmt,
    time::{Duration, Instant},
};

use crate::MemoryUsage;

pub struct StopWatch {
    time: Instant,
    #[cfg(all(target_os = "linux", not(target_env = "ohos")))]
    counter: Option<perf_event::Counter>,
    memory: MemoryUsage,
}

pub struct StopWatchSpan {
    pub time: Duration,
    pub instructions: Option<u64>,
    pub memory: MemoryUsage,
}

impl StopWatch {
    pub fn start() -> StopWatch {
        #[cfg(all(target_os = "linux", not(target_env = "ohos")))]
        let counter = {
            // When debugging rust-analyzer using rr, the perf-related syscalls cause it to abort.
            // We allow disabling perf by setting the env var `RA_DISABLE_PERF`.

            use std::sync::OnceLock;
            static PERF_ENABLED: OnceLock<bool> = OnceLock::new();

            if *PERF_ENABLED.get_or_init(|| std::env::var_os("RA_DISABLE_PERF").is_none()) {
                let mut counter = perf_event::Builder::new()
                    .build()
                    .map_err(|err| eprintln!("Failed to create perf counter: {err}"))
                    .ok();
                if let Some(counter) = &mut counter {
                    if let Err(err) = counter.enable() {
                        eprintln!("Failed to start perf counter: {err}")
                    }
                }
                counter
            } else {
                None
            }
        };
        let memory = MemoryUsage::now();
        let time = Instant::now();
        StopWatch {
            time,
            #[cfg(all(target_os = "linux", not(target_env = "ohos")))]
            counter,
            memory,
        }
    }

    pub fn elapsed(&mut self) -> StopWatchSpan {
        let time = self.time.elapsed();

        #[cfg(all(target_os = "linux", not(target_env = "ohos")))]
        let instructions = self.counter.as_mut().and_then(|it| {
            it.read().map_err(|err| eprintln!("Failed to read perf counter: {err}")).ok()
        });
        #[cfg(all(target_os = "linux", target_env = "ohos"))]
        let instructions = None;
        #[cfg(not(target_os = "linux"))]
        let instructions = None;

        let memory = MemoryUsage::now() - self.memory;
        StopWatchSpan { time, instructions, memory }
    }
}

impl fmt::Display for StopWatchSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2?}", self.time)?;
        if let Some(mut instructions) = self.instructions {
            let mut prefix = "";
            if instructions > 10000 {
                instructions /= 1000;
                prefix = "k";
            }
            if instructions > 10000 {
                instructions /= 1000;
                prefix = "m";
            }
            if instructions > 10000 {
                instructions /= 1000;
                prefix = "g";
            }
            write!(f, ", {instructions}{prefix}instr")?;
        }
        write!(f, ", {}", self.memory)?;
        Ok(())
    }
}
