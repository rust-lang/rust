//! Like `std::time::Instant`, but also measures memory & CPU cycles.
use std::{
    fmt,
    time::{Duration, Instant},
};

use crate::MemoryUsage;

pub struct StopWatch {
    time: Instant,
    #[cfg(target_os = "linux")]
    counter: Option<perf_event::Counter>,
    memory: Option<MemoryUsage>,
}

pub struct StopWatchSpan {
    pub time: Duration,
    pub instructions: Option<u64>,
    pub memory: Option<MemoryUsage>,
}

impl StopWatch {
    pub fn start() -> StopWatch {
        #[cfg(target_os = "linux")]
        let counter = {
            // When debugging rust-analyzer using rr, the perf-related syscalls cause it to abort.
            // We allow disabling perf by setting the env var `RA_DISABLE_PERF`.

            use once_cell::sync::Lazy;
            static PERF_ENABLED: Lazy<bool> =
                Lazy::new(|| std::env::var_os("RA_DISABLE_PERF").is_none());

            if *PERF_ENABLED {
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
        let time = Instant::now();
        StopWatch {
            time,
            #[cfg(target_os = "linux")]
            counter,
            memory: None,
        }
    }
    pub fn memory(mut self, yes: bool) -> StopWatch {
        if yes {
            self.memory = Some(MemoryUsage::now());
        }
        self
    }
    pub fn elapsed(&mut self) -> StopWatchSpan {
        let time = self.time.elapsed();

        #[cfg(target_os = "linux")]
        let instructions = self.counter.as_mut().and_then(|it| {
            it.read().map_err(|err| eprintln!("Failed to read perf counter: {err}")).ok()
        });
        #[cfg(not(target_os = "linux"))]
        let instructions = None;

        let memory = self.memory.map(|it| MemoryUsage::now() - it);
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
        if let Some(memory) = self.memory {
            write!(f, ", {memory}")?;
        }
        Ok(())
    }
}
