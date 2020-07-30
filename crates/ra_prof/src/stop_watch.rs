use crate::MemoryUsage;
use std::{
    fmt,
    time::{Duration, Instant},
};

pub struct StopWatch {
    time: Instant,
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
        let mut counter = perf_event::Builder::new().build().ok();
        if let Some(counter) = &mut counter {
            let _ = counter.enable();
        }
        let time = Instant::now();
        StopWatch { time, counter, memory: None }
    }
    pub fn memory(mut self, yes: bool) -> StopWatch {
        if yes {
            self.memory = Some(MemoryUsage::current());
        }
        self
    }
    pub fn elapsed(&mut self) -> StopWatchSpan {
        let time = self.time.elapsed();
        let instructions = self.counter.as_mut().and_then(|it| it.read().ok());
        let memory = self.memory.map(|it| MemoryUsage::current() - it);
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
                prefix = "k"
            }
            if instructions > 10000 {
                instructions /= 1000;
                prefix = "m"
            }
            write!(f, ", {}{}i", instructions, prefix)?;
        }
        if let Some(memory) = self.memory {
            write!(f, ", {}", memory)?;
        }
        Ok(())
    }
}

// Unclear if we need this:
// https://github.com/jimblandy/perf-event/issues/8
impl Drop for StopWatch {
    fn drop(&mut self) {
        if let Some(mut counter) = self.counter.take() {
            let _ = counter.disable();
        }
    }
}
