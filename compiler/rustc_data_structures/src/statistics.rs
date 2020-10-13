//! # Support for collecting simple statistics
//!
//! Statistics are useful for collecting metrics from optimization passes, like
//! the number of simplifications performed. To avoid introducing overhead, the
//! collection of statistics is enabled only when rustc is compiled with
//! debug-assertions.
//!
//! Statistics are static variables defined in the module they are used, and
//! lazy registered in the global collector on the first use. Once registered,
//! the collector will obtain their values at the end of compilation process
//! when requested with -Zmir-opt-stats option.

use parking_lot::{const_mutex, Mutex};
use std::io::{self, stdout, Write as _};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static COLLECTOR: Collector = Collector::new();

/// Enables the collection of statistics.
/// To be effective it has to be called before the first use of a statistics.
pub fn try_enable() -> Result<(), ()> {
    COLLECTOR.try_enable()
}

/// Prints all statistics collected so far.
pub fn print() {
    COLLECTOR.print();
}

pub struct Statistic {
    category: &'static str,
    name: &'static str,
    initialized: AtomicBool,
    value: AtomicUsize,
}

struct Collector(Mutex<State>);

struct State {
    enabled: bool,
    stats: Vec<&'static Statistic>,
}

#[derive(Eq, PartialEq, Ord, PartialOrd)]
struct Snapshot {
    category: &'static str,
    name: &'static str,
    value: usize,
}

impl Statistic {
    pub const fn new(category: &'static str, name: &'static str) -> Self {
        Statistic {
            category,
            name,
            initialized: AtomicBool::new(false),
            value: AtomicUsize::new(0),
        }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn category(&self) -> &'static str {
        self.category.rsplit("::").next().unwrap()
    }

    #[inline]
    pub fn register(&'static self) {
        if cfg!(debug_assertions) {
            if !self.initialized.load(Ordering::Acquire) {
                COLLECTOR.register(self);
            }
        }
    }

    #[inline]
    pub fn increment(&'static self, value: usize) {
        if cfg!(debug_assertions) {
            self.value.fetch_add(value, Ordering::Relaxed);
            self.register();
        }
    }

    #[inline]
    pub fn update_max(&'static self, value: usize) {
        if cfg!(debug_assertions) {
            self.value.fetch_max(value, Ordering::Relaxed);
            self.register();
        }
    }

    fn snapshot(&'static self) -> Snapshot {
        Snapshot {
            name: self.name(),
            category: self.category(),
            value: self.value.load(Ordering::Relaxed),
        }
    }
}

impl Collector {
    const fn new() -> Self {
        Collector(const_mutex(State { enabled: false, stats: Vec::new() }))
    }

    fn try_enable(&self) -> Result<(), ()> {
        if cfg!(debug_assertions) {
            self.0.lock().enabled = true;
            Ok(())
        } else {
            Err(())
        }
    }

    fn snapshot(&self) -> Vec<Snapshot> {
        self.0.lock().stats.iter().copied().map(Statistic::snapshot).collect()
    }

    fn register(&self, s: &'static Statistic) {
        let mut state = self.0.lock();
        if !s.initialized.load(Ordering::Relaxed) {
            if state.enabled {
                state.stats.push(s);
            }
            s.initialized.store(true, Ordering::Release);
        }
    }

    fn print(&self) {
        let mut stats = self.snapshot();
        stats.sort();
        match self.write(&stats) {
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::BrokenPipe => {}
            Err(e) => panic!(e),
        }
    }

    fn write(&self, stats: &[Snapshot]) -> io::Result<()> {
        let mut cat_width = 0;
        let mut val_width = 0;

        for s in stats {
            cat_width = cat_width.max(s.category.len());
            val_width = val_width.max(s.value.to_string().len());
        }

        let mut out = Vec::new();
        for s in stats {
            write!(
                &mut out,
                "{val:val_width$} {cat:cat_width$} {name}\n",
                val = s.value,
                val_width = val_width,
                cat = s.category,
                cat_width = cat_width,
                name = s.name,
            )?;
        }

        stdout().write_all(&out)
    }
}
