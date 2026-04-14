use core::sync::atomic::{AtomicUsize, Ordering};

pub trait PerfReporter: Send + Sync {
    fn span_enter(&self, name: &'static str);
    fn span_exit(&self, name: &'static str, duration_ns: u64);
    fn counter(&self, name: &'static str, value: u64);
}

static REPORTER: AtomicUsize = AtomicUsize::new(0);

pub fn set_reporter(reporter: &'static dyn PerfReporter) {
    REPORTER.store(
        reporter as *const dyn PerfReporter as *const () as usize,
        Ordering::SeqCst,
    );
}

#[inline(always)]
pub fn get_reporter() -> Option<&'static dyn PerfReporter> {
    let ptr = REPORTER.load(Ordering::Relaxed);
    if ptr == 0 {
        None
    } else {
        None // Temporary
    }
}

// Re-thinking: Instead of dyn trait in atomic, use function pointers directly for lowest overhead.
pub type SpanEnterFn = fn(&'static str);
pub type SpanExitFn = fn(&'static str, u64);
pub type CounterFn = fn(&'static str, u64);
pub type EventFn = fn(&'static str, &'static str);

static ENTER_FN: AtomicUsize = AtomicUsize::new(0);
static EXIT_FN: AtomicUsize = AtomicUsize::new(0);
static COUNTER_FN: AtomicUsize = AtomicUsize::new(0);
static EVENT_FN: AtomicUsize = AtomicUsize::new(0);

pub fn register_hooks(enter: SpanEnterFn, exit: SpanExitFn, counter: CounterFn, event: EventFn) {
    ENTER_FN.store(enter as usize, Ordering::SeqCst);
    EXIT_FN.store(exit as usize, Ordering::SeqCst);
    COUNTER_FN.store(counter as usize, Ordering::SeqCst);
    EVENT_FN.store(event as usize, Ordering::SeqCst);
}

#[inline(always)]
pub fn span_enter(name: &'static str) {
    let ptr = ENTER_FN.load(Ordering::Relaxed);
    if ptr != 0 {
        let f: SpanEnterFn = unsafe { core::mem::transmute(ptr) };
        f(name);
    }
}

#[inline(always)]
pub fn span_exit(name: &'static str, duration: u64) {
    let ptr = EXIT_FN.load(Ordering::Relaxed);
    if ptr != 0 {
        let f: SpanExitFn = unsafe { core::mem::transmute(ptr) };
        f(name, duration);
    }
}

#[inline(always)]
pub fn counter(name: &'static str, value: u64) {
    let ptr = COUNTER_FN.load(Ordering::Relaxed);
    if ptr != 0 {
        let f: CounterFn = unsafe { core::mem::transmute(ptr) };
        f(name, value);
    }
}

#[inline(always)]
pub fn event(name: &'static str, data: &'static str) {
    let ptr = EVENT_FN.load(Ordering::Relaxed);
    if ptr != 0 {
        let f: EventFn = unsafe { core::mem::transmute(ptr) };
        f(name, data);
    }
}

pub struct PerfSpan {
    name: &'static str,
    start: u64,
}

impl PerfSpan {
    #[inline(always)]
    pub fn new(name: &'static str) -> Self {
        span_enter(name);
        Self {
            name,
            start: crate::time::monotonic_ns(),
        }
    }
}

impl Drop for PerfSpan {
    #[inline(always)]
    fn drop(&mut self) {
        let duration = crate::time::monotonic_ns().saturating_sub(self.start);
        span_exit(self.name, duration);
    }
}

#[macro_export]
macro_rules! trace_span {
    ($name:expr) => {
        let _span = $crate::perf::PerfSpan::new($name);
    };
}

#[macro_export]
macro_rules! trace_counter {
    ($name:expr, $val:expr) => {
        $crate::perf::counter($name, $val as u64);
    };
}

#[macro_export]
macro_rules! trace_event {
    ($name:expr, $data:expr) => {
        $crate::perf::event($name, $data);
    };
}
