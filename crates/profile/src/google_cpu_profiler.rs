//! https://github.com/gperftools/gperftools

use std::{
    ffi::CString,
    os::raw::c_char,
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
};

#[link(name = "profiler")]
#[allow(non_snake_case)]
extern "C" {
    fn ProfilerStart(fname: *const c_char) -> i32;
    fn ProfilerStop();
}

static PROFILER_STATE: AtomicUsize = AtomicUsize::new(OFF);
const OFF: usize = 0;
const ON: usize = 1;
const PENDING: usize = 2;

pub fn start(path: &Path) {
    if PROFILER_STATE.compare_and_swap(OFF, PENDING, Ordering::SeqCst) != OFF {
        panic!("profiler already started");
    }
    let path = CString::new(path.display().to_string()).unwrap();
    if unsafe { ProfilerStart(path.as_ptr()) } == 0 {
        panic!("profiler failed to start")
    }
    assert!(PROFILER_STATE.compare_and_swap(PENDING, ON, Ordering::SeqCst) == PENDING);
}

pub fn stop() {
    if PROFILER_STATE.compare_and_swap(ON, PENDING, Ordering::SeqCst) != ON {
        panic!("profiler is not started")
    }
    unsafe { ProfilerStop() };
    assert!(PROFILER_STATE.compare_and_swap(PENDING, OFF, Ordering::SeqCst) == PENDING);
}
