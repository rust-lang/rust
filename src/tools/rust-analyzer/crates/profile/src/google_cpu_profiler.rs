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

const OFF: usize = 0;
const ON: usize = 1;
const PENDING: usize = 2;

fn transition(current: usize, new: usize) -> bool {
    static STATE: AtomicUsize = AtomicUsize::new(OFF);

    STATE.compare_exchange(current, new, Ordering::SeqCst, Ordering::SeqCst).is_ok()
}

pub(crate) fn start(path: &Path) {
    if !transition(OFF, PENDING) {
        panic!("profiler already started");
    }
    let path = CString::new(path.display().to_string()).unwrap();
    if unsafe { ProfilerStart(path.as_ptr()) } == 0 {
        panic!("profiler failed to start")
    }
    assert!(transition(PENDING, ON));
}

pub(crate) fn stop() {
    if !transition(ON, PENDING) {
        panic!("profiler is not started")
    }
    unsafe { ProfilerStop() };
    assert!(transition(PENDING, OFF));
}
