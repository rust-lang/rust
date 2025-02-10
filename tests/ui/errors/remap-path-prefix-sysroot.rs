//@ revisions: with-remap without-remap
//@ compile-flags: -g -Ztranslate-remapped-path-to-local-path=yes
//@ [with-remap]compile-flags: --remap-path-prefix={{sysroot-rust-src-base}}=remapped
//@ [with-remap]compile-flags: --remap-path-prefix={{test-suite-src-base}}=remapped-tests-ui
//@ [without-remap]compile-flags:
//@ error-pattern: E0507

// The $SRC_DIR*.rs:LL:COL normalisation doesn't kick in automatically
// as the remapped revision will not begin with $SRC_DIR_REAL,
// so we have to do it ourselves.
//@ normalize-stderr: ".rs:\d+:\d+" -> ".rs:LL:COL"

use std::thread;
struct Worker {
    thread: thread::JoinHandle<()>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.thread.join().unwrap();
    }
}

pub fn main(){}
