//@ revisions: with-remap without-remap
//@ compile-flags: -g -Ztranslate-remapped-path-to-local-path=yes
//@ [with-remap]compile-flags: --remap-path-prefix={{rust-src-base}}=remapped
//@ [with-remap]compile-flags: --remap-path-prefix={{src-base}}=remapped-tests-ui
// [without-remap] no extra compile-flags

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
        //[without-remap]~^ ERROR cannot move out of `self.thread` which is behind a mutable reference
    }
}

pub fn main(){}

//[with-remap]~? ERROR cannot move out of `self.thread` which is behind a mutable reference
