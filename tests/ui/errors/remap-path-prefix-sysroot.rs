// revisions: with-remap without-remap
// compile-flags: -g -Ztranslate-remapped-path-to-local-path=yes
// [with-remap]compile-flags: --remap-path-prefix={{rust-src-base}}=remapped
// [without-remap]compile-flags:
// error-pattern: E0507

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
