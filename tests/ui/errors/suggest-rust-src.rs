// When a diagnostic points into the standard library but its source isn't
// available locally, rustc can't render the snippet. In that case it should
// suggest installing the `rust-src` component. See #156402.
//
// We simulate the "std source unavailable" situation (as happens with a `rustup`
// toolchain that doesn't have `rust-src` installed) by remapping the rust-src
// base and disabling translation back to the local path.
//@ compile-flags: -Z simulate-remapped-rust-src-base=/rustc/FAKE_PREFIX -Z translate-remapped-path-to-local-path=no
//
// The line:col of the remapped std path is volatile, so normalise it. (The
// `$SRC_DIR` normalisation doesn't kick in because the path is remapped.)
//@ normalize-stderr: ".rs:\d+:\d+" -> ".rs:LL:COL"

use std::thread;

struct Worker {
    thread: thread::JoinHandle<()>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.thread.join().unwrap();
        //~^ ERROR cannot move out of `self.thread` which is behind a mutable reference
    }
}

fn main() {}
