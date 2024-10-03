// Test that `MainThreadMarker` is neither `Send` nor `Sync`.
#![feature(darwin_mtm)]

//@ only-apple

use std::os::darwin::thread::MainThreadMarker;

fn needs_sync<T: Sync>() {}
fn needs_send<T: Send>() {}

fn main() {
    needs_sync::<MainThreadMarker>();
    //~^ ERROR `MainThreadMarker` cannot be shared between threads safely
    needs_send::<MainThreadMarker>();
    //~^ ERROR `MainThreadMarker` cannot be sent between threads safely
}
