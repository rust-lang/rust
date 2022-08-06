//@ignore-target-windows
//@compile-flags: -Zmiri-disable-isolation
use std::env;

fn main() {
    // The actual value we get is a bit odd: we get the Miri binary that interprets us.
    env::current_exe().unwrap();
}
