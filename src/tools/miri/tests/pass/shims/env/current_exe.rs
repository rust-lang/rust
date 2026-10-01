//@ignore-target: freebsd illumos solaris # we don't have the shims this needs here
//@compile-flags: -Zmiri-disable-isolation
use std::env;

fn main() {
    // The actual value we get is a bit odd: we get the Miri binary that interprets us.
    env::current_exe().unwrap();
}
