//@only-on-host: the Linux std implementation opens /proc/self/exe, which doesn't work cross-target
//@ignore-target: freebsd # On FreeBSD the standard library uses `sysctl` for this, which is not shimmed
//@compile-flags: -Zmiri-disable-isolation
use std::env;

fn main() {
    // The actual value we get is a bit odd: we get the Miri binary that interprets us.
    env::current_exe().unwrap();
}
