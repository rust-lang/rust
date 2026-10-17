//@compile-flags: -Zmiri-disable-isolation
use std::env;

fn main() {
    // If this is set it may hide calling some shims.
    unsafe { env::remove_var("TMPDIR") };

    let _path = env::temp_dir();
}
