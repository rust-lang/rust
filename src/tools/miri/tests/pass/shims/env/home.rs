//@compile-flags: -Zmiri-disable-isolation
use std::env;

fn main() {
    // Remove the env vars to hit the underlying shim -- except
    // on android where the env var is all we have.
    #[cfg(not(target_os = "android"))]
    env::remove_var("HOME");
    env::remove_var("USERPROFILE");

    #[allow(deprecated)]
    env::home_dir().unwrap();
}
