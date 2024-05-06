//@compile-flags: -Zmiri-disable-isolation
use std::env;

fn main() {
    env::remove_var("HOME"); // make sure we enter the interesting codepath
    env::remove_var("USERPROFILE"); // Windows also looks as this env var
    #[allow(deprecated)]
    env::home_dir().unwrap();
}
