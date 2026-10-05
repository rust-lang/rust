//@ignore-target: freebsd # we don't have the shims this needs here
//@compile-flags: -Zmiri-disable-isolation
//@run-native
use std::env;

fn main() {
    let exe = env::current_exe().unwrap();
    assert!(exe.is_absolute());
    println!("{}", exe.file_name().unwrap().display());
}
