//@aux-build:proc_macros.rs
#![warn(clippy::option_env_unwrap)]
#![allow(clippy::map_flatten)]

extern crate proc_macros;
use proc_macros::{external, inline_macros};

#[inline_macros]
fn main() {
    let _ = option_env!("PATH").unwrap();
    //~^ option_env_unwrap
    let _ = option_env!("PATH").expect("environment variable PATH isn't set");
    //~^ option_env_unwrap
    let _ = option_env!("__Y__do_not_use").unwrap(); // This test only works if you don't have a __Y__do_not_use env variable in your environment.
    //
    //~^^ option_env_unwrap
    let _ = inline!(option_env!($"PATH").unwrap());
    //~^ option_env_unwrap
    let _ = inline!(option_env!($"PATH").expect($"environment variable PATH isn't set"));
    //~^ option_env_unwrap
    let _ = external!(option_env!($"PATH").unwrap());
    //~^ option_env_unwrap
    let _ = external!(option_env!($"PATH").expect($"environment variable PATH isn't set"));
    //~^ option_env_unwrap
}
