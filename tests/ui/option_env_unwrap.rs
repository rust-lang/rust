//@aux-build:proc_macros.rs:proc-macro
#![warn(clippy::option_env_unwrap)]
#![allow(clippy::map_flatten)]

extern crate proc_macros;
use proc_macros::{external, inline_macros};

#[inline_macros]
fn main() {
    let _ = option_env!("PATH").unwrap();
    let _ = option_env!("PATH").expect("environment variable PATH isn't set");
    let _ = inline!(option_env!($"PATH").unwrap());
    let _ = inline!(option_env!($"PATH").expect($"environment variable PATH isn't set"));
    let _ = external!(option_env!($"PATH").unwrap());
    let _ = external!(option_env!($"PATH").expect($"environment variable PATH isn't set"));
}
