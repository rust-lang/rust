// aux-build:macro_rules.rs
#![warn(clippy::option_env_unwrap)]
#![allow(clippy::map_flatten)]

#[macro_use]
extern crate macro_rules;

macro_rules! option_env_unwrap {
    ($env: expr) => {
        option_env!($env).unwrap()
    };
    ($env: expr, $message: expr) => {
        option_env!($env).expect($message)
    };
}

fn main() {
    let _ = option_env!("PATH").unwrap();
    let _ = option_env!("PATH").expect("environment variable PATH isn't set");
    let _ = option_env_unwrap!("PATH");
    let _ = option_env_unwrap!("PATH", "environment variable PATH isn't set");
    let _ = option_env_unwrap_external!("PATH");
    let _ = option_env_unwrap_external!("PATH", "environment variable PATH isn't set");
}
