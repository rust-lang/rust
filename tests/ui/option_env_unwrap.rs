#![warn(clippy::option_env_unwrap)]

macro_rules! option_env_unwrap {
    ($env: expr) => {
        option_env!($env).unwrap()
    };
}

fn main() {
    let _ = option_env!("HOME").unwrap();
    let _ = option_env_unwrap!("HOME");
}
