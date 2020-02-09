#![warn(clippy::option_env_unwrap)]

macro_rules! option_env_unwrap {
    ($env: expr) => {
        option_env!($env).unwrap()
    };
    ($env: expr, $message: expr) => {
        option_env!($env).expect($message)
    };
}

fn main() {
    let _ = option_env!("HOME").unwrap();
    let _ = option_env!("HOME").expect("environment variable HOME isn't set");
    let _ = option_env_unwrap!("HOME");
    let _ = option_env_unwrap!("HOME", "environment variable HOME isn't set");
}
