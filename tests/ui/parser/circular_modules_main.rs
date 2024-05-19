//@ error-pattern: circular modules

#[path = "circular_modules_hello.rs"]
mod circular_modules_hello;

pub fn hi_str() -> String {
    "Hi!".to_string()
}

fn main() {
    circular_modules_hello::say_hello();
}
