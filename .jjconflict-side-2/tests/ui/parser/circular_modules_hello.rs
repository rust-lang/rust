//@ ignore-auxiliary (used by `./circular-modules-main.rs`)

#[path = "circular_modules_main.rs"]
mod circular_modules_main;

pub fn say_hello() {
    println!("{}", circular_modules_main::hi_str());
}
