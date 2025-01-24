#![warn(clippy::zombie_processes)]
#![allow(clippy::needless_return)]

use std::process::{Child, Command};

fn main() {
    let _ = Command::new("").spawn().unwrap();
    //~^ ERROR: spawned process is never `wait()`ed on
    Command::new("").spawn().unwrap();
    //~^ ERROR: spawned process is never `wait()`ed on
    spawn_proc();
    //~^ ERROR: spawned process is never `wait()`ed on
    spawn_proc().wait().unwrap(); // OK
}

fn not_main() {
    Command::new("").spawn().unwrap();
}

fn spawn_proc() -> Child {
    Command::new("").spawn().unwrap()
}

fn spawn_proc_2() -> Child {
    return Command::new("").spawn().unwrap();
}
