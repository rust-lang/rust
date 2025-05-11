#![warn(clippy::zombie_processes)]
#![allow(clippy::needless_return)]

use std::process::{Child, Command};

fn main() {
    let _ = Command::new("").spawn().unwrap();
    //~^ zombie_processes

    Command::new("").spawn().unwrap();
    //~^ zombie_processes

    spawn_proc();
    //~^ zombie_processes

    spawn_proc().wait().unwrap(); // OK
}

fn not_main() {
    Command::new("").spawn().unwrap();
    //~^ zombie_processes
}

fn spawn_proc() -> Child {
    Command::new("").spawn().unwrap()
}

fn spawn_proc_2() -> Child {
    return Command::new("").spawn().unwrap();
}
