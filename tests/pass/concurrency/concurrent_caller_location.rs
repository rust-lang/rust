// ignore-windows: Concurrency on Windows is not supported yet.

use std::thread::spawn;
use std::panic::Location;

fn initialize() {
    let _ignore = initialize_inner();
}

fn initialize_inner() -> &'static Location<'static> {
    Location::caller()
}

fn main() {
    let j1 = spawn(initialize);
    let j2 = spawn(initialize);
    j1.join().unwrap();
    j2.join().unwrap();
}
