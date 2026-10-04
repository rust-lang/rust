use std::thread::spawn;

fn initialize() {
    initialize_inner(&mut || false)
}

fn initialize_inner(_init: &mut dyn FnMut() -> bool) {}

fn main() {
    let j1 = spawn(initialize);
    let j2 = spawn(initialize);
    j1.join().unwrap();
    j2.join().unwrap();
}
