//@ run-pass

// Testing that a plain .rs file can load modules from other source files

#[path = "mod_file_aux.rs"]
mod m;

pub fn main() {
    assert_eq!(m::foo(), 10);
}
