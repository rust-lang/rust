// xfail-pretty

// Testing that a plain .rs file can load modules from other source files

#[path = "mod_file_aux.rs"]
mod m;

fn main() {
    assert m::foo() == 10;
}