//@ run-pass

// Testing that a plain .rs file can load modules from other source files

mod mod_file_aux;

pub fn main() {
    assert_eq!(mod_file_aux::foo(), 10);
}
