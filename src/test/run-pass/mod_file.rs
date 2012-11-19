// xfail-pretty

// Testing that a plain .rs file can load modules from other source files

mod mod_file_aux;

fn main() {
    assert mod_file_aux::foo() == 10;
}