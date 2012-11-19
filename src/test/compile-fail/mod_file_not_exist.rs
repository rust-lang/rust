mod not_a_real_file; //~ ERROR not_a_real_file.rs

fn main() {
    assert mod_file_aux::bar() == 10;
}