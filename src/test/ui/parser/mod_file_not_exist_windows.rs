// only-windows

mod not_a_real_file; //~ ERROR file not found for module `not_a_real_file`
//~^ HELP name the file either not_a_real_file.rs or not_a_real_file\mod.rs inside the directory

fn main() {
    assert_eq!(mod_file_aux::bar(), 10);
}
