//@ only-windows

mod not_a_real_file; //~ ERROR file not found for module `not_a_real_file`
//~^ HELP to create the module `not_a_real_file`, create file

fn main() {
    assert_eq!(mod_file_aux::bar(), 10);
    //~^ ERROR cannot find item `mod_file_aux`
}
