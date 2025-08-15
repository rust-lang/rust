mod mod_file_disambig_aux; //~ ERROR file for module `mod_file_disambig_aux` found at both

fn main() {
    assert_eq!(mod_file_aux::bar(), 10);
    //~^ ERROR: cannot find module or crate `mod_file_aux`
    //~| NOTE: use of unresolved module or unlinked crate `mod_file_aux`
}
