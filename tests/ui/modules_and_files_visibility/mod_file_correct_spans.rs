// Testing that the source_map is maintained correctly when parsing mods from external files

mod mod_file_aux;

fn main() {
    assert!(mod_file_aux::bar() == 10);
    //~^ ERROR cannot find function `bar` in module `mod_file_aux`
}
