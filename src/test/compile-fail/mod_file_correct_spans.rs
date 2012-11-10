// Testing that the codemap is maintained correctly when parsing mods from external files

mod mod_file_aux;

fn main() {
    assert mod_file_aux::bar() == 10; //~ ERROR unresolved name
}