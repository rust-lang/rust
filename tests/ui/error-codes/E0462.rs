//@ aux-build:found-staticlib.rs

//@ normalize-stderr: "\.nll/" -> "/"
//@ normalize-stderr: "\\\?\\" -> ""
//@ normalize-stderr: "(lib)?found_staticlib\.[a-z]+" -> "libfound_staticlib.somelib"

extern crate found_staticlib; //~ ERROR E0462

fn main() {
    found_staticlib::foo();
}
