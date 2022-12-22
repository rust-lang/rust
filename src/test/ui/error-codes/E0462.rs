// aux-build:found-staticlib.rs

extern crate found_staticlib; //~ ERROR E0462

fn main() {
    found_staticlib::foo();
}
