extern crate a_dylib;
extern crate a_rlib;
extern crate b_dylib;

fn main() {
    a_rlib::hello();
    a_dylib::hello();
    b_dylib::hello();
}
