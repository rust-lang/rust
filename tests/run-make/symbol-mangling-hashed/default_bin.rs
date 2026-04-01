extern crate default_dylib;
extern crate hashed_dylib;
extern crate hashed_rlib;

fn main() {
    hashed_rlib::hrhello();
    hashed_dylib::hdhello();
    default_dylib::ddhello();
}
