//@ aux-build:other_file.rs
//@ compile-flags: --error-format human

extern crate other_file;

fn main() {
    other_file::WithPrivateMethod.private_method();
}
