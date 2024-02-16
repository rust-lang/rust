//@ aux-build:other_file.rs
//@ compile-flags: --error-format human-annotate-rs -Z unstable-options

extern crate other_file;

fn main() {
    other_file::WithPrivateMethod.private_method();
}
