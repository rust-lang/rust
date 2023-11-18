// aux-build:glob-conflict.rs

extern crate glob_conflict;

fn main() {
    glob_conflict::f(); //~ ERROR `f` is ambiguous
    glob_conflict::glob::f(); //~ ERROR `f` is ambiguous
}
