// aux-build:glob-conflict.rs

extern crate glob_conflict;

fn main() {
    glob_conflict::f(); //~ ERROR cannot find function `f` in module `glob_conflict`
    glob_conflict::glob::f(); //~ ERROR cannot find function `f` in module `glob_conflict::glob`
}
