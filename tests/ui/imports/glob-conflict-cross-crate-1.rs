//@ aux-build:glob-conflict.rs

extern crate glob_conflict;

fn main() {
    glob_conflict::f(); //~ ERROR cannot find function `f`
    //^ FIXME: `glob_conflict::f` should raise an
    // ambiguity error instead of a not found error.
    glob_conflict::glob::f(); //~ ERROR cannot find function `f`
    //^ FIXME: `glob_conflict::glob::f` should raise an
    // ambiguity error instead of a not found error.
}
