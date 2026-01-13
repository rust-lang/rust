//@ check-pass
//@ edition:2015
//@ aux-build:glob-conflict.rs

extern crate glob_conflict;

fn main() {
    glob_conflict::f(); //~ WARN `f` is ambiguous
                        //~| WARN this was previously accepted
    glob_conflict::glob::f(); //~ WARN `f` is ambiguous
                              //~| WARN this was previously accepted
}
