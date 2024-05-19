//@ aux-build: foreign-generic-mismatch.rs

extern crate foreign_generic_mismatch;

fn main() {
    foreign_generic_mismatch::const_arg::<()>();
    //~^ ERROR function takes 2 generic arguments but 1 generic argument was supplied
    foreign_generic_mismatch::lt_arg::<'static, 'static>();
    //~^ ERROR function takes 1 lifetime argument but 2 lifetime arguments were supplied
}
