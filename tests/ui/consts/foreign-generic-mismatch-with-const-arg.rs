// aux-build: foreign-generic-mismatch-with-const-arg.rs

extern crate foreign_generic_mismatch_with_const_arg;

fn main() {
    foreign_generic_mismatch_with_const_arg::test::<1>();
    //~^ ERROR function takes 2 generic arguments but 1 generic argument was supplied
}
