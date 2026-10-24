//@ build-fail

#![feature(type_info)]

fn main() {
    const {
        core::mem::type_info::const_env::var_os("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA").unwrap();
        //~^ ERROR: evaluation panicked: called `Option::unwrap()` on a `None` value
    };
}
