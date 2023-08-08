#![feature(rustc_private)]

extern crate libc;

#[link(name = "foo", kind = "static")]
extern "C" {
    fn should_return_one() -> libc::c_int;
}

fn main() {
    let result = unsafe { should_return_one() };

    if result != 1 {
        std::process::exit(255);
    }
}
