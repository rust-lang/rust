//@aux-build:proc_macros.rs

#![warn(clippy::empty_loop)]

extern crate proc_macros;
use proc_macros::{external, inline_macros};

fn should_trigger() {
    loop {}
    //~^ empty_loop
    #[allow(clippy::never_loop)]
    loop {
        loop {}
        //~^ empty_loop
    }

    #[allow(clippy::never_loop)]
    'outer: loop {
        'inner: loop {}
        //~^ empty_loop
    }
}

#[inline_macros]
fn should_not_trigger() {
    #[allow(clippy::never_loop)]
    loop {
        panic!("This is fine")
    }
    let ten_millis = std::time::Duration::from_millis(10);
    loop {
        std::thread::sleep(ten_millis)
    }

    #[allow(clippy::never_loop)]
    'outer: loop {
        'inner: loop {
            break 'inner;
        }
        break 'outer;
    }

    // Make sure `allow` works for this lint
    #[allow(clippy::empty_loop)]
    loop {}

    // We don't lint loops inside macros
    inline!(loop {});

    // We don't lint external macros
    external!(loop {});
}

fn main() {}
