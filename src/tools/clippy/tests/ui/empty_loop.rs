// aux-build:macro_rules.rs

#![warn(clippy::empty_loop)]

#[macro_use]
extern crate macro_rules;

fn should_trigger() {
    loop {}
    loop {
        loop {}
    }

    'outer: loop {
        'inner: loop {}
    }
}

fn should_not_trigger() {
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
    macro_rules! foo {
        () => {
            loop {}
        };
    }

    // We don't lint external macros
    foofoo!()
}

fn main() {}
