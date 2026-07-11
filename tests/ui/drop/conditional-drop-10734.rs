//! Regression test for https://github.com/rust-lang/rust/issues/10734

//@ run-pass
#![allow(non_upper_case_globals)]

static mut drop_count: usize = 0;

fn increment_drop_count() {
    unsafe {
        let drop_count = &raw mut drop_count;
        drop_count.write(drop_count.read() + 1);
    }
}

fn get_drop_count() -> usize {
    unsafe { (&raw const drop_count).read() }
}

struct Foo {
    dropped: bool
}

impl Drop for Foo {
    fn drop(&mut self) {
        // Test to make sure we haven't dropped already
        assert!(!self.dropped);
        self.dropped = true;
        // And record the fact that we dropped for verification later
        increment_drop_count();
    }
}

pub fn main() {
    // An `if true { expr }` statement should compile the same as `{ expr }`.
    if true {
        let _a = Foo{ dropped: false };
    }
    // Check that we dropped already (as expected from a `{ expr }`).
    assert_eq!(get_drop_count(), 1);

    // An `if false {} else { expr }` statement should compile the same as `{ expr }`.
    if false {
        panic!();
    } else {
        let _a = Foo{ dropped: false };
    }
    // Check that we dropped already (as expected from a `{ expr }`).
    assert_eq!(get_drop_count(), 2);
}
