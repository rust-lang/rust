//! Regression test for https://github.com/rust-lang/rust/issues/10734

//@ run-pass
#![allow(non_upper_case_globals)]

// FIXME(static_mut_refs): this could use an atomic
#![allow(static_mut_refs)]

static mut drop_count: usize = 0;

struct Foo {
    dropped: bool
}

impl Drop for Foo {
    fn drop(&mut self) {
        // Test to make sure we haven't dropped already
        assert!(!self.dropped);
        self.dropped = true;
        // And record the fact that we dropped for verification later
        unsafe { drop_count += 1; }
    }
}

pub fn main() {
    // An `if true { expr }` statement should compile the same as `{ expr }`.
    if true {
        let _a = Foo{ dropped: false };
    }
    // Check that we dropped already (as expected from a `{ expr }`).
    unsafe { assert_eq!(drop_count, 1); }

    // An `if false {} else { expr }` statement should compile the same as `{ expr }`.
    if false {
        panic!();
    } else {
        let _a = Foo{ dropped: false };
    }
    // Check that we dropped already (as expected from a `{ expr }`).
    unsafe { assert_eq!(drop_count, 2); }
}
