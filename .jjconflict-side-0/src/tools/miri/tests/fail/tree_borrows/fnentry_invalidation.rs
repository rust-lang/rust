//@compile-flags: -Zmiri-tree-borrows
// This test is the TB counterpart to fail/stacked_borrows/fnentry_invalidation,
// but the SB version passes TB without error.
// An additional write access is inserted so that this test properly fails.

// Test that spans displayed in diagnostics identify the function call, not the function
// definition, as the location of invalidation due to FnEntry retag. Technically the FnEntry retag
// occurs inside the function, but what the user wants to know is which call produced the
// invalidation.

fn main() {
    let mut x = 0i32;
    let z = &mut x as *mut i32;
    unsafe {
        *z = 1;
    }
    x.do_bad();
    unsafe {
        *z = 2; //~ ERROR: /write access through .* is forbidden/
    }
}

trait Bad {
    fn do_bad(&mut self) {
        // who knows
    }
}

impl Bad for i32 {}
