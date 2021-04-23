// Issue #14893. Tests that casts from vectors don't behave strangely in the
// presence of the `_` type shorthand notation.
//
// Update: after a change to the way casts are done, we have more type information
// around and so the errors here are no longer exactly the same.
//
// Update: With PR #81479 some of the previously rejected cases are now allowed.
// New test cases added.

struct X {
    y: [u8; 2],
}

fn main() {
    let x1 = X { y: [0, 0] };

    // No longer a type mismatch - the `_` can be fully resolved by type inference.
    let p1: *const u8 = &x1.y as *const _;
    let p1: *mut u8 = &x1.y as *mut _;
    //~^ ERROR: casting `&[u8; 2]` as `*mut u8` is invalid
    let t1: *const [u8; 2] = &x1.y as *const _;
    let t1: *mut [u8; 2] = &x1.y as *mut _;
    //~^ ERROR: casting `&[u8; 2]` as `*mut [u8; 2]` is invalid
    let h1: *const [u8; 2] = &x1.y as *const [u8; 2];
    let t1: *mut [u8; 2] = &x1.y as *mut [u8; 2];
    //~^ ERROR: casting `&[u8; 2]` as `*mut [u8; 2]` is invalid

    let mut x1 = X { y: [0, 0] };

    let p1: *mut u8 = &mut x1.y as *mut _;
    let p2: *const u8 = &mut x1.y as *const _;
    let t1: *mut [u8; 2] = &mut x1.y as *mut _;
    let h1: *mut [u8; 2] = &mut x1.y as *mut [u8; 2];
}
