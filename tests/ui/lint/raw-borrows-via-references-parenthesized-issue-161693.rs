//@ check-pass

// Parenthesized reference-to-pointer casts should produce syntactically valid suggestions.

#![feature(const_volatile)]
#![warn(raw_borrows_via_references)]

const READ: () = unsafe {
    let x = 42i32;
    let y = (&x as *const i32).read_volatile();
    //~^ WARN creating an intermediate reference implies aliasing requirements
    assert!(x == y);
};

const WRITE: () = unsafe {
    let mut x = 42i32;
    (&mut x as *mut i32).write_volatile(13);
    //~^ WARN creating an intermediate reference implies aliasing requirements
    assert!(x == 13);
};

fn main() {}
