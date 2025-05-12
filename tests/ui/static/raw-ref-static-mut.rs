//@ check-pass
use std::ptr;

// see https://github.com/rust-lang/rust/issues/125833
// notionally, taking the address of a static mut is a safe operation,
// as we only point at it instead of generating a true reference to it
static mut NOWHERE: usize = 0;

fn main() {
    let p2nowhere = ptr::addr_of!(NOWHERE);
    let p2nowhere = ptr::addr_of_mut!(NOWHERE);

    // testing both addr_of and the expression it directly expands to
    let raw2nowhere  = &raw const NOWHERE;
    let raw2nowhere  = &raw mut NOWHERE;
}
