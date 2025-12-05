//@ check-pass
use std::ptr;

// see https://github.com/rust-lang/rust/issues/125833
// notionally, taking the address of an extern static is a safe operation,
// as we only point at it instead of generating a true reference to it

// it may potentially induce linker errors, but the safety of that is not about taking addresses!
// any safety obligation of the extern static's correctness in declaration is on the extern itself,
// see RFC 3484 for more on that: https://rust-lang.github.io/rfcs/3484-unsafe-extern-blocks.html

extern "C" {
    static THERE: u8;
    static mut SOMEWHERE: u8;
}

fn main() {
    let ptr2there = ptr::addr_of!(THERE);
    let ptr2somewhere = ptr::addr_of!(SOMEWHERE);
    let ptr2somewhere = ptr::addr_of_mut!(SOMEWHERE);

    // testing both addr_of and the expression it directly expands to
    let raw2there = &raw const THERE;
    let raw2somewhere = &raw const SOMEWHERE;
    let raw2somewhere = &raw mut SOMEWHERE;
}
