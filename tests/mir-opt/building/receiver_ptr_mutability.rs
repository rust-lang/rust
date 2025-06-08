//@ compile-flags: -Zmir-opt-level=0
// skip-filecheck
// EMIT_MIR receiver_ptr_mutability.main.built.after.mir

#![feature(arbitrary_self_types_pointers)]

struct Test {}

impl Test {
    fn x(self: *const Self) {
        println!("x called");
    }
}

fn main() {
    let ptr: *mut Test = std::ptr::null_mut();
    ptr.x();

    // Test autoderefs
    let ptr_ref: &&&&*mut Test = &&&&ptr;
    ptr_ref.x();
}
