// EMIT_MIR receiver_ptr_mutability.main.mir_map.0.mir

#![feature(arbitrary_self_types)]

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
