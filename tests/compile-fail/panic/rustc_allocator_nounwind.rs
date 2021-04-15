#![feature(rustc_attrs, c_unwind)]

#[rustc_allocator_nounwind]
extern "C-unwind" fn nounwind() {
    panic!();
}

fn main() {
    nounwind(); //~ ERROR unwinding past a stack frame that does not allow unwinding
}
