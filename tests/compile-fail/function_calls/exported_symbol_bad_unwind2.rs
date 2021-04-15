// revisions: extern_block definition both
#![feature(rustc_attrs, c_unwind)]

#[cfg_attr(any(definition, both), rustc_allocator_nounwind)]
#[no_mangle]
extern "C-unwind" fn nounwind() {
    panic!();
}

fn main() {
    extern "C-unwind" {
        #[cfg_attr(any(extern_block, both), rustc_allocator_nounwind)]
        fn nounwind();
    }
    unsafe { nounwind() }
    //[extern_block]~^ ERROR unwinding past a stack frame that does not allow unwinding
    //[definition]~^^ ERROR unwinding past a stack frame that does not allow unwinding
    //[both]~^^^ ERROR unwinding past a stack frame that does not allow unwinding
}
