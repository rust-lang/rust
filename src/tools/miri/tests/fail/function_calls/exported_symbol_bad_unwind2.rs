//@revisions: extern_block definition both
//@normalize-stderr-test: "\n +[0-9]+:[^\n]+" -> ""
//@normalize-stderr-test: "\n +at [^\n]+" -> ""
//@[definition,both]error-in-other-file: aborted execution
#![feature(rustc_attrs)]

#[cfg_attr(any(definition, both), rustc_nounwind)]
#[no_mangle]
extern "C-unwind" fn nounwind() {
    panic!();
}

fn main() {
    extern "C-unwind" {
        #[cfg_attr(any(extern_block, both), rustc_nounwind)]
        fn nounwind();
    }
    unsafe { nounwind() }
    //~[extern_block]^ ERROR: unwinding past a stack frame that does not allow unwinding
}
