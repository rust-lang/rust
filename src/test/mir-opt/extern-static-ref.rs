#![feature(never_type)]
#![feature(raw_ref_macros)]
#![feature(raw_ref_op)]

extern "C" {
    static FOO: !;
}

mod foo {
    #[no_mangle]
    static FOO: u32 = 5;
}

#[inline(never)]
fn bar<T>(_: T) {}

// EMIT_MIR extern_static_ref.main.PreCodegen.after.mir
fn main() {
    bar(unsafe { core::ptr::raw_const!(FOO) });
    bar(unsafe { &raw const FOO });
}
