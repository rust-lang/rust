//@ needs-asm-support
//@ ignore-nvptx64
//@ ignore-spirv

#![feature(naked_functions)]
#![feature(asm_unwind, linkage)]
#![crate_type = "lib"]

use std::arch::{asm, naked_asm};

#[naked]
pub unsafe extern "C" fn inline_asm_macro() {
    asm!("", options(raw));
    //~^ERROR the `asm!` macro is not allowed in naked functions
}

#[repr(C)]
pub struct P {
    x: u8,
    y: u16,
}

#[naked]
pub unsafe extern "C" fn patterns(
    mut a: u32,
    //~^ ERROR patterns not allowed in naked function parameters
    &b: &i32,
    //~^ ERROR patterns not allowed in naked function parameters
    (None | Some(_)): Option<std::ptr::NonNull<u8>>,
    //~^ ERROR patterns not allowed in naked function parameters
    P { x, y }: P,
    //~^ ERROR patterns not allowed in naked function parameters
) {
    naked_asm!("")
}

#[naked]
pub unsafe extern "C" fn inc(a: u32) -> u32 {
    //~^ ERROR naked functions must contain a single `naked_asm!` invocation
    a + 1
    //~^ ERROR referencing function parameters is not allowed in naked functions
}

#[naked]
#[allow(asm_sub_register)]
pub unsafe extern "C" fn inc_asm(a: u32) -> u32 {
    naked_asm!("/* {0} */", in(reg) a)
    //~^ ERROR the `in` operand cannot be used with `naked_asm!`
}

#[naked]
pub unsafe extern "C" fn inc_closure(a: u32) -> u32 {
    //~^ ERROR naked functions must contain a single `naked_asm!` invocation
    (|| a + 1)()
}

#[naked]
pub unsafe extern "C" fn unsupported_operands() {
    //~^ ERROR naked functions must contain a single `naked_asm!` invocation
    let mut a = 0usize;
    let mut b = 0usize;
    let mut c = 0usize;
    let mut d = 0usize;
    let mut e = 0usize;
    const F: usize = 0usize;
    static G: usize = 0usize;
    naked_asm!("/* {0} {1} {2} {3} {4} {5} {6} */",
         in(reg) a,
         //~^ ERROR the `in` operand cannot be used with `naked_asm!`
         inlateout(reg) b,
         inout(reg) c,
         lateout(reg) d,
         out(reg) e,
         const F,
         sym G,
    );
}

#[naked]
pub extern "C" fn missing_assembly() {
    //~^ ERROR naked functions must contain a single `naked_asm!` invocation
}

#[naked]
pub extern "C" fn too_many_asm_blocks() {
    //~^ ERROR naked functions must contain a single `naked_asm!` invocation
    unsafe {
        naked_asm!("", options(noreturn));
        //~^ ERROR the `noreturn` option cannot be used with `naked_asm!`
        naked_asm!("");
    }
}

pub fn outer(x: u32) -> extern "C" fn(usize) -> usize {
    #[naked]
    pub extern "C" fn inner(y: usize) -> usize {
        //~^ ERROR naked functions must contain a single `naked_asm!` invocation
        *&y
        //~^ ERROR referencing function parameters is not allowed in naked functions
    }
    inner
}

#[naked]
unsafe extern "C" fn invalid_options() {
    naked_asm!("", options(nomem, preserves_flags));
    //~^ ERROR the `nomem` option cannot be used with `naked_asm!`
    //~| ERROR the `preserves_flags` option cannot be used with `naked_asm!`
}

#[naked]
unsafe extern "C" fn invalid_options_continued() {
    naked_asm!("", options(readonly, nostack), options(pure));
    //~^ ERROR the `readonly` option cannot be used with `naked_asm!`
    //~| ERROR the `nostack` option cannot be used with `naked_asm!`
    //~| ERROR the `pure` option cannot be used with `naked_asm!`
}

#[naked]
unsafe extern "C" fn invalid_may_unwind() {
    naked_asm!("", options(may_unwind));
    //~^ ERROR the `may_unwind` option cannot be used with `naked_asm!`
}

#[naked]
pub unsafe fn default_abi() {
    //~^ WARN Rust ABI is unsupported in naked functions
    naked_asm!("");
}

#[naked]
pub unsafe fn rust_abi() {
    //~^ WARN Rust ABI is unsupported in naked functions
    naked_asm!("");
}

#[naked]
pub extern "C" fn valid_a<T>() -> T {
    unsafe {
        naked_asm!("");
    }
}

#[naked]
pub extern "C" fn valid_b() {
    unsafe {
        {
            {
                naked_asm!("");
            };
        };
    }
}

#[naked]
pub unsafe extern "C" fn valid_c() {
    naked_asm!("");
}

#[cfg(target_arch = "x86_64")]
#[naked]
pub unsafe extern "C" fn valid_att_syntax() {
    naked_asm!("", options(att_syntax));
}

#[naked]
#[naked]
pub unsafe extern "C" fn allow_compile_error(a: u32) -> u32 {
    compile_error!("this is a user specified error")
    //~^ ERROR this is a user specified error
}

#[naked]
pub unsafe extern "C" fn allow_compile_error_and_asm(a: u32) -> u32 {
    compile_error!("this is a user specified error");
    //~^ ERROR this is a user specified error
    naked_asm!("")
}

#[naked]
pub unsafe extern "C" fn invalid_asm_syntax(a: u32) -> u32 {
    naked_asm!(invalid_syntax)
    //~^ ERROR asm template must be a string literal
}

// this previously ICE'd, see https://github.com/rust-lang/rust/issues/124375
#[naked]
pub unsafe extern "C" fn naked_with_args_and_return(a: isize, b: isize) -> isize {
    naked_asm!("lea rax, [rdi + rsi]", "ret");
}

#[cfg(target_arch = "x86_64")]
#[cfg_attr(target_pointer_width = "64", no_mangle)]
#[naked]
pub unsafe extern "C" fn compatible_cfg_attributes() {
    naked_asm!("", options(att_syntax));
}

#[allow(dead_code)]
#[warn(dead_code)]
#[deny(dead_code)]
#[forbid(dead_code)]
#[naked]
pub unsafe extern "C" fn compatible_diagnostic_attributes() {
    naked_asm!("", options(raw));
}

#[deprecated = "test"]
#[naked]
pub unsafe extern "C" fn compatible_deprecated_attributes() {
    naked_asm!("", options(raw));
}

#[cfg(target_arch = "x86_64")]
#[must_use]
#[naked]
pub unsafe extern "C" fn compatible_must_use_attributes() -> u64 {
    naked_asm!(
        "
        mov rax, 42
        ret
        ",
    )
}

#[export_name = "exported_function_name"]
#[link_section = ".custom_section"]
#[no_mangle]
#[naked]
pub unsafe extern "C" fn compatible_ffi_attributes_1() {
    naked_asm!("", options(raw));
}

#[cold]
#[naked]
pub unsafe extern "C" fn compatible_codegen_attributes() {
    naked_asm!("", options(raw));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[naked]
pub unsafe extern "C" fn compatible_target_feature() {
    naked_asm!("");
}

#[doc = "foo bar baz"]
/// a doc comment
// a normal comment
#[doc(alias = "ADocAlias")]
#[naked]
pub unsafe extern "C" fn compatible_doc_attributes() {
    naked_asm!("", options(raw));
}

#[linkage = "external"]
#[naked]
pub unsafe extern "C" fn compatible_linkage() {
    naked_asm!("", options(raw));
}
