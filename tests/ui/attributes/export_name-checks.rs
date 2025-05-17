#![crate_type = "lib"]

// rdtsc is an existing LLVM intrinsic
#[export_name = "llvm.x86.rdtsc"]
//~^ ERROR: exported symbol name must not start with `llvm.`
pub unsafe fn foo(a: u8) -> u8 {
    2 * a
}

// qwerty is not a real llvm intrinsic
#[export_name = "llvm.x86.qwerty"]
//~^ ERROR: exported symbol name must not start with `llvm.`
pub unsafe fn bar(a: u8) -> u8 {
    2 * a
}

#[export_name="ab\0cd"] //~ ERROR `export_name` may not contain null characters
pub fn qux() {}
