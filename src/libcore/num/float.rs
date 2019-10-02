// HACK(https://github.com/rust-lang/rust/issues/62785): uefi targets need special LLVM support
// unless we emit the _fltused
#[cfg(target_os = "uefi")]
#[no_mangle]
#[used]
static _fltused: i32 = 0;
