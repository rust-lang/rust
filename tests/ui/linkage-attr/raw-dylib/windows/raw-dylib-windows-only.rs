//@ revisions: elf notelf
//@ [elf] only-elf
//@ [notelf] ignore-windows
//@ [notelf] ignore-elf
//@ compile-flags: --crate-type lib
#[link(name = "foo", kind = "raw-dylib")]
//[notelf]~^ ERROR: link kind `raw-dylib` is only supported on Windows targets
//[elf]~^^ ERROR: link kind `raw-dylib` is unstable on ELF platforms
extern "C" {}
