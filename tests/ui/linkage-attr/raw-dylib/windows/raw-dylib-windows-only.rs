//@ revisions: elf notelf macho
//@ [elf] only-elf
//@ [macho] only-apple
//@ [notelf] ignore-windows
//@ [notelf] ignore-elf
//@ [notelf] ignore-apple
//@ compile-flags: --crate-type lib
#[link(name = "foo", kind = "raw-dylib")]
//[notelf]~^ ERROR: link kind `raw-dylib` is only supported on Windows targets
//[elf]~^^ ERROR: link kind `raw-dylib` is unstable on ELF platforms
//[macho]~^^^ ERROR: link kind `raw-dylib` is unstable on Mach-O platforms
//[macho]~^^^^ ERROR: link kind `raw-dylib` should use the `+verbatim` linkage modifier
extern "C" {}
