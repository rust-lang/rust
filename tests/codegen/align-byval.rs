// revisions:m68k wasm x86_64-linux x86_64-windows

//[m68k] compile-flags: --target m68k-unknown-linux-gnu
//[m68k] needs-llvm-components: m68k
//[wasm] compile-flags: --target wasm32-unknown-emscripten
//[wasm] needs-llvm-components: webassembly
//[x86_64-linux] compile-flags: --target x86_64-unknown-linux-gnu
//[x86_64-linux] needs-llvm-components: x86
//[x86_64-windows] compile-flags: --target x86_64-pc-windows-msvc
//[x86_64-windows] needs-llvm-components: x86

// Tests that `byval` alignment is properly specified (#80127).
// The only targets that use `byval` are m68k, wasm, x86-64, and x86. Note that
// x86 has special rules (see #103830), and it's therefore ignored here.
// Note also that Windows mandates a by-ref ABI here, so it does not use byval.

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_std]
#![no_core]

#[lang="sized"] trait Sized { }
#[lang="freeze"] trait Freeze { }
#[lang="copy"] trait Copy { }

impl Copy for i32 {}
impl Copy for i64 {}

#[repr(C)]
#[repr(align(16))]
struct Foo {
    a: [i32; 16],
    b: i8
}

extern "C" {
    // m68k: declare void @f({{.*}}byval(%Foo) align 16{{.*}})

    // wasm: declare void @f({{.*}}byval(%Foo) align 16{{.*}})

    // x86_64-linux: declare void @f({{.*}}byval(%Foo) align 16{{.*}})

    // x86_64-windows: declare void @f(
    // x86_64-windows-NOT: byval
    // x86_64-windows-SAME: align 16{{.*}})
    fn f(foo: Foo);
}

pub fn main() {
    unsafe { f(Foo { a: [1; 16], b: 2 }) }
}
