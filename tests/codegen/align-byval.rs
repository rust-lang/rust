// revisions:m68k wasm x86_64-linux x86_64-windows i686-linux i686-windows

//[m68k] compile-flags: --target m68k-unknown-linux-gnu
//[m68k] needs-llvm-components: m68k
//[wasm] compile-flags: --target wasm32-unknown-emscripten
//[wasm] needs-llvm-components: webassembly
//[x86_64-linux] compile-flags: --target x86_64-unknown-linux-gnu
//[x86_64-linux] needs-llvm-components: x86
//[x86_64-windows] compile-flags: --target x86_64-pc-windows-msvc
//[x86_64-windows] needs-llvm-components: x86
//[i686-linux] compile-flags: --target i686-unknown-linux-gnu
//[i686-linux] needs-llvm-components: x86
//[i686-windows] compile-flags: --target i686-pc-windows-msvc
//[i686-windows] needs-llvm-components: x86

// Tests that `byval` alignment is properly specified (#80127).
// The only targets that use `byval` are m68k, wasm, x86-64, and x86.
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
pub struct NaturalAlign2 {
    a: [i16; 16],
    b: i16,
}

#[repr(C)]
#[repr(align(4))]
pub struct ForceAlign4 {
    a: [i8; 16],
    b: i8,
}

// on i686-windows, this should be passed on stack using `byval`
#[repr(C)]
pub struct NaturalAlign8 {
    a: i64,
    b: i64,
    c: i64
}

// on i686-windows, this is passed by reference (because alignment is >4 and requested/forced),
// even though it has the exact same layout as `NaturalAlign8` (!!!)
#[repr(C)]
#[repr(align(8))]
pub struct ForceAlign8 {
    a: i64,
    b: i64,
    c: i64
}

#[repr(C)]
#[repr(align(16))]
pub struct ForceAlign16 {
    a: [i32; 16],
    b: i8
}

extern "C" {
    // m68k: declare void @natural_align_2({{.*}}byval(%NaturalAlign2) align 2{{.*}})

    // wasm: declare void @natural_align_2({{.*}}byval(%NaturalAlign2) align 2{{.*}})

    // x86_64-linux: declare void @natural_align_2({{.*}}byval(%NaturalAlign2) align 2{{.*}})

    // x86_64-windows: declare void @natural_align_2(
    // x86_64-windows-NOT: byval
    // x86_64-windows-SAME: align 2{{.*}})

    // i686-linux: declare void @natural_align_2({{.*}}byval(%NaturalAlign2) align 4{{.*}})

    // i686-windows: declare void @natural_align_2({{.*}}byval(%NaturalAlign2) align 4{{.*}})
    fn natural_align_2(a: NaturalAlign2);

    // m68k: declare void @force_align_4({{.*}}byval(%ForceAlign4) align 4{{.*}})

    // wasm: declare void @force_align_4({{.*}}byval(%ForceAlign4) align 4{{.*}})

    // x86_64-linux: declare void @force_align_4({{.*}}byval(%ForceAlign4) align 4{{.*}})

    // x86_64-windows: declare void @force_align_4(
    // x86_64-windows-NOT: byval
    // x86_64-windows-SAME: align 4{{.*}})

    // i686-linux: declare void @force_align_4({{.*}}byval(%ForceAlign4) align 4{{.*}})

    // i686-windows: declare void @force_align_4({{.*}}byval(%ForceAlign4) align 4{{.*}})
    fn force_align_4(b: ForceAlign4);

    // m68k: declare void @natural_align_8({{.*}}byval(%NaturalAlign8) align 4{{.*}})

    // wasm: declare void @natural_align_8({{.*}}byval(%NaturalAlign8) align 8{{.*}})

    // x86_64-linux: declare void @natural_align_8({{.*}}byval(%NaturalAlign8) align 8{{.*}})

    // x86_64-windows: declare void @natural_align_8(
    // x86_64-windows-NOT: byval
    // x86_64-windows-SAME: align 8{{.*}})

    // i686-linux: declare void @natural_align_8({{.*}}byval(%NaturalAlign8) align 4{{.*}})

    // i686-windows: declare void @natural_align_8({{.*}}byval(%NaturalAlign8) align 4{{.*}})
    fn natural_align_8(x: NaturalAlign8);

    // m68k: declare void @force_align_8({{.*}}byval(%ForceAlign8) align 8{{.*}})

    // wasm: declare void @force_align_8({{.*}}byval(%ForceAlign8) align 8{{.*}})

    // x86_64-linux: declare void @force_align_8({{.*}}byval(%ForceAlign8) align 8{{.*}})

    // x86_64-windows: declare void @force_align_8(
    // x86_64-windows-NOT: byval
    // x86_64-windows-SAME: align 8{{.*}})

    // i686-linux: declare void @force_align_8({{.*}}byval(%ForceAlign8) align 4{{.*}})

    // i686-windows: declare void @force_align_8(
    // i686-windows-NOT: byval
    // i686-windows-SAME: align 8{{.*}})
    fn force_align_8(y: ForceAlign8);

    // m68k: declare void @force_align_16({{.*}}byval(%ForceAlign16) align 16{{.*}})

    // wasm: declare void @force_align_16({{.*}}byval(%ForceAlign16) align 16{{.*}})

    // x86_64-linux: declare void @force_align_16({{.*}}byval(%ForceAlign16) align 16{{.*}})

    // x86_64-windows: declare void @force_align_16(
    // x86_64-windows-NOT: byval
    // x86_64-windows-SAME: align 16{{.*}})

    // i686-linux: declare void @force_align_16({{.*}}byval(%ForceAlign16) align 4{{.*}})

    // i686-windows: declare void @force_align_16(
    // i686-windows-NOT: byval
    // i686-windows-SAME: align 16{{.*}})
    fn force_align_16(z: ForceAlign16);
}

pub unsafe fn main(
    a: NaturalAlign2, b: ForceAlign4,
    x: NaturalAlign8, y: ForceAlign8, z: ForceAlign16
) {
    natural_align_2(a);
    force_align_4(b);
    natural_align_8(x);
    force_align_8(y);
    force_align_16(z);
}
