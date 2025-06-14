//@ compile-flags: -Z merge-functions=disabled

//@ revisions: x86-64
//@[x86-64] compile-flags: --target x86_64-unknown-linux-gnu
//@[x86-64] needs-llvm-components: x86

//@ revisions: x86-32
//@[x86-32] compile-flags: --target i686-unknown-linux-gnu
//@[x86-32] needs-llvm-components: x86

//@ revisions: x86-32-nosse
//@[x86-32-nosse] compile-flags: --target i586-unknown-linux-gnu
//@[x86-32-nosse] needs-llvm-components: x86

#![feature(no_core, lang_items, rustc_attrs, repr_simd)]
#![no_core]
#![crate_type = "lib"]

#[lang = "sized"]
trait Sized {}

#[lang = "copy"]
trait Copy {}

// Ensure this type is passed without ptr indirection on targets that
// require SSE2.
#[repr(simd)]
pub struct Sse([f32; 4]);

// FIXME: due to #139029 we are passing them all indirectly.
// x86-64: void @sse_id(ptr{{( [^,]*)?}} sret([16 x i8]){{( .*)?}}, ptr{{( [^,]*)?}})
// x86-32: void @sse_id(ptr{{( [^,]*)?}} sret([16 x i8]){{( .*)?}}, ptr{{( [^,]*)?}})
// x86-32-nosse: void @sse_id(ptr{{( [^,]*)?}} sret([16 x i8]){{( .*)?}}, ptr{{( [^,]*)?}})
#[no_mangle]
pub fn sse_id(x: Sse) -> Sse {
    x
}
