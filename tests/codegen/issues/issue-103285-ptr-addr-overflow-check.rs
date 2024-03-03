//@ compile-flags: -O -C debug-assertions=yes

#![crate_type = "lib"]
#![feature(strict_provenance)]

#[no_mangle]
pub fn test(src: *const u8, dst: *const u8) -> usize {
    // CHECK-LABEL: @test(
    // CHECK-NOT: panic
    let src_usize = src.bare_addr();
    let dst_usize = dst.bare_addr();
    if src_usize > dst_usize {
        return src_usize - dst_usize;
    }
    return 0;
}
