#![feature(rustc_attrs, fn_align, static_align)]

#[repr(align(UNKNOWN_ALIGN))]
//~^ ERROR const item paths in builtin attributes are experimental
struct ReprAlign;

#[repr(packed(UNKNOWN_PACK))]
//~^ ERROR const item paths in builtin attributes are experimental
struct ReprPacked(u32);

#[rustc_align(UNKNOWN_FN_ALIGN)]
//~^ ERROR const item paths in builtin attributes are experimental
fn aligned_fn() {}

#[rustc_align_static(UNKNOWN_STATIC_ALIGN)]
//~^ ERROR const item paths in builtin attributes are experimental
static ALIGNED_STATIC: u64 = 0;

fn main() {
    let _ = ALIGNED_STATIC;
}
