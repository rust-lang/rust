#![feature(rustc_attrs, fn_align, static_align)]

const REPR_ALIGN: usize = 8;
#[repr(align(REPR_ALIGN))]
//~^ ERROR const item paths in builtin attributes are experimental
struct ReprAlign;

const REPR_PACK: usize = 2;
#[repr(packed(REPR_PACK))]
//~^ ERROR const item paths in builtin attributes are experimental
struct ReprPacked(u32);

const FN_ALIGN: usize = 16;
#[rustc_align(FN_ALIGN)]
//~^ ERROR const item paths in builtin attributes are experimental
fn aligned_fn() {}

const STATIC_ALIGN: usize = 16;
#[rustc_align_static(STATIC_ALIGN)]
//~^ ERROR const item paths in builtin attributes are experimental
static ALIGNED_STATIC: u64 = 0;

fn main() {
    let _ = ALIGNED_STATIC;
}
