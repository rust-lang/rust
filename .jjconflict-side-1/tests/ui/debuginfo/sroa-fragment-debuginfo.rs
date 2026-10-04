// Verify that we do not trigger an LLVM assertion by creating zero-sized DWARF fragments.
//
//@ build-pass
//@ compile-flags: -g -Zmir-opt-level=0 -Zmir-enable-passes=+ScalarReplacementOfAggregates
//@ compile-flags: -Cno-prepopulate-passes

#![crate_type = "lib"]

pub struct ExtraSlice<'input> {
    slice: &'input [u8],
    extra: u32,
}

#[no_mangle]
pub fn extra(s: &[u8]) {
    let slice = ExtraSlice { slice: s, extra: s.len() as u32 };
}

struct Zst;

pub struct ZstSlice<'input> {
    slice: &'input [u8],
    extra: Zst,
}

#[no_mangle]
pub fn zst(s: &[u8]) {
    // The field `extra` is a ZST. The fragment for the field `slice` encompasses the whole
    // variable, so is not a fragment. In that case, the variable must have no fragment.
    let slice = ZstSlice { slice: s, extra: Zst };
}
