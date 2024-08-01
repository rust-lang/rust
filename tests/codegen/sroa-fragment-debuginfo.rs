//@ compile-flags: -g -Zmir-opt-level=0 -Zmir-enable-passes=+ScalarReplacementOfAggregates
//@ compile-flags: -Cno-prepopulate-passes
//
// Tested offsets are only correct for x86_64.
//@ only-x86_64

#![crate_type = "lib"]

pub struct ExtraSlice<'input> {
    slice: &'input [u8],
    extra: u32,
}

#[no_mangle]
pub fn extra(s: &[u8]) {
    // CHECK: void @extra(
    // CHECK: %slice.dbg.spill1 = alloca [4 x i8],
    // CHECK: %slice.dbg.spill = alloca [16 x i8],
    // CHECK: %s.dbg.spill = alloca [16 x i8],
    // CHECK: dbg{{.}}declare({{(metadata )?}}ptr %s.dbg.spill, {{(metadata )?}}![[S_EXTRA:.*]], {{(metadata )?}}!DIExpression()
    // CHECK: dbg{{.}}declare({{(metadata )?}}ptr %slice.dbg.spill, {{(metadata )?}}![[SLICE_EXTRA:.*]], {{(metadata )?}}!DIExpression(DW_OP_LLVM_fragment, 0, 128)
    // CHECK: dbg{{.}}declare({{(metadata )?}}ptr %slice.dbg.spill1, {{(metadata )?}}![[SLICE_EXTRA]], {{(metadata )?}}!DIExpression(DW_OP_LLVM_fragment, 128, 32)
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

    // CHECK: void @zst(
    // CHECK-NOT: dbg{{.}}declare({{(metadata )?}}ptr %slice.dbg.spill, {{(metadata )?}}!{}, {{(metadata )?}}!DIExpression(DW_OP_LLVM_fragment,
    // CHECK: dbg{{.}}declare({{(metadata )?}}ptr %{{.*}}, {{(metadata )?}}![[SLICE_ZST:.*]], {{(metadata )?}}!DIExpression()
    // CHECK-NOT: dbg{{.}}declare({{(metadata )?}}ptr %{{.*}}, {{(metadata )?}}![[SLICE_ZST]],
    let slice = ZstSlice { slice: s, extra: Zst };
}

// CHECK: ![[S_EXTRA]] = !DILocalVariable(name: "s",
// CHECK: ![[SLICE_EXTRA]] = !DILocalVariable(name: "slice",
