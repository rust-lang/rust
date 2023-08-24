// compile-flags: -g -Zmir-opt-level=0 -Zmir-enable-passes=+ScalarReplacementOfAggregates
// compile-flags: -Cno-prepopulate-passes

#![crate_type = "lib"]

pub struct ExtraSlice<'input> {
    slice: &'input [u8],
    extra: u32,
}

#[no_mangle]
pub fn extra(s: &[u8]) {
// CHECK: void @extra(
// CHECK: %slice.dbg.spill1 = alloca i32,
// CHECK: %slice.dbg.spill = alloca { ptr, i64 },
// CHECK: %s.dbg.spill = alloca { ptr, i64 },
// CHECK: call void @llvm.dbg.declare(metadata ptr %s.dbg.spill, metadata ![[S_EXTRA:.*]], metadata !DIExpression()),
// CHECK: call void @llvm.dbg.declare(metadata ptr %slice.dbg.spill, metadata ![[SLICE_EXTRA:.*]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 128)),
// CHECK: call void @llvm.dbg.declare(metadata ptr %slice.dbg.spill1, metadata ![[SLICE_EXTRA]], metadata !DIExpression(DW_OP_LLVM_fragment, 128, 32)),
    let slice = ExtraSlice { slice: s, extra: s.len() as u32 };
}

struct Zst;

pub struct ZstSlice<'input> {
    slice: &'input [u8],
    extra: Zst,
}

#[no_mangle]
pub fn zst(s: &[u8]) {
// CHECK: void @zst(
// CHECK: %slice.dbg.spill1 = alloca { ptr, i64 },
// CHECK: %slice.dbg.spill = alloca %Zst,
// CHECK: %s.dbg.spill = alloca { ptr, i64 },
// CHECK: call void @llvm.dbg.declare(metadata ptr %s.dbg.spill, metadata ![[S_ZST:.*]], metadata !DIExpression()),
// CHECK: call void @llvm.dbg.declare(metadata ptr %slice.dbg.spill, metadata ![[SLICE_ZST:.*]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 0)),
// CHECK: call void @llvm.dbg.declare(metadata ptr %slice.dbg.spill1, metadata ![[SLICE_ZST]], metadata !DIExpression()),
    let slice = ZstSlice { slice: s, extra: Zst };
}

// CHECK: ![[S_EXTRA]] = !DILocalVariable(name: "s",
// CHECK: ![[SLICE_EXTRA]] = !DILocalVariable(name: "slice",
// CHECK: ![[S_ZST]] = !DILocalVariable(name: "s",
// CHECK: ![[SLICE_ZST]] = !DILocalVariable(name: "slice",
