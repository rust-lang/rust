// compile-flags: -g -Zmir-opt-level=0 -Zmir-enable-passes=+ScalarReplacementOfAggregates
// compile-flags: -Cno-prepopulate-passes

#![crate_type = "lib"]

pub struct Endian;

#[allow(dead_code)]
pub struct EndianSlice<'input> {
    slice: &'input [u8],
    endian: Endian,
}

#[no_mangle]
pub fn test(s: &[u8]) {
// CHECK: void @test(
// CHECK: %slice.dbg.spill1 = alloca { ptr, i64 },
// CHECK: %slice.dbg.spill = alloca %Endian,
// CHECK: %s.dbg.spill = alloca { ptr, i64 },
// CHECK: call void @llvm.dbg.declare(metadata ptr %s.dbg.spill, metadata ![[S:.*]], metadata !DIExpression()),
// CHECK: call void @llvm.dbg.declare(metadata ptr %slice.dbg.spill, metadata ![[SLICE:.*]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 0)),
// CHECK: call void @llvm.dbg.declare(metadata ptr %slice.dbg.spill1, metadata ![[SLICE]], metadata !DIExpression()),
    let slice = EndianSlice { slice: s, endian: Endian };
}

// CHECK: ![[S]] = !DILocalVariable(name: "s",
// CHECK: ![[SLICE]] = !DILocalVariable(name: "slice",
