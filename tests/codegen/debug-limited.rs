// Verify that the limited debuginfo option emits llvm's FullDebugInfo, but no type info.
//
//@ compile-flags: -C debuginfo=limited

#[repr(C)]
struct StructType {
    a: i64,
    b: i32,
}

extern "C" {
    fn creator() -> *mut StructType;
    fn save(p: *const StructType);
}

fn main() {
    unsafe {
        let value: &mut StructType = &mut *creator();
        value.a = 7;
        save(value as *const StructType)
    }
}

// CHECK: !DICompileUnit
// CHECK: emissionKind: FullDebug
// CHECK: !DILocation
// CHECK-NOT: !DIBasicType
