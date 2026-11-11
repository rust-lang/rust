// Verify that the only debuginfo generated are the line tables.
//
//@ compile-flags: -C debuginfo=line-tables-only

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
// CHECK: emissionKind: LineTablesOnly
// CHECK: !DILocation
// CHECK-NOT: !DIBasicType
