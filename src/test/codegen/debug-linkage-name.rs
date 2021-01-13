// Verifies that linkage name is omitted when it is
// the same as variable / function name.
//
// compile-flags: -C no-prepopulate-passes
// compile-flags: -C debuginfo=2
#![crate_type = "lib"]

pub mod xyz {
    // CHECK: !DIGlobalVariable(name: "A",
    // CHECK:                   linkageName:
    // CHECK-SAME:              line: 12,
    pub static A: u32 = 1;

    // CHECK: !DIGlobalVariable(name: "B",
    // CHECK-NOT:               linkageName:
    // CHECK-SAME:              line: 18,
    #[no_mangle]
    pub static B: u32 = 2;

    // CHECK: !DIGlobalVariable(name: "C",
    // CHECK-NOT:               linkageName:
    // CHECK-SAME:              line: 24,
    #[export_name = "C"]
    pub static C: u32 = 2;

    // CHECK: !DISubprogram(name: "e",
    // CHECK:               linkageName:
    // CHECK-SAME:          line: 29,
    pub extern "C" fn e() {}

    // CHECK: !DISubprogram(name: "f",
    // CHECK-NOT:           linkageName:
    // CHECK-SAME:          line: 35,
    #[no_mangle]
    pub extern "C" fn f() {}

    // CHECK: !DISubprogram(name: "g",
    // CHECK-NOT:           linkageName:
    // CHECK-SAME:          line: 41,
    #[export_name = "g"]
    pub extern "C" fn g() {}
}
