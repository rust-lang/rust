//@ only-x86_64
//@ revisions: DEFAULT EXPLICIT
//@[DEFAULT] compile-flags: -C code-model=medium
//@[EXPLICIT] compile-flags: -C code-model=medium -Z large-data-threshold=1024

#![crate_type = "lib"]

// DEFAULT-NOT: !"Large Data Threshold"
// EXPLICIT: !{{[0-9]+}} = !{i32 1, !"Large Data Threshold", i64 1024}
