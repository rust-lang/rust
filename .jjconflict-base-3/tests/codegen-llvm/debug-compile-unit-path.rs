//@ compile-flags: -g --remap-path-prefix={{cwd}}=/cwd/ --remap-path-prefix={{src-base}}=/base/
//
//
// Ensure that we remap the compile unit directory and that we set it to the compilers current
// working directory and not something else.
#![crate_type = "rlib"]

// CHECK-DAG: [[FILE:![0-9]*]] = !DIFile(filename: "/base/debug-compile-unit-path.rs{{.*}}", directory: "/cwd/")
// CHECK-DAG: {{![0-9]*}} = distinct !DICompileUnit({{.*}}file: [[FILE]]
