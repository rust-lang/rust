//@ add-minicore
//@ compile-flags: --target=bpfel-unknown-none -Copt-level=0 -Cdebuginfo=2
//@ needs-llvm-components: bpf
#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;

unsafe extern "C" {
    fn bpf_kfunc(a: u32, b: u64) -> u32;
    fn do_work(x: i32) -> i32;
    static ext_var: u32;
}

// CHECK: @ext_var = external {{.*}} !dbg [[VAR_EXPR:![0-9]+]]

// CHECK-DAG: declare !dbg [[KFUNC:![0-9]+]] {{.*}}@bpf_kfunc(
// CHECK-DAG: declare !dbg [[DOWORK:![0-9]+]] {{.*}}@do_work(

// CHECK: !llvm.dbg.cu

// CHECK-DAG: [[VAR_EXPR]] = !DIGlobalVariableExpression(var: [[VAR:![0-9]+]], expr: !DIExpression())
// CHECK-DAG: [[VAR]] = distinct !DIGlobalVariable(name: "ext_var", {{.*}}isLocal: false, isDefinition: false
// CHECK-DAG: [[KFUNC]] = !DISubprogram(name: "bpf_kfunc", {{.*}}type: [[KFUNC_TY:![0-9]+]], {{.*}}flags: DIFlagPrototyped, spFlags: 0
// CHECK-DAG: [[KFUNC_TY]] = !DISubroutineType(types: [[KFUNC_TYS:![0-9]+]])
// CHECK-DAG: [[KFUNC_TYS]] = !{[[U32:![0-9]+]], [[U32]], [[U64:![0-9]+]]}
// CHECK-DAG: [[U32]] = !DIBasicType(name: "u32", size: 32, encoding: DW_ATE_unsigned)
// CHECK-DAG: [[U64]] = !DIBasicType(name: "u64", size: 64, encoding: DW_ATE_unsigned)
// CHECK-DAG: [[DOWORK]] = !DISubprogram(name: "do_work",

#[no_mangle]
pub fn call_kfunc() -> u32 {
    unsafe { bpf_kfunc(1, 2) }
}

#[no_mangle]
pub fn take_addr() -> unsafe extern "C" fn(i32) -> i32 {
    do_work
}

#[no_mangle]
pub fn read_var() -> u32 {
    unsafe { ext_var }
}
