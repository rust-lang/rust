//@ add-minicore
//@ assembly-output: emit-asm
//@ compile-flags: --target bpfel-unknown-none -Copt-level=2 -Cdebuginfo=2 -Zverbose-asm
//@ needs-llvm-components: bpf

#![feature(no_core)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK-LABEL: .section .BTF,

// The string table follows the types, so the names are tied to their entries with
// CHECK-DAG, whose patterns may match in any order within the group.
// BTF_KIND_FUNC with BTF_FUNC_EXTERN linkage is 0xc000002, BTF_KIND_VAR is 0xe000000
// followed by its type and BTF_VAR_GLOBAL_EXTERN (2), and a DATASEC entry starts with
// the type id of the function or variable it describes.

// CHECK-DAG: .ascii "u32" # string offset=[[U32_OFF:[0-9]+]]
// CHECK-DAG: .long [[U32_OFF]] # BTF_KIND_INT(id = [[U32:[0-9]+]])
// CHECK-DAG: .ascii "u64" # string offset=[[U64_OFF:[0-9]+]]
// CHECK-DAG: .long [[U64_OFF]] # BTF_KIND_INT(id = [[U64:[0-9]+]])

// `*mut c_void` must be a pointer to void (type 0), not to the `c_void` enum.
// CHECK-DAG: # BTF_KIND_PTR(id = [[VOID_PTR:[0-9]+]]){{[[:space:]]+}}.long 33554432 # 0x2000000
// CHECK-DAG: # BTF_KIND_FUNC_PROTO(id = [[CALL_PROTO:[0-9]+]]){{[[:space:]]+}}.long 218103810 # 0xd000002
// CHECK-DAG: .ascii "bpf_kfunc_call" # string offset=[[CALL_OFF:[0-9]+]]
// CHECK-DAG: .long [[CALL_OFF]] # BTF_KIND_FUNC(id = [[CALL_ID:[0-9]+]]){{[[:space:]]+}}.long 201326594 # 0xc000002

// CHECK-DAG: .ascii "bpf_kfunc_addr" # string offset=[[ADDR_OFF:[0-9]+]]
// CHECK-DAG: .long [[ADDR_OFF]] # BTF_KIND_FUNC(id = [[ADDR_ID:[0-9]+]]){{[[:space:]]+}}.long 201326594 # 0xc000002

// The BTF name is the symbol name, not the Rust name.
// CHECK-DAG: .ascii "bpf_kfunc_real" # string offset=[[REAL_OFF:[0-9]+]]
// CHECK-DAG: .long [[REAL_OFF]] # BTF_KIND_FUNC(id = [[REAL_ID:[0-9]+]]){{[[:space:]]+}}.long 201326594 # 0xc000002

// CHECK-DAG: .ascii "bpf_ksym_var" # string offset=[[KSYM_OFF:[0-9]+]]
// CHECK-DAG: .long [[KSYM_OFF]] # BTF_KIND_VAR(id = [[KSYM_ID:[0-9]+]]){{[[:space:]]+}}.long 234881024 # 0xe000000

// An extern static without a section still gets a BTF_KIND_VAR, but no DATASEC entry.
// CHECK-DAG: .ascii "plain_extern_var" # string offset=[[PLAIN_OFF:[0-9]+]]
// CHECK-DAG: .long [[PLAIN_OFF]] # BTF_KIND_VAR(id = {{[0-9]+}}){{[[:space:]]+}}.long 234881024 # 0xe000000

// Exactly the three functions and the one variable placed in .ksyms.
// CHECK-DAG: .ascii ".ksyms" # string offset=[[KSYMS_OFF:[0-9]+]]
// CHECK-DAG: .long [[KSYMS_OFF]] # BTF_KIND_DATASEC(id = {{[0-9]+}}){{[[:space:]]+}}.long 251658244 # 0xf000004
// CHECK-DAG: .long [[CALL_ID]]{{[[:space:]]+}}.long bpf_kfunc_call{{$}}
// CHECK-DAG: .long [[ADDR_ID]]{{[[:space:]]+}}.long bpf_kfunc_addr{{$}}
// CHECK-DAG: .long [[REAL_ID]]{{[[:space:]]+}}.long bpf_kfunc_real{{$}}
// CHECK-DAG: .long [[KSYM_ID]]{{[[:space:]]+}}.long bpf_ksym_var{{$}}

// CHECK: .section .BTF.ext,

// Functions in .ksyms are weak, like vmlinux.h's `__weak __ksym` kfuncs.
// CHECK-DAG: .weak bpf_kfunc_call
// CHECK-DAG: .weak bpf_kfunc_addr
// CHECK-DAG: .weak bpf_kfunc_real

unsafe extern "C" {
    #[unsafe(link_section = ".ksyms")]
    fn bpf_kfunc_call(a: u32, p: *mut c_void) -> u32;
    #[unsafe(link_section = ".ksyms")]
    fn bpf_kfunc_addr(x: i32) -> i32;
    #[unsafe(link_section = ".ksyms")]
    #[link_name = "bpf_kfunc_real"]
    fn bpf_kfunc_alias();
    #[unsafe(link_section = ".ksyms")]
    static bpf_ksym_var: u32;
    static plain_extern_var: u64;
}

#[no_mangle]
pub fn call_kfunc(p: *mut c_void) -> u32 {
    unsafe { bpf_kfunc_call(1, p) }
}

#[no_mangle]
pub fn take_addr() -> unsafe extern "C" fn(i32) -> i32 {
    bpf_kfunc_addr
}

#[no_mangle]
pub fn call_alias() {
    unsafe { bpf_kfunc_alias() }
}

#[no_mangle]
pub fn read_ksym() -> u32 {
    unsafe { bpf_ksym_var }
}

#[no_mangle]
pub fn read_plain() -> u64 {
    unsafe { plain_extern_var }
}
