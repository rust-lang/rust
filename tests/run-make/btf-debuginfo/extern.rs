#![feature(no_core)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK-BTF:      [1] FUNC_PROTO '(anon)' ret_type_id=0 vlen=0
// CHECK-BTF-NEXT: [2] FUNC 'call_alias' type_id=1 linkage=global
// CHECK-BTF-NEXT: [3] FUNC_PROTO '(anon)' ret_type_id=0 vlen=0
// CHECK-BTF-NEXT: [4] FUNC 'bpf_kfunc_real' type_id=3 linkage=extern
// `*mut c_void` must be a pointer to void (type 0), not to the `c_void` enum.
// CHECK-BTF-NEXT: [5] PTR '(anon)' type_id=0
// CHECK-BTF-NEXT: [6] FUNC_PROTO '(anon)' ret_type_id=7 vlen=1
// CHECK-BTF-NEXT:         'p' type_id=5
// CHECK-BTF-NEXT: [7] INT 'u32' size=4 bits_offset=0 nr_bits=32 encoding=(none)
// CHECK-BTF-NEXT: [8] FUNC 'call_kfunc' type_id=6 linkage=global
// CHECK-BTF-NEXT: [9] FUNC_PROTO '(anon)' ret_type_id=7 vlen=2
// CHECK-BTF-NEXT:         '(anon)' type_id=7
// CHECK-BTF-NEXT:         '(anon)' type_id=5
// CHECK-BTF-NEXT: [10] FUNC 'bpf_kfunc_call' type_id=9 linkage=extern
// CHECK-BTF-NEXT: [11] FUNC_PROTO '(anon)' ret_type_id=7 vlen=0
// CHECK-BTF-NEXT: [12] FUNC 'read_ksym' type_id=11 linkage=global
// CHECK-BTF-NEXT: [13] FUNC_PROTO '(anon)' ret_type_id=14 vlen=0
// CHECK-BTF-NEXT: [14] INT 'u64' size=8 bits_offset=0 nr_bits=64 encoding=(none)
// CHECK-BTF-NEXT: [15] FUNC 'read_plain' type_id=13 linkage=global
// CHECK-BTF-NEXT: [16] FUNC_PROTO '(anon)' ret_type_id=17 vlen=0
// CHECK-BTF-NEXT: [17] PTR '(anon)' type_id=18
// CHECK-BTF-NEXT: [18] FUNC_PROTO '(anon)' ret_type_id=19 vlen=1
// CHECK-BTF-NEXT:         '(anon)' type_id=19
// CHECK-BTF-NEXT: [19] INT 'i32' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
// CHECK-BTF-NEXT: [20] FUNC 'take_addr' type_id=16 linkage=global
// CHECK-BTF-NEXT: [21] FUNC_PROTO '(anon)' ret_type_id=19 vlen=1
// CHECK-BTF-NEXT:         '(anon)' type_id=19
// CHECK-BTF-NEXT: [22] FUNC 'bpf_kfunc_addr' type_id=21 linkage=extern
// CHECK-BTF-NEXT: [23] VAR 'bpf_ksym_var' type_id=7, linkage=extern
// A foreign static without a section still gets a BTF_KIND_VAR, but no DATASEC entry.
// CHECK-BTF-NEXT: [24] VAR 'plain_extern_var' type_id=14, linkage=extern
// Exactly the three functions and the one variable placed in .ksyms.
// CHECK-BTF-NEXT: [25] DATASEC '.ksyms' size=0 vlen=4
// CHECK-BTF-NEXT:         type_id=4 offset=0 size=0
// CHECK-BTF-NEXT:         type_id=10 offset=0 size=0
// CHECK-BTF-NEXT:         type_id=22 offset=0 size=0
// CHECK-BTF-NEXT:         type_id=23 offset=0 size=4
// CHECK-BTF-NOT:  {{.}}

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
