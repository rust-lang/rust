#![feature(no_core, btf_tags)]
#![crate_type = "rlib"]
#![no_core]
#![allow(unused_variables)]

extern crate minicore;

#[btf_decl_tag("s_tag")]
#[repr(C)]
pub struct S {
    #[btf_decl_tag("f_tag")]
    pub a: u32,
    #[btf_type_tag("user")]
    #[btf_type_tag("rcu")]
    pub p: *const u64,
}

#[btf_decl_tag("u_tag")]
#[repr(C)]
pub union U {
    pub x: u32,
    #[btf_decl_tag("uf_tag")]
    pub y: u64,
}

#[btf_decl_tag("g_tag")]
#[btf_type_tag("g_ptr")]
#[no_mangle]
pub static mut G: *mut u32 = 0 as *mut u32;

#[btf_decl_tag("fn_tag")]
#[no_mangle]
pub fn func(
    #[btf_decl_tag("arg_tag")]
    #[btf_type_tag("user")]
    buf: *mut u8,
    s: &S,
    u: U,
) -> u32 {
    unsafe { *buf as u32 }
}

// CHECK-BTF:      [1] TYPE_TAG 'user' type_id=3
// CHECK-BTF-NEXT: [2] PTR '(anon)' type_id=1
// CHECK-BTF-NEXT: [3] INT 'u8' size=1 bits_offset=0 nr_bits=8 encoding=(none)
// CHECK-BTF-NEXT: [4] PTR '(anon)' type_id=5
// CHECK-BTF-NEXT: [5] STRUCT 'S' size=16 vlen=2
// CHECK-BTF-NEXT:         'a' type_id=7 bits_offset=0
// CHECK-BTF-NEXT:         'p' type_id=11 bits_offset=64
// CHECK-BTF-NEXT: [6] DECL_TAG 's_tag' type_id=5 component_idx=-1
// CHECK-BTF-NEXT: [7] INT 'u32' size=4 bits_offset=0 nr_bits=32 encoding=(none)
// CHECK-BTF-NEXT: [8] DECL_TAG 'f_tag' type_id=5 component_idx=0
// CHECK-BTF-NEXT: [9] TYPE_TAG 'user' type_id=12
// CHECK-BTF-NEXT: [10] TYPE_TAG 'rcu' type_id=9
// CHECK-BTF-NEXT: [11] PTR '(anon)' type_id=10
// CHECK-BTF-NEXT: [12] INT 'u64' size=8 bits_offset=0 nr_bits=64 encoding=(none)
// CHECK-BTF-NEXT: [13] UNION 'U' size=8 vlen=2
// CHECK-BTF-NEXT:         'x' type_id=7 bits_offset=0
// CHECK-BTF-NEXT:         'y' type_id=12 bits_offset=0
// CHECK-BTF-NEXT: [14] DECL_TAG 'u_tag' type_id=13 component_idx=-1
// CHECK-BTF-NEXT: [15] DECL_TAG 'uf_tag' type_id=13 component_idx=1
// CHECK-BTF-NEXT: [16] FUNC_PROTO '(anon)' ret_type_id=7 vlen=3
// CHECK-BTF-NEXT:         'buf' type_id=2
// CHECK-BTF-NEXT:         's' type_id=4
// CHECK-BTF-NEXT:         'u' type_id=13
// CHECK-BTF-NEXT: [17] FUNC 'func' type_id=16 linkage=global
// CHECK-BTF-NEXT: [18] DECL_TAG 'arg_tag' type_id=17 component_idx=0
// CHECK-BTF-NEXT: [19] DECL_TAG 'fn_tag' type_id=17 component_idx=-1
// CHECK-BTF-NEXT: [20] TYPE_TAG 'g_ptr' type_id=7
// CHECK-BTF-NEXT: [21] PTR '(anon)' type_id=20
// CHECK-BTF-NEXT: [22] VAR 'G' type_id=21, linkage=global
// CHECK-BTF-NEXT: [23] DECL_TAG 'g_tag' type_id=22 component_idx=-1
// CHECK-BTF-NEXT: [24] DATASEC '.bss' size=0 vlen=1
// CHECK-BTF-NEXT:         type_id=22 offset=0 size=8
