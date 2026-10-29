#![feature(no_core, linkage, lang_items)]
#![allow(internal_features)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

// rustc only declares an `extern_weak` static of type `Option<fn>` as a weak function when
// `Option` is the lang item, as it is in core. Minicore's `Option` is not, so define one
// here to shadow it.
#[lang = "Option"]
pub enum Option<T> {
    None,
    Some(T),
}
impl<T: Copy> Copy for Option<T> {}

// Matching on the static is equivalent to C's `bpf_ksym_exists()`: after the static is
// folded, the kfunc's address is compared against null, and the kfunc is called directly
// only when it exists, as in clang. libbpf resolves a missing weak kfunc to 0, and the
// verifier then prunes the call.

// CHECK-OBJ-LABEL: <call_weak>:
// CHECK-OBJ-NOT: _rust_extern_with_linkage
// CHECK-OBJ-NOT: callx
// CHECK-OBJ: r[[ADDR:[0-9]]] = 0x0 ll
// CHECK-OBJ-NEXT: R_BPF_64_64 bpf_weak_kfunc
// CHECK-OBJ-NOT: callx
// CHECK-OBJ: if r[[ADDR]] == 0x0 goto {{.*}} <{{(L[0-9]+|call_weak\+0x[0-9a-f]+)}}>
// CHECK-OBJ-NOT: callx
// CHECK-OBJ: call -0x1
// CHECK-OBJ-NEXT: R_BPF_64_32 bpf_weak_kfunc
// CHECK-OBJ: exit

// CHECK-SYM-DAG: w bpf_weak_kfunc
// CHECK-SYM-DAG: w bpf_weak_ksym
// CHECK-SYM-DAG: w bpf_weak_typeless

// CHECK-BTF:      [1] PTR '(anon)' type_id=0
// CHECK-BTF-NEXT: [2] FUNC_PROTO '(anon)' ret_type_id=3 vlen=1
// CHECK-BTF-NEXT:         'p' type_id=1
// CHECK-BTF-NEXT: [3] INT 'u64' size=8 bits_offset=0 nr_bits=64 encoding=(none)
// CHECK-BTF-NEXT: [4] FUNC 'call_weak' type_id=2 linkage=global
// CHECK-BTF-NEXT: [5] FUNC_PROTO '(anon)' ret_type_id=3 vlen=2
// CHECK-BTF-NEXT:         '(anon)' type_id=6
// CHECK-BTF-NEXT:         '(anon)' type_id=1
// CHECK-BTF-NEXT: [6] INT 'u32' size=4 bits_offset=0 nr_bits=32 encoding=(none)
// CHECK-BTF-NEXT: [7] FUNC 'bpf_weak_kfunc' type_id=5 linkage=extern
// CHECK-BTF-NEXT: [8] FUNC_PROTO '(anon)' ret_type_id=9 vlen=0
// CHECK-BTF-NEXT: [9] PTR '(anon)' type_id=6
// CHECK-BTF-NEXT: [10] FUNC 'weak_ksym' type_id=8 linkage=global
// CHECK-BTF-NEXT: [11] FUNC_PROTO '(anon)' ret_type_id=12 vlen=0
// CHECK-BTF-NEXT: [12] PTR '(anon)' type_id=0
// CHECK-BTF-NEXT: [13] FUNC 'weak_typeless' type_id=11 linkage=global
// CHECK-BTF-NEXT: [14] VAR 'bpf_weak_ksym' type_id=6, linkage=extern
// CHECK-BTF-NEXT: [15] VAR 'bpf_weak_typeless' type_id=0, linkage=extern
// CHECK-BTF-NEXT: [16] DATASEC '.ksyms' size=0 vlen=3
// CHECK-BTF-NEXT:         type_id=7 offset=0 size=0
// CHECK-BTF-NEXT:         type_id=14 offset=0 size=1
// CHECK-BTF-NEXT:         type_id=15 offset=0 size=1
// CHECK-BTF-NOT:  {{.}}

unsafe extern "C" {
    #[unsafe(link_section = ".ksyms")]
    #[linkage = "extern_weak"]
    static bpf_weak_kfunc: Option<unsafe extern "C" fn(u32, *mut c_void) -> u64>;
    #[unsafe(link_section = ".ksyms")]
    #[linkage = "extern_weak"]
    static bpf_weak_ksym: *const u32;
    #[unsafe(link_section = ".ksyms")]
    #[linkage = "extern_weak"]
    static bpf_weak_typeless: *const c_void;
}

#[no_mangle]
pub fn call_weak(p: *mut c_void) -> u64 {
    unsafe {
        match bpf_weak_kfunc {
            Option::Some(f) => f(1, p),
            Option::None => 0,
        }
    }
}

#[no_mangle]
pub fn weak_ksym() -> *const u32 {
    unsafe { bpf_weak_ksym }
}

#[no_mangle]
pub fn weak_typeless() -> *const c_void {
    unsafe { bpf_weak_typeless }
}
