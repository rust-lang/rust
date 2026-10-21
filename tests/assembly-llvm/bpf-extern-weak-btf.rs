//@ add-minicore
//@ assembly-output: emit-asm
//@ compile-flags: --target bpfel-unknown-none -Copt-level=2 -Cdebuginfo=2 -Zverbose-asm
//@ needs-llvm-components: bpf

#![feature(no_core, linkage, lang_items)]
#![allow(internal_features)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

// rustc only declares an `extern_weak` static of type `Option<fn>` as a weak function when
// `Option` is the lang item, as it is in core. Minicore's is not, so this one shadows it.
#[lang = "Option"]
pub enum Option<T> {
    None,
    Some(T),
}
impl<T: Copy> Copy for Option<T> {}

// `if let Some(f)` on the static is the equivalent of C's bpf_ksym_exists(): once the
// static is folded, the address of the kfunc itself is tested against null and the kfunc
// is called directly, only when it exists, as clang does. libbpf resolves a missing weak
// kfunc to 0, and the verifier then prunes the call.

// CHECK-LABEL: call_weak:
// CHECK-NOT: _rust_extern_with_linkage
// CHECK: r[[ADDR:[0-9]]] = bpf_weak_kfunc ll
// CHECK-NOT: callx
// CHECK: if r[[ADDR]] == 0 goto [[SKIP:\.?LBB[0-9_]+]]
// CHECK-NOT: callx
// CHECK: call bpf_weak_kfunc
// CHECK: [[SKIP]]:

// CHECK-LABEL: .section .BTF,

// CHECK-DAG: .ascii "u32" # string offset=[[U32_OFF:[0-9]+]]
// CHECK-DAG: .long [[U32_OFF]] # BTF_KIND_INT(id = [[U32:[0-9]+]])
// CHECK-DAG: .ascii "u64" # string offset=[[U64_OFF:[0-9]+]]
// CHECK-DAG: .long [[U64_OFF]] # BTF_KIND_INT(id = [[U64:[0-9]+]])

// CHECK-DAG: # BTF_KIND_PTR(id = [[VOID_PTR:[0-9]+]]){{[[:space:]]+}}.long 33554432 # 0x2000000
// CHECK-DAG: # BTF_KIND_FUNC_PROTO(id = [[PROTO:[0-9]+]]){{[[:space:]]+}}.long 218103810 # 0xd000002
// CHECK-DAG: .ascii "bpf_weak_kfunc" # string offset=[[KFUNC_OFF:[0-9]+]]
// CHECK-DAG: .long [[KFUNC_OFF]] # BTF_KIND_FUNC(id = [[KFUNC_ID:[0-9]+]]){{[[:space:]]+}}.long 201326594 # 0xc000002

// CHECK-DAG: .ascii "bpf_weak_ksym" # string offset=[[KSYM_OFF:[0-9]+]]
// CHECK-DAG: .long [[KSYM_OFF]] # BTF_KIND_VAR(id = [[KSYM_ID:[0-9]+]]){{[[:space:]]+}}.long 234881024 # 0xe000000

// A pointer to `c_void` gives a typeless ksym, like C's `extern const void sym __ksym`.
// CHECK-DAG: .ascii "bpf_weak_typeless" # string offset=[[TYPELESS_OFF:[0-9]+]]
// CHECK-DAG: .long [[TYPELESS_OFF]] # BTF_KIND_VAR(id = [[TYPELESS_ID:[0-9]+]]){{[[:space:]]+}}.long 234881024 # 0xe000000

// CHECK-DAG: .ascii ".ksyms" # string offset=[[KSYMS_OFF:[0-9]+]]
// CHECK-DAG: .long [[KSYMS_OFF]] # BTF_KIND_DATASEC(id = {{[0-9]+}}){{[[:space:]]+}}.long 251658243 # 0xf000003
// CHECK-DAG: .long [[KFUNC_ID]]{{[[:space:]]+}}.long bpf_weak_kfunc{{$}}
// CHECK-DAG: .long [[KSYM_ID]]{{[[:space:]]+}}.long bpf_weak_ksym{{$}}
// CHECK-DAG: .long [[TYPELESS_ID]]{{[[:space:]]+}}.long bpf_weak_typeless{{$}}

// CHECK: .section .BTF.ext,

// CHECK-DAG: .weak bpf_weak_kfunc
// CHECK-DAG: .weak bpf_weak_ksym
// CHECK-DAG: .weak bpf_weak_typeless

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
