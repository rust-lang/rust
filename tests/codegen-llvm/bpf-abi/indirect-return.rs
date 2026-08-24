// Checks that results larger than one register are returned indirectly
//@ add-minicore
//@ revisions: bpfel bpfeb
//@[bpfel] compile-flags: --target=bpfel-unknown-none
//@[bpfeb] compile-flags: --target=bpfeb-unknown-none
//@ needs-llvm-components: bpf
//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]

extern crate minicore;

struct Big {
    a: [u16; 32],
    b: u64,
}

// CHECK-LABEL: define{{.*}} @inner_big_rust(
// CHECK-SAME:   ptr{{[^,]*}},
// CHECK-SAME:   i64{{[^)]*}}
#[unsafe(no_mangle)]
fn inner_big_rust(a: u64) -> Big {
    Big { a: [a as u16; 32], b: 42 }
}

// CHECK-LABEL: define{{.*}} @inner_big_c(
// CHECK-SAME:   ptr{{[^,]*}},
// CHECK-SAME:   i64{{[^)]*}}
#[unsafe(no_mangle)]
extern "C" fn inner_big_c(a: u64) -> Big {
    Big { a: [a as u16; 32], b: 42 }
}
