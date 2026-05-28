// Checks that results larger than one register are returned indirectly
//@ add-minicore
//@ needs-llvm-components: bpf
//@ compile-flags: --target bpfel-unknown-none

#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]

extern crate minicore;

#[no_mangle]
fn outer(a: u64) -> u64 {
    inner_big(a).b
}

struct Big {
    a: [u16; 32],
    b: u64,
}

// CHECK-LABEL: define {{.*}} @_R{{.*}}inner_big(
// CHECK-SAME:   ptr{{[^,]*}},
// CHECK-SAME:   i64{{[^)]*}}
#[inline(never)]
fn inner_big(a: u64) -> Big {
    Big { a: [a as u16; 32], b: 42 }
}
