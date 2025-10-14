// Checks that results larger than one register are returned indirectly
//@ only-bpf
//@ needs-llvm-components: bpf
//@ compile-flags: --target bpfel-unknown-none

#![no_std]
#![no_main]

#[no_mangle]
fn outer(a: u64) -> u64 {
    let v = match inner_res(a) {
        Ok(v) => v,
        Err(()) => 0,
    };

    inner_big(v).a[0] as u64
}

// CHECK-LABEL: define {{.*}} @_ZN{{.*}}inner_res{{.*}}E(
// CHECK-SAME:   ptr{{[^,]*}},
// CHECK-SAME:   i64{{[^)]*}}
#[inline(never)]
fn inner_res(a: u64) -> Result<u64, ()> {
    if a == 0 { Err(()) } else { Ok(a + 1) }
}

struct Big {
    a: [u16; 32],
    b: u64,
}

// CHECK-LABEL: define {{.*}} @_ZN{{.*}}inner_big{{.*}}E(
// CHECK-SAME:   ptr{{[^,]*}},
// CHECK-SAME:   i64{{[^)]*}}
#[inline(never)]
fn inner_big(a: u64) -> Big {
    Big { a: [a as u16; 32], b: 42 }
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
