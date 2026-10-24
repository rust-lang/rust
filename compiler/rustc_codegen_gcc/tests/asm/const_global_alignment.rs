//@ assembly-output: emit-asm
//@ only-x86_64-unknown-linux-gnu

#![crate_type = "lib"]

// Constants with identical bytes share one global, which needs the largest alignment among
// their uses: lowering it to a later, smaller one or reading it as bits fails these checks.

// CHECK: "A_WIDE":
// CHECK-NEXT: .quad "[[WIDE_THEN_BYTES:global\.[0-9]+]]"
#[no_mangle]
pub static A_WIDE: &[u128; 4] = &[0x09090909_09090909_09090909_09090909; 4];

// CHECK: "B_BYTES":
// CHECK-NEXT: .quad "[[WIDE_THEN_BYTES]]"
#[no_mangle]
pub static B_BYTES: &[u8; 64] = &[9; 64];

// CHECK: "C_BYTES":
// CHECK-NEXT: .quad "[[BYTES_THEN_WIDE:global\.[0-9]+]]"
#[no_mangle]
pub static C_BYTES: &[u8; 64] = &[8; 64];

// CHECK: "D_WIDE":
// CHECK-NEXT: .quad "[[BYTES_THEN_WIDE]]"
#[no_mangle]
pub static D_WIDE: &[u128; 4] = &[0x08080808_08080808_08080808_08080808; 4];

// CHECK: "E_BYTES":
// CHECK-NEXT: .quad "[[BYTES_THEN_HALVES:global\.[0-9]+]]"
#[no_mangle]
pub static E_BYTES: &[u8; 4] = &[3; 4];

// CHECK: "F_HALVES":
// CHECK-NEXT: .quad "[[BYTES_THEN_HALVES]]"
#[no_mangle]
pub static F_HALVES: &[u16; 2] = &[0x0303; 2];

// CHECK: .section .rodata.[[WIDE_THEN_BYTES]],"a"
// CHECK-NEXT: .align {{16|32|64}}{{$}}
// CHECK: .section .rodata.[[BYTES_THEN_WIDE]],"a"
// CHECK-NEXT: .align 16{{$}}
// CHECK: .section .rodata.[[BYTES_THEN_HALVES]],"a"
// CHECK-NEXT: .align 2{{$}}
