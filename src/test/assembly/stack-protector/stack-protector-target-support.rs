// Test that stack smash protection code is emitted for all tier1 and tier2
// targets, with the exception of nvptx64-nvidia-cuda
//
// revisions: r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23
// revisions: r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 r37 r38 r39 r40 r41 r42 r43 r44
// revisions: r45 r46 r47 r48 r49 r50 r51 r52 r53 r54 r55 r56 r57 r58 r59 r60 r61 r62 r63 r64 r65
// revisions: r66 r67 r68 r69 r70 r71 r72 r73 r74 r75 r76 r77 r78 r79 r80 r81 r82 r83 r84
// assembly-output: emit-asm
// [r1] compile-flags: --target aarch64-unknown-linux-gnu
// [r1] needs-llvm-components: aarch64
// [r2] compile-flags: --target i686-pc-windows-gnu
// [r2] needs-llvm-components: x86
// [r3] compile-flags: --target i686-pc-windows-msvc
// [r3] needs-llvm-components: x86
// [r4] compile-flags: --target i686-unknown-linux-gnu
// [r4] needs-llvm-components: x86
// [r5] compile-flags: --target x86_64-apple-darwin
// [r5] needs-llvm-components: x86
// [r6] compile-flags: --target x86_64-pc-windows-gnu
// [r6] needs-llvm-components: x86
// [r7] compile-flags: --target x86_64-pc-windows-msvc
// [r7] needs-llvm-components: x86
// [r8] compile-flags: --target x86_64-unknown-linux-gnu
// [r8] needs-llvm-components: x86
// [r9] compile-flags: --target aarch64-apple-darwin
// [r9] needs-llvm-components: aarch64
// [r10] compile-flags: --target aarch64-apple-ios
// [r10] needs-llvm-components: aarch64
// [r11] compile-flags: --target aarch64-fuchsia
// [r11] needs-llvm-components: aarch64
// [r12] compile-flags: --target aarch64-linux-android
// [r12] needs-llvm-components: aarch64
// [r13] compile-flags: --target aarch64-pc-windows-msvc
// [r13] needs-llvm-components: aarch64
// [r14] compile-flags: --target aarch64-unknown-linux-musl
// [r14] needs-llvm-components: aarch64
// [r15] compile-flags: --target aarch64-unknown-none
// [r15] needs-llvm-components: aarch64
// [r16] compile-flags: --target aarch64-unknown-none-softfloat
// [r16] needs-llvm-components: aarch64
// [r17] compile-flags: --target arm-linux-androideabi
// [r17] needs-llvm-components: arm
// [r18] compile-flags: --target arm-unknown-linux-gnueabi
// [r18] needs-llvm-components: arm
// [r19] compile-flags: --target arm-unknown-linux-gnueabihf
// [r19] needs-llvm-components: arm
// [r20] compile-flags: --target arm-unknown-linux-musleabi
// [r20] needs-llvm-components: arm
// [r21] compile-flags: --target arm-unknown-linux-musleabihf
// [r21] needs-llvm-components: arm
// [r22] compile-flags: --target armebv7r-none-eabi
// [r22] needs-llvm-components: arm
// [r23] compile-flags: --target armebv7r-none-eabihf
// [r23] needs-llvm-components: arm
// [r24] compile-flags: --target armv5te-unknown-linux-gnueabi
// [r24] needs-llvm-components: arm
// [r25] compile-flags: --target armv5te-unknown-linux-musleabi
// [r25] needs-llvm-components: arm
// [r26] compile-flags: --target armv7-linux-androideabi
// [r26] needs-llvm-components: arm
// [r27] compile-flags: --target armv7a-none-eabi
// [r27] needs-llvm-components: arm
// [r28] compile-flags: --target armv7r-none-eabi
// [r28] needs-llvm-components: arm
// [r29] compile-flags: --target armv7r-none-eabihf
// [r29] needs-llvm-components: arm
// [r30] compile-flags: --target armv7-unknown-linux-gnueabi
// [r30] needs-llvm-components: arm
// [r31] compile-flags: --target armv7-unknown-linux-gnueabihf
// [r31] needs-llvm-components: arm
// [r32] compile-flags: --target armv7-unknown-linux-musleabi
// [r32] needs-llvm-components: arm
// [r33] compile-flags: --target armv7-unknown-linux-musleabihf
// [r33] needs-llvm-components: arm
// [r34] compile-flags: --target asmjs-unknown-emscripten
// [r34] needs-llvm-components: webassembly
// [r35] compile-flags: --target i586-pc-windows-msvc
// [r35] needs-llvm-components: x86
// [r36] compile-flags: --target i586-unknown-linux-gnu
// [r36] needs-llvm-components: x86
// [r37] compile-flags: --target i586-unknown-linux-musl
// [r37] needs-llvm-components: x86
// [r38] compile-flags: --target i686-linux-android
// [r38] needs-llvm-components: x86
// [r39] compile-flags: --target i686-unknown-freebsd
// [r39] needs-llvm-components: x86
// [r40] compile-flags: --target i686-unknown-linux-musl
// [r40] needs-llvm-components: x86
// [r41] compile-flags: --target mips-unknown-linux-gnu
// [r41] needs-llvm-components: mips
// [r42] compile-flags: --target mips-unknown-linux-musl
// [r42] needs-llvm-components: mips
// [r43] compile-flags: --target mips64-unknown-linux-gnuabi64
// [r43] needs-llvm-components: mips
// [r44] compile-flags: --target mips64-unknown-linux-muslabi64
// [r44] needs-llvm-components: mips
// [r45] compile-flags: --target mips64el-unknown-linux-gnuabi64
// [r45] needs-llvm-components: mips
// [r46] compile-flags: --target mips64el-unknown-linux-muslabi64
// [r46] needs-llvm-components: mips
// [r47] compile-flags: --target mipsel-unknown-linux-gnu
// [r47] needs-llvm-components: mips
// [r48] compile-flags: --target mipsel-unknown-linux-musl
// [r48] needs-llvm-components: mips
// [r49] compile-flags: --target nvptx64-nvidia-cuda
// [r49] needs-llvm-components: nvptx
// [r50] compile-flags: --target powerpc-unknown-linux-gnu
// [r50] needs-llvm-components: powerpc
// [r51] compile-flags: --target powerpc64-unknown-linux-gnu
// [r51] needs-llvm-components: powerpc
// [r52] compile-flags: --target powerpc64le-unknown-linux-gnu
// [r52] needs-llvm-components: powerpc
// [r53] compile-flags: --target riscv32i-unknown-none-elf
// [r53] needs-llvm-components: riscv
// [r54] compile-flags: --target riscv32imac-unknown-none-elf
// [r54] needs-llvm-components: riscv
// [r55] compile-flags:--target riscv32imc-unknown-none-elf
// [r55] needs-llvm-components: riscv
// [r56] compile-flags:--target riscv64gc-unknown-linux-gnu
// [r56] needs-llvm-components: riscv
// [r57] compile-flags:--target riscv64gc-unknown-none-elf
// [r57] needs-llvm-components: riscv
// [r58] compile-flags:--target riscv64imac-unknown-none-elf
// [r58] needs-llvm-components: riscv
// [r59] compile-flags:--target s390x-unknown-linux-gnu
// [r59] needs-llvm-components: systemz
// [r60] compile-flags:--target sparc64-unknown-linux-gnu
// [r60] needs-llvm-components: sparc
// [r61] compile-flags:--target sparcv9-sun-solaris
// [r61] needs-llvm-components: sparc
// [r62] compile-flags:--target thumbv6m-none-eabi
// [r62] needs-llvm-components: arm
// [r63] compile-flags:--target thumbv7em-none-eabi
// [r63] needs-llvm-components: arm
// [r64] compile-flags:--target thumbv7em-none-eabihf
// [r64] needs-llvm-components: arm
// [r65] compile-flags:--target thumbv7m-none-eabi
// [r65] needs-llvm-components: arm
// [r66] compile-flags:--target thumbv7neon-linux-androideabi
// [r66] needs-llvm-components: arm
// [r67] compile-flags:--target thumbv7neon-unknown-linux-gnueabihf
// [r67] needs-llvm-components: arm
// [r68] compile-flags:--target thumbv8m.base-none-eabi
// [r68] needs-llvm-components: arm
// [r69] compile-flags:--target thumbv8m.main-none-eabi
// [r69] needs-llvm-components: arm
// [r70] compile-flags:--target thumbv8m.main-none-eabihf
// [r70] needs-llvm-components: arm
// [r71] compile-flags:--target wasm32-unknown-emscripten
// [r71] needs-llvm-components: webassembly
// [r72] compile-flags:--target wasm32-unknown-unknown
// [r72] needs-llvm-components: webassembly
// [r73] compile-flags:--target wasm32-wasi
// [r73] needs-llvm-components: webassembly
// [r74] compile-flags:--target x86_64-apple-ios
// [r74] needs-llvm-components: x86
// [r75] compile-flags:--target x86_64-fortanix-unknown-sgx
// [r75] needs-llvm-components: x86
// [r75] min-llvm-version: 11.0.0
// [r76] compile-flags:--target x86_64-fuchsia
// [r76] needs-llvm-components: x86
// [r77] compile-flags:--target x86_64-linux-android
// [r77] needs-llvm-components: x86
// [r78] compile-flags:--target x86_64-sun-solaris
// [r78] needs-llvm-components: x86
// [r79] compile-flags:--target x86_64-unknown-freebsd
// [r79] needs-llvm-components: x86
// [r80] compile-flags:--target x86_64-unknown-illumos
// [r80] needs-llvm-components: x86
// [r81] compile-flags:--target x86_64-unknown-linux-gnux32
// [r81] needs-llvm-components: x86
// [r82] compile-flags:--target x86_64-unknown-linux-musl
// [r82] needs-llvm-components: x86
// [r83] compile-flags:--target x86_64-unknown-netbsd
// [r83] needs-llvm-components: x86
// [r84] compile-flags: --target x86_64-unknown-redox
// [r84] needs-llvm-components: x86
// compile-flags: -Z stack-protector=all
// compile-flags: -C opt-level=2

#![crate_type = "lib"]

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

#[no_mangle]
pub fn foo() {
    // CHECK: foo{{:|()}}

    // MSVC does the stack checking within a stack-check function:
    // r3: calll @__security_check_cookie
    // r7: callq __security_check_cookie
    // r13: bl __security_check_cookie
    // r35: calll @__security_check_cookie

    // cuda doesn't support stack-smash protection
    // r49-NOT: __security_check_cookie
    // r49-NOT: __stack_chk_fail

    // Other targets do stack checking within the function, and call a failure function on error
    // r1: __stack_chk_fail
    // r2: __stack_chk_fail
    // r4: __stack_chk_fail
    // r5: __stack_chk_fail
    // r6: __stack_chk_fail
    // r8: __stack_chk_fail
    // r9: __stack_chk_fail
    // r10: __stack_chk_fail
    // r11: __stack_chk_fail
    // r12: __stack_chk_fail
    // r14: __stack_chk_fail
    // r15: __stack_chk_fail
    // r16: __stack_chk_fail
    // r17: __stack_chk_fail
    // r18: __stack_chk_fail
    // r19: __stack_chk_fail
    // r20: __stack_chk_fail
    // r21: __stack_chk_fail
    // r22: __stack_chk_fail
    // r23: __stack_chk_fail
    // r24: __stack_chk_fail
    // r25: __stack_chk_fail
    // r26: __stack_chk_fail
    // r27: __stack_chk_fail
    // r28: __stack_chk_fail
    // r29: __stack_chk_fail
    // r30: __stack_chk_fail
    // r31: __stack_chk_fail
    // r32: __stack_chk_fail
    // r33: __stack_chk_fail
    // r34: __stack_chk_fail
    // r36: __stack_chk_fail
    // r37: __stack_chk_fail
    // r38: __stack_chk_fail
    // r39: __stack_chk_fail
    // r40: __stack_chk_fail
    // r41: __stack_chk_fail
    // r42: __stack_chk_fail
    // r43: __stack_chk_fail
    // r44: __stack_chk_fail
    // r45: __stack_chk_fail
    // r46: __stack_chk_fail
    // r47: __stack_chk_fail
    // r48: __stack_chk_fail
    // r50: __stack_chk_fail
    // r51: __stack_chk_fail
    // r52: __stack_chk_fail
    // r53: __stack_chk_fail
    // r54: __stack_chk_fail
    // r55: __stack_chk_fail
    // r56: __stack_chk_fail
    // r57: __stack_chk_fail
    // r58: __stack_chk_fail
    // r59: __stack_chk_fail
    // r60: __stack_chk_fail
    // r61: __stack_chk_fail
    // r62: __stack_chk_fail
    // r63: __stack_chk_fail
    // r64: __stack_chk_fail
    // r65: __stack_chk_fail
    // r66: __stack_chk_fail
    // r67: __stack_chk_fail
    // r68: __stack_chk_fail
    // r69: __stack_chk_fail
    // r70: __stack_chk_fail
    // r71: __stack_chk_fail
    // r72: __stack_chk_fail
    // r73: __stack_chk_fail
    // r74: __stack_chk_fail
    // r75: __stack_chk_fail
    // r76: __stack_chk_fail
    // r77: __stack_chk_fail
    // r78: __stack_chk_fail
    // r79: __stack_chk_fail
    // r80: __stack_chk_fail
    // r81: __stack_chk_fail
    // r82: __stack_chk_fail
    // r83: __stack_chk_fail
    // r84: __stack_chk_fail
}
