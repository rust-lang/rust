// only-x86_64
// ignore-tidy-linelength
// gate-test-intrinsics
// gate-test-platform_intrinsics
// gate-test-abi_vectorcall
// gate-test-abi_thiscall
// gate-test-abi_ptx
// gate-test-abi_x86_interrupt
// gate-test-abi_amdgpu_kernel

// Functions
extern "rust-intrinsic" fn f1() {} //~ ERROR intrinsics are subject to change
extern "platform-intrinsic" fn f2() {} //~ ERROR platform intrinsics are experimental
extern "vectorcall" fn f3() {} //~ ERROR vectorcall is experimental and subject to change
extern "rust-call" fn f4() {} //~ ERROR rust-call ABI is subject to change
extern "msp430-interrupt" fn f5() {} //~ ERROR msp430-interrupt ABI is experimental
extern "ptx-kernel" fn f6() {} //~ ERROR PTX ABIs are experimental and subject to change
extern "x86-interrupt" fn f7() {} //~ ERROR x86-interrupt ABI is experimental
extern "thiscall" fn f8() {} //~ ERROR thiscall is experimental and subject to change
extern "amdgpu-kernel" fn f9() {} //~ ERROR amdgpu-kernel ABI is experimental and subject to change

// Methods in trait definition
trait Tr {
    extern "rust-intrinsic" fn m1(); //~ ERROR intrinsics are subject to change
    extern "platform-intrinsic" fn m2(); //~ ERROR platform intrinsics are experimental
    extern "vectorcall" fn m3(); //~ ERROR vectorcall is experimental and subject to change
    extern "rust-call" fn m4(); //~ ERROR rust-call ABI is subject to change
    extern "msp430-interrupt" fn m5(); //~ ERROR msp430-interrupt ABI is experimental
    extern "ptx-kernel" fn m6(); //~ ERROR PTX ABIs are experimental and subject to change
    extern "x86-interrupt" fn m7(); //~ ERROR x86-interrupt ABI is experimental
    extern "thiscall" fn m8(); //~ ERROR thiscall is experimental and subject to change
    extern "amdgpu-kernel" fn m9(); //~ ERROR amdgpu-kernel ABI is experimental and subject to change

    extern "rust-intrinsic" fn dm1() {} //~ ERROR intrinsics are subject to change
    extern "platform-intrinsic" fn dm2() {} //~ ERROR platform intrinsics are experimental
    extern "vectorcall" fn dm3() {} //~ ERROR vectorcall is experimental and subject to change
    extern "rust-call" fn dm4() {} //~ ERROR rust-call ABI is subject to change
    extern "msp430-interrupt" fn dm5() {} //~ ERROR msp430-interrupt ABI is experimental
    extern "ptx-kernel" fn dm6() {} //~ ERROR PTX ABIs are experimental and subject to change
    extern "x86-interrupt" fn dm7() {} //~ ERROR x86-interrupt ABI is experimental
    extern "thiscall" fn dm8() {} //~ ERROR thiscall is experimental and subject to change
    extern "amdgpu-kernel" fn dm9() {} //~ ERROR amdgpu-kernel ABI is experimental and subject to change
}

struct S;

// Methods in trait impl
impl Tr for S {
    extern "rust-intrinsic" fn m1() {} //~ ERROR intrinsics are subject to change
    extern "platform-intrinsic" fn m2() {} //~ ERROR platform intrinsics are experimental
    extern "vectorcall" fn m3() {} //~ ERROR vectorcall is experimental and subject to change
    extern "rust-call" fn m4() {} //~ ERROR rust-call ABI is subject to change
    extern "msp430-interrupt" fn m5() {} //~ ERROR msp430-interrupt ABI is experimental
    extern "ptx-kernel" fn m6() {} //~ ERROR PTX ABIs are experimental and subject to change
    extern "x86-interrupt" fn m7() {} //~ ERROR x86-interrupt ABI is experimental
    extern "thiscall" fn m8() {} //~ ERROR thiscall is experimental and subject to change
    extern "amdgpu-kernel" fn m9() {} //~ ERROR amdgpu-kernel ABI is experimental and subject to change
}

// Methods in inherent impl
impl S {
    extern "rust-intrinsic" fn im1() {} //~ ERROR intrinsics are subject to change
    extern "platform-intrinsic" fn im2() {} //~ ERROR platform intrinsics are experimental
    extern "vectorcall" fn im3() {} //~ ERROR vectorcall is experimental and subject to change
    extern "rust-call" fn im4() {} //~ ERROR rust-call ABI is subject to change
    extern "msp430-interrupt" fn im5() {} //~ ERROR msp430-interrupt ABI is experimental
    extern "ptx-kernel" fn im6() {} //~ ERROR PTX ABIs are experimental and subject to change
    extern "x86-interrupt" fn im7() {} //~ ERROR x86-interrupt ABI is experimental
    extern "thiscall" fn im8() {} //~ ERROR thiscall is experimental and subject to change
    extern "amdgpu-kernel" fn im9() {} //~ ERROR amdgpu-kernel ABI is experimental and subject to change
}

// Function pointer types
type A1 = extern "rust-intrinsic" fn(); //~ ERROR intrinsics are subject to change
type A2 = extern "platform-intrinsic" fn(); //~ ERROR platform intrinsics are experimental
type A3 = extern "vectorcall" fn(); //~ ERROR vectorcall is experimental and subject to change
type A4 = extern "rust-call" fn(); //~ ERROR rust-call ABI is subject to change
type A5 = extern "msp430-interrupt" fn(); //~ ERROR msp430-interrupt ABI is experimental
type A6 = extern "ptx-kernel" fn (); //~ ERROR PTX ABIs are experimental and subject to change
type A7 = extern "x86-interrupt" fn(); //~ ERROR x86-interrupt ABI is experimental
type A8 = extern "thiscall" fn(); //~ ERROR thiscall is experimental and subject to change
type A9 = extern "amdgpu-kernel" fn(); //~ ERROR amdgpu-kernel ABI is experimental and subject to change

// Foreign modules
extern "rust-intrinsic" {} //~ ERROR intrinsics are subject to change
extern "platform-intrinsic" {} //~ ERROR platform intrinsics are experimental
extern "vectorcall" {} //~ ERROR vectorcall is experimental and subject to change
extern "rust-call" {} //~ ERROR rust-call ABI is subject to change
extern "msp430-interrupt" {} //~ ERROR msp430-interrupt ABI is experimental
extern "ptx-kernel" {} //~ ERROR PTX ABIs are experimental and subject to change
extern "x86-interrupt" {} //~ ERROR x86-interrupt ABI is experimental
extern "thiscall" {} //~ ERROR thiscall is experimental and subject to change
extern "amdgpu-kernel" {} //~ ERROR amdgpu-kernel ABI is experimental and subject to change

fn main() {}
