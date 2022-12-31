// ignore-x86
// ignore-aarch64
// ignore-aarch64_be
// ignore-arm
// ignore-armeb
// ignore-avr
// ignore-bpfel
// ignore-bpfeb
// ignore-hexagon
// ignore-mips
// ignore-mips64
// ignore-msp430
// ignore-powerpc64
// ignore-powerpc64le
// ignore-powerpc
// ignore-r600
// ignore-amdgcn
// ignore-sparc
// ignore-sparcv9
// ignore-sparcel
// ignore-s390x
// ignore-tce
// ignore-thumb
// ignore-thumbeb
// ignore-xcore
// ignore-nvptx
// ignore-nvptx64
// ignore-le32
// ignore-le64
// ignore-amdil
// ignore-amdil64
// ignore-hsail
// ignore-hsail64
// ignore-spir
// ignore-spir64
// ignore-kalimba
// ignore-shave
//
// Tests that `byval` alignment is properly specified (#80127).
// The only targets that use `byval` are m68k, wasm, x86-64, and x86. Note that
// x86 has special rules (see #103830), and it's therefore ignored here.

#[repr(C)]
#[repr(align(16))]
struct Foo {
    a: [i32; 16],
}

extern "C" {
    // CHECK: declare void @f({{.*}}byval(%Foo) align 16{{.*}})
    fn f(foo: Foo);
}

pub fn main() {
    unsafe { f(Foo { a: [1; 16] }) }
}
