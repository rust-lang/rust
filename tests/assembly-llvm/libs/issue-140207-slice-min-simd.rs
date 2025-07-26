//@ assembly-output: emit-asm
// # avx has a dedicated instruction for this
//@ compile-flags: --crate-type=lib -Ctarget-cpu=znver2 -Copt-level=3
//@ only-x86_64
//@ ignore-sgx
// https://github.com/rust-lang/rust/issues/140207

#[unsafe(no_mangle)]
pub fn array_min(a: &[u16; 8]) -> u16 {
    // CHECK: vphminposuw
    // CHECK: ret
    a.iter().copied().min().unwrap()
}
