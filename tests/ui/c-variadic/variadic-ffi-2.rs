#![feature(extended_varargs_abi_support)]

fn baz(f: extern "Rust" fn(usize, ...)) {
    //~^ ERROR: C-variadic functions with the "Rust" calling convention are not supported
    f(22, 44);
}

#[cfg(target_arch = "x86_64")]
fn sysv(f: extern "sysv64" fn(usize, ...)) {
    f(22, 44);
}
#[cfg(target_arch = "x86_64")]
fn win(f: extern "win64" fn(usize, ...)) {
    f(22, 44);
}
#[cfg(any(
    target_arch = "arm",
    target_arch = "aarch64",
    target_arch = "riscv32",
    target_arch = "riscv64",
    target_arch = "x86",
    target_arch = "x86_64"
))]
fn efiapi(f: extern "efiapi" fn(usize, ...)) {
    f(22, 44);
}

fn main() {}
