//@ known-bug: #138738
//@ only-x86_64

#![feature(abi_ptx)]
fn main() {
    let a = unsafe { core::mem::transmute::<usize, extern "ptx-kernel" fn(i32)>(4) }(2);
}
