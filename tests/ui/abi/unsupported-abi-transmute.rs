#![feature(abi_gpu_kernel)]
// Check we error before unsupported ABIs reach codegen stages.

fn main() {
    let a = unsafe { core::mem::transmute::<usize, extern "gpu-kernel" fn(i32)>(4) }(2);
    //~^ ERROR E0570
}
