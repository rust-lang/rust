#![feature(abi_gpu_kernel, gpu_offload, offload)]
#![no_std]

use core::offload::offload::*;

#[cfg(target_os = "linux")]
#[unsafe(no_mangle)]
fn main() {
    //println!("Hello, world!");
    let mut x = [1234.0f64; 256];
    let p: PreloadMut<[f64; 256]> = preload_mut(&mut x);
    // The next line does not compile
    //let p2: PreloadMut<[f64; 256]> = preload_mut(&mut x);
    core::hint::black_box(p);
    let y = [1234.0f64; 128];
    let q: Preload<[f64; 128]> = preload(&y);
    let r: Preload<[f64; 128]> = preload(&y);
    core::hint::black_box(&q);
    core::hint::black_box(&r);
    core::hint::black_box(&q);
}

use core::offload::offload_kernel;

//#[offload_kernel]
//fn foo(a: &[f32], b: &[f32], c: *mut f32) {
//    unsafe { *c = a[0] + b[0] };
//}
