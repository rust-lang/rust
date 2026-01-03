# Usage

This feature is work-in-progress, and not ready for usage. The instructions here are for contributors, or people interested in following the latest progress.
We currently work on launching the following Rust kernel on the GPU. To follow along, copy it to a `src/lib.rs` file.

```rust
#![feature(abi_gpu_kernel)]
#![feature(rustc_attrs)]
#![feature(core_intrinsics)]
#![no_std]

#[cfg(target_os = "linux")]
extern crate libc;
#[cfg(target_os = "linux")]
use libc::c_char;

#[cfg(target_os = "linux")]
use core::mem;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[cfg(target_os = "linux")]
#[unsafe(no_mangle)]
#[inline(never)]
fn main() {
    let array_c: *mut [f64; 256] =
        unsafe { libc::calloc(256, (mem::size_of::<f64>()) as libc::size_t) as *mut [f64; 256] };
    let output = c"The first element is zero %f\n";
    let output2 = c"The first element is NOT zero %f\n";
    let output3 = c"The second element is %f\n";
    unsafe {
        let val: *const c_char = if (*array_c)[0] < 0.1 {
            output.as_ptr()
        } else {
            output2.as_ptr()
        };
        libc::printf(val, (*array_c)[0]);
    }

    unsafe {
        kernel(array_c);
    }
    core::hint::black_box(&array_c);
    unsafe {
        let val: *const c_char = if (*array_c)[0] < 0.1 {
            output.as_ptr()
        } else {
            output2.as_ptr()
        };
        libc::printf(val, (*array_c)[0]);
        libc::printf(output3.as_ptr(), (*array_c)[1]);
    }
}

#[inline(never)]
unsafe fn kernel(x: *mut [f64; 256]) {
    core::intrinsics::offload(kernel_1, (x,))
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn kernel_1(array_b: *mut [f64; 256]);
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub extern "gpu-kernel" fn kernel_1(x: *mut [f64; 256]) {
    unsafe { (*x)[0] = 21.0 };
}
```

## Compile instructions
It is important to use a clang compiler build on the same llvm as rustc. Just calling clang without the full path will likely use your system clang, which probably will be incompatible. So either substitute clang/lld invocations below with absolute path, or set your `PATH` accordingly.

First we generate the device (gpu) code. Replace the target-cpu with the right code for your gpu.
```
RUSTFLAGS="-Ctarget-cpu=gfx90a --emit=llvm-bc,llvm-ir -Zoffload=Device -Csave-temps -Zunstable-options" cargo +offload build -Zunstable-options -r -v --target amdgcn-amd-amdhsa -Zbuild-std=core
```
You might afterwards need to copy your target/release/deps/<lib_name>.bc to lib.bc for now, before the next step.

Now we generate the host (cpu) code.
```
RUSTFLAGS="--emit=llvm-bc,llvm-ir -Csave-temps -Zoffload=Host=/p/lustre1/drehwald1/prog/offload/r/target/amdgcn-amd-amdhsa/release/deps/host.out -Zunstable-options" cargo +offload build -r
```
This call also does a lot of work and generates multiple intermediate files for llvm offload.
While we integrated most offload steps into rustc by now, one binary invocation still remains for now:

```
"clang-linker-wrapper" "--should-extract=gfx90a" "--device-compiler=amdgcn-amd-amdhsa=-g" "--device-compiler=amdgcn-amd-amdhsa=-save-temps=cwd" "--device-linker=amdgcn-amd-amdhsa=-lompdevice" "--host-triple=x86_64-unknown-linux-gnu" "--save-temps" "--linker-path=/ABSOlUTE_PATH_TO/rust/build/x86_64-unknown-linux-gnu/lld/bin/ld.lld" "--hash-style=gnu" "--eh-frame-hdr" "-m" "elf_x86_64" "-pie" "-dynamic-linker" "/lib64/ld-linux-x86-64.so.2" "-o" "bare" "/lib/../lib64/Scrt1.o" "/lib/../lib64/crti.o" "/ABSOLUTE_PATH_TO/crtbeginS.o" "-L/ABSOLUTE_PATH_TO/rust/build/x86_64-unknown-linux-gnu/llvm/bin/../lib/x86_64-unknown-linux-gnu" "-L/ABSOLUTE_PATH_TO/rust/build/x86_64-unknown-linux-gnu/llvm/lib/clang/21/lib/x86_64-unknown-linux-gnu" "-L/lib/../lib64" "-L/usr/lib64" "-L/lib" "-L/usr/lib" "target/<GPU_DIR>/release/host.o" "-lstdc++" "-lm" "-lomp" "-lomptarget" "-L/ABSOLUTE_PATH_TO/rust/build/x86_64-unknown-linux-gnu/llvm/lib" "-lgcc_s" "-lgcc" "-lpthread" "-lc" "-lgcc_s" "-lgcc" "/ABSOLUTE_PATH_TO/crtendS.o" "/lib/../lib64/crtn.o"
```

You can try to find the paths to those files on your system. However, I recommend to not fix the paths, but rather just re-generate them by copying a bare-mode openmp example and compiling it with your clang. By adding `-###` to your clang invocation, you can see the invidual steps.
It will show multiple steps, just look for the clang-linker-wrapper example. Make sure to still include the path to the `host.o` file, and not whatever tmp file you got when compiling your c++ example with the following call.
```
myclang++ -fuse-ld=lld -O3 -fopenmp  -fopenmp-offload-mandatory --offload-arch=gfx90a omp_bare.cpp -o main -###
```

In the final step, you can now run your binary

```
./main
The first element is zero 0.000000
The first element is NOT zero 21.000000
The second element is  0.000000
```

To receive more information about the memory transfer, you can enable info printing with
```
LIBOMPTARGET_INFO=-1  ./main
```
