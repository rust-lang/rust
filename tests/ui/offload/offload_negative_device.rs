//@ compile-flags: -Zunstable-options -Zoffload=Test -Clto=fat --emit=llvm-ir -Zdeduplicate-diagnostics=yes
//@ no-prefer-dynamic
//@ needs-offload

#![feature(gpu_offload)]

fn kernel() {}

fn main() {
    core::offload::offload! { kernel = kernel, args = (), device = -1 }
    //~^ ERROR evaluation panicked: offload device must be non-negative; omit `device` to use the default device
}
