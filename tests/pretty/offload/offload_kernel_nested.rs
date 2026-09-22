//@ check-pass
#![feature(gpu_offload)]

fn kernel() {
    #[core::offload::offload_kernel]
    fn inner_kernel() {}
}

fn main() {}
