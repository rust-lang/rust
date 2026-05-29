// Even though this test only checks 2 of the 10 or so unstable linker flavors, it exercizes the
// unique codepath checking all unstable options (see `LinkerFlavorCli::is_unstable` and its
// caller). If it passes, all the other unstable options are rejected as well.
//
//@ revisions: bpf llbc
//@ [bpf] compile-flags: --target=bpfel-unknown-none -C linker-flavor=bpf --crate-type=rlib
//@ [bpf] needs-llvm-components: bpf
//@ [llbc] compile-flags: --target=nvptx64-nvidia-cuda -C linker-flavor=llbc --crate-type=rlib
//@ [llbc] needs-llvm-components: nvptx

#![feature(no_core)]
#![no_core]

//[bpf]~? ERROR the linker flavor `bpf` is unstable
//[llbc]~? ERROR the linker flavor `llbc` is unstable
