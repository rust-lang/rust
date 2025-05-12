// Even though this test only checks 2 of the 10 or so unstable linker flavors, it exercizes the
// unique codepath checking all unstable options (see `LinkerFlavorCli::is_unstable` and its
// caller). If it passes, all the other unstable options are rejected as well.
//
//@ revisions: bpf ptx
//@ [bpf] compile-flags: --target=bpfel-unknown-none -C linker-flavor=bpf --crate-type=rlib
//@ [bpf] needs-llvm-components:
//@ [ptx] compile-flags: --target=nvptx64-nvidia-cuda -C linker-flavor=ptx --crate-type=rlib
//@ [ptx] needs-llvm-components:

#![feature(no_core)]
#![no_core]

//[bpf]~? ERROR the linker flavor `bpf` is unstable
//[ptx]~? ERROR the linker flavor `ptx` is unstable
