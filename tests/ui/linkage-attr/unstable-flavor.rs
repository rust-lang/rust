// Even though this test only checks 2 of the 10 or so unstable linker flavors, it exercizes the
// unique codepath checking all unstable options (see `LinkerFlavorCli::is_unstable` and its
// caller). If it passes, all the other unstable options are rejected as well.
//
// revisions: bpf ptx
// [bpf] compile-flags: --target=bpfel-unknown-none -C linker-flavor=bpf-linker --crate-type=rlib
// [bpf] error-pattern: linker flavor `bpf-linker` is unstable, the `-Z unstable-options` flag
// [bpf] needs-llvm-components:
// [ptx] compile-flags: --target=nvptx64-nvidia-cuda -C linker-flavor=ptx-linker --crate-type=rlib
// [ptx] error-pattern: linker flavor `ptx-linker` is unstable, the `-Z unstable-options` flag
// [ptx] needs-llvm-components:

#![feature(no_core)]
#![no_core]
