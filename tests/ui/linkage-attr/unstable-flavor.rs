// revisions: bpf ptx
// [bpf] compile-flags: --target=bpfel-unknown-none -C linker-flavor=bpf-linker --crate-type=rlib
// [bpf] error-pattern: linker flavor `bpf-linker` is unstable, `-Z unstable-options` flag
// [bpf] needs-llvm-components:
// [ptx] compile-flags: --target=nvptx64-nvidia-cuda -C linker-flavor=ptx-linker --crate-type=rlib
// [ptx] error-pattern: linker flavor `ptx-linker` is unstable, `-Z unstable-options` flag
// [ptx] needs-llvm-components:

#![feature(no_core)]
#![no_core]
