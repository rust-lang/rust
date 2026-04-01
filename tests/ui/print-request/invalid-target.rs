//@ compile-flags: --target x86_64-unknown-linux-gnu --print deployment-target
//@ needs-llvm-components: x86

fn main() {}

//~? ERROR only Apple targets currently support deployment version info
