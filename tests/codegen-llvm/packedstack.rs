//@ add-minicore
//@ compile-flags: -Copt-level=3 --crate-type=lib --target=s390x-unknown-none-softfloat -Zpacked-stack
//@ needs-llvm-components: systemz
#![feature(s390x_target_feature)]
#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]

pub fn test_packedstack() {
    // CHECK: @test_packedstack() unnamed_addr #0
}

// CHECK: attributes #0 = { {{.*}}"packed-stack"{{.*}} }
