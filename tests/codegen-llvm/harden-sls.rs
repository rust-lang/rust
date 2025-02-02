// ignore-tidy-linelength
// Test that the `harden-sls-ijmp`, `harden-sls-ret` target features is (not) emitted when
// the `harden-sls=[none|all|return|indirect-jmp]` flag is (not) set.

//@ add-core-stubs
//@ revisions: none all return indirect_jmp
//@ needs-llvm-components: x86
//@ compile-flags: --target x86_64-unknown-linux-gnu
//@ [none] compile-flags: -Zharden-sls=none
//@ [all] compile-flags: -Zharden-sls=all
//@ [return] compile-flags: -Zharden-sls=return
//@ [indirect_jmp] compile-flags: -Zharden-sls=indirect-jmp

#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub fn foo() {
    // CHECK: @foo() unnamed_addr #0

    // none-NOT: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+harden-sls-ijmp{{.*}} }
    // none-NOT: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+harden-sls-ret{{.*}} }

    // all: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+harden-sls-ijmp,+harden-sls-ret{{.*}} }

    // return-NOT: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+harden-sls-ijmp{{.*}} }
    // return: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+harden-sls-ret{{.*}} }

    // indirect_jmp-NOT: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+harden-sls-ret{{.*}} }
    // indirect_jmp: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+harden-sls-ijmp{{.*}} }
}
