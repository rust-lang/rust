// ignore-tidy-linelength
// Test that the `harden-sls-ijmp`, `harden-sls-ret` target features is (not) emitted when
// the `harden-sls=[none|all|ret|indirect-jmp]` flag is (not) set.

//@ add-minicore
//@ revisions: unset all ret indirect_jmp
//@ needs-llvm-components: x86
//@ compile-flags: --target x86_64-unknown-linux-gnu
//@ [unset] compile-flags: -Zharden-sls=none
//@ [all] compile-flags: -Zharden-sls=all
//@ [ret] compile-flags: -Zharden-sls=return
//@ [indirect_jmp] compile-flags: -Zharden-sls=indirect-jmp

#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub fn foo() {
    // CHECK: @foo() unnamed_addr #0

    // unset-NOT: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+harden-sls-ijmp{{.*}} }
    // unset-NOT: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+harden-sls-ret{{.*}} }

    // all: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+harden-sls-ijmp,+harden-sls-ret{{.*}} }

    // ret-NOT: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+harden-sls-ijmp{{.*}} }
    // ret: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+harden-sls-ret{{.*}} }

    // indirect_jmp-NOT: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+harden-sls-ret{{.*}} }
    // indirect_jmp: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+harden-sls-ijmp{{.*}} }
}
