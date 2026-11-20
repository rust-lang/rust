// Test that the `no-jump-tables` function attribute are (not) emitted when
// the `-Cjump-tables=no` flag is (not) set.

//@ add-minicore
//@ revisions: unset set_no set_yes
//@ needs-llvm-components: x86
//@ compile-flags: --target x86_64-unknown-linux-gnu
//@ [set_no] compile-flags: -Cjump-tables=no
//@ [set_yes] compile-flags: -Cjump-tables=yes

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub fn foo() {
    // CHECK: @foo() unnamed_addr #0

    // unset-NOT: attributes #0 = { {{.*}}"no-jump-tables"="true"{{.*}} }
    // set_yes-NOT: attributes #0 = { {{.*}}"no-jump-tables"="true"{{.*}} }
    // set_no: attributes #0 = { {{.*}}"no-jump-tables"="true"{{.*}} }
}
