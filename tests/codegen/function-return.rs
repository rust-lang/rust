// Test that the `fn_ret_thunk_extern` function attribute is (not) emitted when
// the `-Zfunction-return={keep,thunk-extern}` flag is (not) set.

//@ add-core-stubs
//@ revisions: unset keep thunk-extern keep-thunk-extern thunk-extern-keep
//@ needs-llvm-components: x86
//@ compile-flags: --target x86_64-unknown-linux-gnu
//@ [keep] compile-flags: -Zfunction-return=keep
//@ [thunk-extern] compile-flags: -Zfunction-return=thunk-extern
//@ [keep-thunk-extern] compile-flags: -Zfunction-return=keep -Zfunction-return=thunk-extern
//@ [thunk-extern-keep] compile-flags: -Zfunction-return=thunk-extern -Zfunction-return=keep

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub fn foo() {
    // CHECK: @foo() unnamed_addr #0

    // unset-NOT: fn_ret_thunk_extern
    // keep-NOT: fn_ret_thunk_extern
    // thunk-extern: attributes #0 = { {{.*}}fn_ret_thunk_extern{{.*}} }
    // keep-thunk-extern: attributes #0 = { {{.*}}fn_ret_thunk_extern{{.*}} }
    // thunk-extern-keep-NOT: fn_ret_thunk_extern
}

// unset-NOT: !{{[0-9]+}} = !{i32 4, !"function_return_thunk_extern", i32 1}
// keep-NOT: !{{[0-9]+}} = !{i32 4, !"function_return_thunk_extern", i32 1}
// thunk-extern: !{{[0-9]+}} = !{i32 4, !"function_return_thunk_extern", i32 1}
// keep-thunk-extern: !{{[0-9]+}} = !{i32 4, !"function_return_thunk_extern", i32 1}
// thunk-extern-keep-NOT: !{{[0-9]+}} = !{i32 4, !"function_return_thunk_extern", i32 1}
