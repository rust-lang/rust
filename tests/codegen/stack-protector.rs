//@ revisions: all strong basic none
//@ ignore-nvptx64 stack protector not supported
//@ [all] compile-flags: -Z stack-protector=all
//@ [strong] compile-flags: -Z stack-protector=strong
//@ [basic] compile-flags: -Z stack-protector=basic

#![crate_type = "lib"]

#[no_mangle]
pub fn foo() {
    // CHECK: @foo() unnamed_addr #0

    // CHECK-ALL-NOT: attributes #0 = { {{.*}}sspstrong {{.*}} }
    // CHECK-ALL-NOT: attributes #0 = { {{.*}}ssp {{.*}} }
    // CHECK-ALL: attributes #0 = { {{.*}}sspreq {{.*}} }
    // CHECK-ALL-NOT: attributes #0 = { {{.*}}sspstrong {{.*}} }
    // CHECK-ALL-NOT: attributes #0 = { {{.*}}ssp {{.*}} }

    // CHECK-STRONG-NOT: attributes #0 = { {{.*}}sspreq {{.*}} }
    // CHECK-STRONG-NOT: attributes #0 = { {{.*}}ssp {{.*}} }
    // CHECK-STRONG: attributes #0 = { {{.*}}sspstrong {{.*}} }
    // CHECK-STRONG-NOT: attributes #0 = { {{.*}}sspreq {{.*}} }
    // CHECK-STRONG-NOT: attributes #0 = { {{.*}}ssp {{.*}} }

    // CHECK-BASIC-NOT: attributes #0 = { {{.*}}sspreq {{.*}} }
    // CHECK-BASIC-NOT: attributes #0 = { {{.*}}sspstrong {{.*}} }
    // CHECK-BASIC: attributes #0 = { {{.*}}ssp {{.*}} }
    // CHECK-BASIC-NOT: attributes #0 = { {{.*}}sspreq {{.*}} }
    // CHECK-BASIC-NOT: attributes #0 = { {{.*}}sspstrong {{.*}} }

    // CHECK-NONE-NOT: attributes #0 = { {{.*}}sspreq {{.*}} }
    // CHECK-NONE-NOT: attributes #0 = { {{.*}}sspstrong {{.*}} }
    // CHECK-NONE-NOT: attributes #0 = { {{.*}}ssp {{.*}} }
}
