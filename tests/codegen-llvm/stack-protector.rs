//@ revisions: all strong basic none
//@ ignore-nvptx64 stack protector not supported
//@ [all] compile-flags: -Z stack-protector=all -Z allow-partial-mitigations=stack-protector
//@ [strong] compile-flags: -Z stack-protector=strong -Z allow-partial-mitigations=stack-protector
//@ [basic] compile-flags: -Z stack-protector=basic -Z allow-partial-mitigations=stack-protector

#![crate_type = "lib"]

#[no_mangle]
pub fn foo() {
    // CHECK: @foo() unnamed_addr #0

    // all-NOT: attributes #0 = { {{.*}}sspstrong {{.*}} }
    // all-NOT: attributes #0 = { {{.*}}ssp {{.*}} }
    // all: attributes #0 = { {{.*}}sspreq {{.*}} }
    // all-NOT: attributes #0 = { {{.*}}sspstrong {{.*}} }
    // all-NOT: attributes #0 = { {{.*}}ssp {{.*}} }

    // strong-NOT: attributes #0 = { {{.*}}sspreq {{.*}} }
    // strong-NOT: attributes #0 = { {{.*}}ssp {{.*}} }
    // strong: attributes #0 = { {{.*}}sspstrong {{.*}} }
    // strong-NOT: attributes #0 = { {{.*}}sspreq {{.*}} }
    // strong-NOT: attributes #0 = { {{.*}}ssp {{.*}} }

    // basic-NOT: attributes #0 = { {{.*}}sspreq {{.*}} }
    // basic-NOT: attributes #0 = { {{.*}}sspstrong {{.*}} }
    // basic: attributes #0 = { {{.*}}ssp {{.*}} }
    // basic-NOT: attributes #0 = { {{.*}}sspreq {{.*}} }
    // basic-NOT: attributes #0 = { {{.*}}sspstrong {{.*}} }

    // none-NOT: attributes #0 = { {{.*}}sspreq {{.*}} }
    // none-NOT: attributes #0 = { {{.*}}sspstrong {{.*}} }
    // none-NOT: attributes #0 = { {{.*}}ssp {{.*}} }
}
