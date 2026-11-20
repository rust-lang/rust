// ignore-tidy-file-linelength
//@ revisions: all all-z strong strong-z none strong-c-overrides-z
//@ ignore-nvptx64 stack protector not supported
//@ [all] compile-flags: -C stack-protector=all -Z allow-partial-mitigations=stack-protector
//@ [all-z] compile-flags: -Z stack-protector=all -Z allow-partial-mitigations=stack-protector
//@ [strong] compile-flags: -C stack-protector=strong -Z allow-partial-mitigations=stack-protector
//@ [strong-z] compile-flags: -Z stack-protector=strong -Z allow-partial-mitigations=stack-protector
//@ [strong-c-overrides-z] compile-flags: -C stack-protector=strong -Z stack-protector=all -Z allow-partial-mitigations=stack-protector

#![crate_type = "lib"]

#[no_mangle]
pub fn foo() {
    // CHECK: @foo() unnamed_addr #0

    // all-NOT: attributes #0 = { {{.*}}sspstrong {{.*}} }
    // all-NOT: attributes #0 = { {{.*}}ssp {{.*}} }
    // all: attributes #0 = { {{.*}}sspreq {{.*}} }
    // all-NOT: attributes #0 = { {{.*}}sspstrong {{.*}} }
    // all-NOT: attributes #0 = { {{.*}}ssp {{.*}} }

    // all-z-NOT: attributes #0 = { {{.*}}sspstrong {{.*}} }
    // all-z-NOT: attributes #0 = { {{.*}}ssp {{.*}} }
    // all-z: attributes #0 = { {{.*}}sspreq {{.*}} }
    // all-z-NOT: attributes #0 = { {{.*}}sspstrong {{.*}} }
    // all-z-NOT: attributes #0 = { {{.*}}ssp {{.*}} }

    // strong-NOT: attributes #0 = { {{.*}}sspreq {{.*}} }
    // strong-NOT: attributes #0 = { {{.*}}ssp {{.*}} }
    // strong: attributes #0 = { {{.*}}sspstrong {{.*}} }
    // strong-NOT: attributes #0 = { {{.*}}sspreq {{.*}} }
    // strong-NOT: attributes #0 = { {{.*}}ssp {{.*}} }

    // strong-z-NOT: attributes #0 = { {{.*}}sspreq {{.*}} }
    // strong-z-NOT: attributes #0 = { {{.*}}ssp {{.*}} }
    // strong-z: attributes #0 = { {{.*}}sspstrong {{.*}} }
    // strong-z-NOT: attributes #0 = { {{.*}}sspreq {{.*}} }
    // strong-z-NOT: attributes #0 = { {{.*}}ssp {{.*}} }

    // strong-c-overrides-z-NOT: attributes #0 = { {{.*}}sspreq {{.*}} }
    // strong-c-overrides-z-NOT: attributes #0 = { {{.*}}ssp {{.*}} }
    // strong-c-overrides-z: attributes #0 = { {{.*}}sspstrong {{.*}} }
    // strong-c-overrides-z-NOT: attributes #0 = { {{.*}}sspreq {{.*}} }
    // strong-c-overrides-z-NOT: attributes #0 = { {{.*}}ssp {{.*}} }

    // none-NOT: attributes #0 = { {{.*}}sspreq {{.*}} }
    // none-NOT: attributes #0 = { {{.*}}sspstrong {{.*}} }
    // none-NOT: attributes #0 = { {{.*}}ssp {{.*}} }
}
