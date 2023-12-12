// compile-flags: -Z patchable-function-entry=15,10

#![crate_type = "lib"]

#[no_mangle]
pub fn foo() {}
// CHECK: @foo() unnamed_addr #0
// CHECK: attributes #0 = { {{.*}}"patchable-function-entry"="5"{{.*}}"patchable-function-prefix"="10" {{.*}} }
