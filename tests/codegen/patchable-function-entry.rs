#![feature(patchable_function_entry)]
// compile-flags: -Z patchable-function-entry=15,10

#![crate_type = "lib"]

// This should have the default, as set by the compile flags
#[no_mangle]
pub fn foo() {}

// The attribute should override the compile flags
#[no_mangle]
#[patchable_function_entry(prefix(1), entry(2))]
pub fn bar() {}

// If we override an attribute to 0 or unset, the attribute should go away
#[no_mangle]
#[patchable_function_entry(entry(0))]
pub fn baz() {}

// CHECK: @foo() unnamed_addr #0
// CHECK: @bar() unnamed_addr #1
// CHECK: @baz() unnamed_addr #2

// CHECK: attributes #0 = { {{.*}}"patchable-function-entry"="5"{{.*}}"patchable-function-prefix"="10" {{.*}} }
// CHECK: attributes #1 = { {{.*}}"patchable-function-entry"="2"{{.*}}"patchable-function-prefix"="1" {{.*}} }
// CHECK-NOT: attributes #2 = { {{.*}}patchable-function-entry{{.*}} }
// CHECK-NOT: attributes #2 = { {{.*}}patchable-function-prefix{{.*}} }
// CHECK: attributes #2 = { {{.*}} }
