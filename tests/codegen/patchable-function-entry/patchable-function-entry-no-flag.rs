#![feature(patchable_function_entry)]
#![crate_type = "lib"]

// No patchable function entry should be set
#[no_mangle]
pub fn fun0() {}

// The attribute should work even without compiler flags
#[no_mangle]
#[patchable_function_entry(prefix_nops = 1, entry_nops = 2)]
pub fn fun1() {}

// The attribute should work even without compiler flags
// and only set patchable-function-entry to 3.
#[no_mangle]
#[patchable_function_entry(entry_nops = 3)]
pub fn fun2() {}

// The attribute should work even without compiler flags
// and only set patchable-function-prefix to 4.
#[no_mangle]
#[patchable_function_entry(prefix_nops = 4)]
pub fn fun3() {}

// CHECK: @fun0() unnamed_addr #0
// CHECK: @fun1() unnamed_addr #1
// CHECK: @fun2() unnamed_addr #2
// CHECK: @fun3() unnamed_addr #3

// CHECK-NOT: attributes #0 = { {{.*}}patchable-function-entry{{.*}} }
// CHECK-NOT: attributes #0 = { {{.*}}patchable-function-prefix{{.*}} }

// CHECK: attributes #1 = { {{.*}}"patchable-function-entry"="2"{{.*}}"patchable-function-prefix"="1" {{.*}} }

// CHECK: attributes #2 = { {{.*}}"patchable-function-entry"="3"{{.*}} }
// CHECK-NOT: attributes #2 = { {{.*}}patchable-function-prefix{{.*}} }

// CHECK: attributes #3 = { {{.*}}"patchable-function-prefix"="4"{{.*}} }
// CHECK-NOT: attributes #3 = { {{.*}}patchable-function-entry{{.*}} }
