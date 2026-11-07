//@ compile-flags: -Z patchable-function-entry=15

#![feature(patchable_function_entry)]
#![crate_type = "lib"]

// This should have the default, as set by the compile flags
#[no_mangle]
pub fn fun0() {}

// The attribute should override the compile flags
#[no_mangle]
#[patchable_function_entry(prefix_nops = 1, entry_nops = 2)]
pub fn fun1() {}

// If we override an attribute to 0 or unset, the attribute should go away
#[no_mangle]
#[patchable_function_entry(entry_nops = 0)]
pub fn fun2() {}

// The attribute should override the compile flags
#[no_mangle]
#[patchable_function_entry(prefix_nops = 20, entry_nops = 1)]
pub fn fun3() {}

// The attribute should override the compile flags
#[no_mangle]
#[patchable_function_entry(prefix_nops = 2, entry_nops = 19)]
pub fn fun4() {}

// The attribute should override patchable-function-entry to 3
// and patchable-function-prefix to the default of 0, clearing it entirely
#[no_mangle]
#[patchable_function_entry(entry_nops = 3)]
pub fn fun5() {}

// The attribute should override patchable-function-prefix to 4
// and patchable-function-entry to the default of 0, clearing it entirely
#[no_mangle]
#[patchable_function_entry(prefix_nops = 4)]
pub fn fun6() {}

// CHECK: @fun0() unnamed_addr #0
// CHECK: @fun1() unnamed_addr #1
// CHECK: @fun2() unnamed_addr #2
// CHECK: @fun3() unnamed_addr #3
// CHECK: @fun4() unnamed_addr #4
// CHECK: @fun5() unnamed_addr #5
// CHECK: @fun6() unnamed_addr #6

// CHECK: attributes #0 = { {{.*}}"patchable-function-entry"="15" {{.*}} }
// CHECK-NOT: attributes #0 = { {{.*}}patchable-function-prefix{{.*}} }

// CHECK: attributes #1 = { {{.*}}"patchable-function-entry"="2"{{.*}}"patchable-function-prefix"="1" {{.*}} }

// CHECK-NOT: attributes #2 = { {{.*}}patchable-function-entry{{.*}} }
// CHECK-NOT: attributes #2 = { {{.*}}patchable-function-prefix{{.*}} }
// CHECK: attributes #2 = { {{.*}} }

// CHECK: attributes #3 = { {{.*}}"patchable-function-entry"="1"{{.*}}"patchable-function-prefix"="20" {{.*}} }
// CHECK: attributes #4 = { {{.*}}"patchable-function-entry"="19"{{.*}}"patchable-function-prefix"="2" {{.*}} }

// CHECK: attributes #5 = { {{.*}}"patchable-function-entry"="3"{{.*}} }
// CHECK-NOT: attributes #5 = { {{.*}}patchable-function-prefix{{.*}} }

// CHECK: attributes #6 = { {{.*}}"patchable-function-prefix"="4"{{.*}} }
// CHECK-NOT: attributes #6 = { {{.*}}patchable-function-entry{{.*}} }
