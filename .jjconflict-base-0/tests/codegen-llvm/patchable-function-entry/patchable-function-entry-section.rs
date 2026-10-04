//@ compile-flags: -Z patchable-function-entry=15,10,default_foo_section
//

#![feature(patchable_function_entry)]
#![crate_type = "lib"]

// This should have the default, as set by the compile flags
#[no_mangle]
pub fn fun0() {}

// This should override the default section name
#[no_mangle]
#[patchable_function_entry(section = "bar_section")]
pub fn fun1() {}

// CHECK: @fun0() unnamed_addr #0
// CHECK: @fun1() unnamed_addr #1

// CHECK: attributes #0 = { {{.*}}"patchable-function-entry"="5"{{.*}}"patchable-function-entry-section"="default_foo_section"{{.*}}"patchable-function-prefix"="10" {{.*}} }
// CHECK: attributes #1 = { {{.*}}"patchable-function-entry"="5"{{.*}}"patchable-function-entry-section"="bar_section"{{.*}}"patchable-function-prefix"="10" {{.*}} }
