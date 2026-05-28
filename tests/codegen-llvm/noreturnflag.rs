//@ compile-flags: -g -C no-prepopulate-passes

#![crate_type = "lib"]

#[no_mangle]
pub fn foo() -> ! {
    // CHECK: @foo() unnamed_addr #0
    loop {}
}

pub enum EmptyEnum {}

#[no_mangle]
pub fn bar() -> EmptyEnum {
    // CHECK: @bar() unnamed_addr #0
    loop {}
}

// CHECK: attributes #0 = {{{.*}} noreturn {{.*}}}

// CHECK: DISubprogram(name: "foo", {{.*}} DIFlagNoReturn
// CHECK: DISubprogram(name: "bar", {{.*}} DIFlagNoReturn
