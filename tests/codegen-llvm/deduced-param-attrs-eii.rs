// This test ensures that externally implementable functions do not have parameter attributes
// deduced. Attribute deduction is based on analyzing the body and the point of EII is that the body
// can be replaced downstream.

//@ compile-flags: -O -Cno-prepopulate-passes
#![crate_type = "lib"]
#![feature(extern_item_impls)]

pub struct Big {
    pub blah: [i32; 1024],
}

// CHECK-LABEL: @external(ptr {{.*}}readonly {{.*}}captures(none) {{.*}}%_b)
#[unsafe(no_mangle)]
pub fn external(_b: Big) {}

// CHECK-LABEL: ; deduced_param_attrs_eii::decl_default
// CHECK-NEXT:  ;
// CHECK-NEXT:  define weak void
// CHECK-NOT:   readonly
// CHECK-NOT:   captures(none)
// CHECK-SAME:  %0)
#[eii]
pub fn decl_default(_b: Big) {}
