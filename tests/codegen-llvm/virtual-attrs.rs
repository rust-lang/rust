//! Regression test for https://github.com/rust-lang/rust/issues/163856.
//!
//! Checks that reify shim containing a virtual call, does NOT inherit optimization attributes from
//! the method. Neither those deduced from a MIR body, nor #[rustc_nounwind] attribute.
//@ compile-flags: -O -C no-prepopulate-passes
//@ needs-unwind

#![crate_type = "lib"]
#![feature(rustc_attrs)]

pub trait Foo {
    #[rustc_nounwind]
    fn foo(&self, _: [u8; 1024]) {}
}
// CHECK-LABEL: ; <dyn virtual_attrs::Foo as virtual_attrs::Foo>::foo::{shim:reify#0}
// CHECK-NEXT: ; Function Attrs:
// CHECK-NOT: nounwind
// Notably readonly and captures(none) should NOT be present on the last argument
// CHECK-NEXT: define {{.*}}({{.*}}, ptr{{( dead_on_return)?}} noalias nofree noundef align 1 captures(address){{( dead_on_return)?}} dereferenceable(1024) %_2)
pub static A: fn(&'static dyn Foo, [u8; 1024]) = <dyn Foo as Foo>::foo;
