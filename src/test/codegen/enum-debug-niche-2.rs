// This test depends on a patch that was committed to upstream LLVM
// before 7.0, then backported to the Rust LLVM fork.  It tests that
// optimized enum debug info accurately reflects the enum layout.

// ignore-tidy-linelength
// ignore-windows
// min-system-llvm-version 8.0

// compile-flags: -g -C no-prepopulate-passes

// CHECK: {{.*}}DICompositeType{{.*}}tag: DW_TAG_variant_part,{{.*}}size: 32,{{.*}}
// CHECK: {{.*}}DIDerivedType{{.*}}tag: DW_TAG_member,{{.*}}name: "Placeholder",{{.*}}extraData: i64 4294967295{{[,)].*}}
// CHECK: {{.*}}DIDerivedType{{.*}}tag: DW_TAG_member,{{.*}}name: "Error",{{.*}}extraData: i64 0{{[,)].*}}

#![feature(never_type)]

#[derive(Copy, Clone)]
pub struct Entity {
    private: std::num::NonZeroU32,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Declaration;

impl TypeFamily for Declaration {
    type Base = Base;
    type Placeholder = !;

    fn intern_base_data(_: BaseKind<Self>) {}
}

#[derive(Copy, Clone)]
pub struct Base;

pub trait TypeFamily: Copy + 'static {
    type Base: Copy;
    type Placeholder: Copy;

    fn intern_base_data(_: BaseKind<Self>);
}

#[derive(Copy, Clone)]
pub enum BaseKind<F: TypeFamily> {
    Named(Entity),
    Placeholder(F::Placeholder),
    Error,
}

pub fn main() {
    let x = BaseKind::Error::<Declaration>;
    let y = 7;
}
