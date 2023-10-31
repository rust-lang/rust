// This tests that optimized enum debug info accurately reflects the enum layout.
// This is ignored for the fallback mode on MSVC due to problems with PDB.

//
// ignore-msvc

// compile-flags: -g -C no-prepopulate-passes

// CHECK: {{.*}}DICompositeType{{.*}}tag: DW_TAG_variant_part,{{.*}}size: 32,{{.*}}
// CHECK: {{.*}}DIDerivedType{{.*}}tag: DW_TAG_member,{{.*}}name: "Placeholder",{{.*}}extraData: i128 4294967295{{[,)].*}}
// CHECK: {{.*}}DIDerivedType{{.*}}tag: DW_TAG_member,{{.*}}name: "Error",{{.*}}extraData: i128 0{{[,)].*}}

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
