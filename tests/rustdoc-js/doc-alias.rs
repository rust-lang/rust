#[doc(alias = "StructItem")]
pub struct Struct {
    #[doc(alias = "StructFieldItem")]
    pub field: u32,
}

impl Struct {
    #[doc(alias = "StructImplConstItem")]
    pub const ImplConstItem: i32 = 0;
    #[doc(alias = "StructMethodItem")]
    pub fn method(&self) {}
}

impl Trait for Struct {
    type Target = u32;
    const AssociatedConst: i32 = 12;

    #[doc(alias = "ImplTraitFunction")]
    fn function() -> Self::Target { 0 }
}

#[doc(alias = "EnumItem")]
pub enum Enum {
    #[doc(alias = "VariantItem")]
    Variant,
}

impl Enum {
    #[doc(alias = "EnumMethodItem")]
    pub fn method(&self) {}
}

#[doc(alias = "TypedefItem")]
pub type Typedef = i32;

#[doc(alias = "TraitItem")]
pub trait Trait {
    #[doc(alias = "TraitTypeItem")]
    type Target;
    #[doc(alias = "AssociatedConstItem")]
    const AssociatedConst: i32;

    #[doc(alias = "TraitFunctionItem")]
    fn function() -> Self::Target;
}

#[doc(alias = "FunctionItem")]
pub fn function() {}

#[doc(alias = "ModuleItem")]
pub mod Module {}

#[doc(alias = "ConstItem")]
pub const Const: u32 = 0;

#[doc(alias = "StaticItem")]
pub static Static: u32 = 0;

#[doc(alias = "UnionItem")]
pub union Union {
    #[doc(alias = "UnionFieldItem")]
    pub union_item: u32,
    pub y: f32,
}

impl Union {
    #[doc(alias = "UnionMethodItem")]
    pub fn method(&self) {}
}

#[doc(alias = "MacroItem")]
#[macro_export]
macro_rules! Macro {
    () => {}
}
