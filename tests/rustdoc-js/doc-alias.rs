/// Doc for `Struct`
#[doc(alias = "StructItem")]
pub struct Struct {
    /// Doc for `Struct`'s `field`
    #[doc(alias = "StructFieldItem")]
    pub field: u32,
}

impl Struct {
    /// Doc for `Struct::ImplConstItem`
    #[doc(alias = "StructImplConstItem")]
    pub const ImplConstItem: i32 = 0;
    /// Doc for `Struct::method`
    #[doc(alias = "StructMethodItem")]
    pub fn method(&self) {}
}

impl Trait for Struct {
    type Target = u32;
    const AssociatedConst: i32 = 12;

    /// Doc for `Trait::function` implemented for Struct
    #[doc(alias = "ImplTraitFunction")]
    fn function() -> Self::Target {
        0
    }
}

/// Doc for `Enum`
#[doc(alias = "EnumItem")]
pub enum Enum {
    /// Doc for `Enum::Variant`
    #[doc(alias = "VariantItem")]
    Variant,
}

impl Enum {
    /// Doc for `Enum::method`
    #[doc(alias = "EnumMethodItem")]
    pub fn method(&self) {}
}

/// Doc for type alias `Typedef`
#[doc(alias = "TypedefItem")]
pub type Typedef = i32;

/// Doc for `Trait`
#[doc(alias = "TraitItem")]
pub trait Trait {
    /// Doc for `Trait::Target`
    #[doc(alias = "TraitTypeItem")]
    type Target;
    /// Doc for `Trait::AssociatedConst`
    #[doc(alias = "AssociatedConstItem")]
    const AssociatedConst: i32;

    /// Doc for `Trait::function`
    #[doc(alias = "TraitFunctionItem")]
    fn function() -> Self::Target;
}

/// Doc for `function`
#[doc(alias = "FunctionItem")]
pub fn function() {}

/// Doc for `Module`
#[doc(alias = "ModuleItem")]
pub mod Module {}

/// Doc for `Const`
#[doc(alias = "ConstItem")]
pub const Const: u32 = 0;

/// Doc for `Static`
#[doc(alias = "StaticItem")]
pub static Static: u32 = 0;

/// Doc for `Union`
#[doc(alias = "UnionItem")]
pub union Union {
    /// Doc for `Union::union_item`
    #[doc(alias = "UnionFieldItem")]
    pub union_item: u32,
    pub y: f32,
}

impl Union {
    /// Doc for `Union::method`
    #[doc(alias = "UnionMethodItem")]
    pub fn method(&self) {}
}

/// Doc for `Macro`
#[doc(alias = "MacroItem")]
#[macro_export]
macro_rules! Macro {
    () => {};
}
