use rustdoc_json_types::ItemEnum;

pub(crate) fn can_appear_in_mod(kind: &ItemEnum) -> bool {
    match kind {
        ItemEnum::Module(_) => true,
        ItemEnum::ExternCrate { .. } => true,
        ItemEnum::Import(_) => true,
        ItemEnum::Union(_) => true,
        ItemEnum::Struct(_) => true,
        ItemEnum::StructField(_) => false, // Only in structs or variants
        ItemEnum::Enum(_) => true,
        ItemEnum::Variant(_) => false, // Only in enums
        ItemEnum::Function(_) => true,
        ItemEnum::Trait(_) => true,
        ItemEnum::TraitAlias(_) => true,
        ItemEnum::Method(_) => false, // Only in traits
        ItemEnum::Impl(_) => true,
        ItemEnum::Typedef(_) => true,
        ItemEnum::OpaqueTy(_) => todo!("IDK"), // On
        ItemEnum::Constant(_) => true,
        ItemEnum::Static(_) => true,
        ItemEnum::ForeignType => todo!("IDK"),
        ItemEnum::Macro(_) => true,
        ItemEnum::ProcMacro(_) => true,
        ItemEnum::PrimitiveType(_) => todo!("IDK"),
        ItemEnum::AssocConst { .. } => false, // Trait Only
        ItemEnum::AssocType { .. } => false,  // Trait only
    }
}
