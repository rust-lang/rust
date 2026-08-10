//! Declares an arena that can allocate values of any `Copy` type, and of
//! any `!Copy` type listed below.

rustc_arena::declare_arena! {
    // HIR types
    asm_template: rustc_ast::InlineAsmTemplatePiece,
    attribute: rustc_attr_ir::Attribute,
    owner_info: crate::OwnerInfo<'tcx>,
    macro_def: rustc_ast::MacroDef,
    delegation_info: crate::DelegationInfo,
}
