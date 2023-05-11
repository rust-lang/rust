/// This higher-order macro declares a list of types which can be allocated by `Arena`.
/// Note that all `Copy` types can be allocated by default and need not be specified here.
#[macro_export]
macro_rules! arena_types {
    ($macro:path) => (
        $macro!([
            // HIR types
            [] hir_krate: rustc_hir::Crate<'tcx>,
            [] asm_template: rustc_ast::InlineAsmTemplatePiece,
            [] attribute: rustc_ast::Attribute,
            [] owner_info: rustc_hir::OwnerInfo<'tcx>,
            [] use_path: rustc_hir::UsePath<'tcx>,
            [] lit: rustc_hir::Lit,
            [] macro_def: rustc_ast::MacroDef,
        ]);
    )
}
