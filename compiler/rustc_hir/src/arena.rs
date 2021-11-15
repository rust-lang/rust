/// This higher-order macro declares a list of types which can be allocated by `Arena`.
///
/// Specifying the `decode` modifier will add decode impls for `&T` and `&[T]`,
/// where `T` is the type listed. These impls will appear in the implement_ty_decoder! macro.
#[macro_export]
macro_rules! arena_types {
    ($macro:path) => (
        $macro!([
            // HIR types
            [] hir_krate: rustc_hir::Crate<'tcx>,
            [] arm: rustc_hir::Arm<'tcx>,
            [] asm_operand: (rustc_hir::InlineAsmOperand<'tcx>, Span),
            [] asm_template: rustc_ast::InlineAsmTemplatePiece,
            [] attribute: rustc_ast::Attribute,
            [] block: rustc_hir::Block<'tcx>,
            [] bare_fn_ty: rustc_hir::BareFnTy<'tcx>,
            [] body: rustc_hir::Body<'tcx>,
            [] generic_arg: rustc_hir::GenericArg<'tcx>,
            [] generic_args: rustc_hir::GenericArgs<'tcx>,
            [] generic_bound: rustc_hir::GenericBound<'tcx>,
            [] generic_param: rustc_hir::GenericParam<'tcx>,
            [] expr: rustc_hir::Expr<'tcx>,
            [] expr_field: rustc_hir::ExprField<'tcx>,
            [] pat_field: rustc_hir::PatField<'tcx>,
            [] fn_decl: rustc_hir::FnDecl<'tcx>,
            [] foreign_item: rustc_hir::ForeignItem<'tcx>,
            [] foreign_item_ref: rustc_hir::ForeignItemRef,
            [] impl_item: rustc_hir::ImplItem<'tcx>,
            [] impl_item_ref: rustc_hir::ImplItemRef,
            [] item: rustc_hir::Item<'tcx>,
            [] inline_asm: rustc_hir::InlineAsm<'tcx>,
            [] llvm_inline_asm: rustc_hir::LlvmInlineAsm<'tcx>,
            [] local: rustc_hir::Local<'tcx>,
            [] mod_: rustc_hir::Mod<'tcx>,
            [] owner_info: rustc_hir::OwnerInfo<'tcx>,
            [] param: rustc_hir::Param<'tcx>,
            [] pat: rustc_hir::Pat<'tcx>,
            [] path: rustc_hir::Path<'tcx>,
            [] path_segment: rustc_hir::PathSegment<'tcx>,
            [] poly_trait_ref: rustc_hir::PolyTraitRef<'tcx>,
            [] qpath: rustc_hir::QPath<'tcx>,
            [] stmt: rustc_hir::Stmt<'tcx>,
            [] field_def: rustc_hir::FieldDef<'tcx>,
            [] trait_item: rustc_hir::TraitItem<'tcx>,
            [] trait_item_ref: rustc_hir::TraitItemRef,
            [] ty: rustc_hir::Ty<'tcx>,
            [] type_binding: rustc_hir::TypeBinding<'tcx>,
            [] variant: rustc_hir::Variant<'tcx>,
            [] where_predicate: rustc_hir::WherePredicate<'tcx>,
        ]);
    )
}
