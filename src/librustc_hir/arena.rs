/// This declares a list of types which can be allocated by `Arena`.
///
/// The `few` modifier will cause allocation to use the shared arena and recording the destructor.
/// This is faster and more memory efficient if there's only a few allocations of the type.
/// Leaving `few` out will cause the type to get its own dedicated `TypedArena` which is
/// faster and more memory efficient if there is lots of allocations.
///
/// Specifying the `decode` modifier will add decode impls for `&T` and `&[T]`,
/// where `T` is the type listed. These impls will appear in the implement_ty_decoder! macro.
#[macro_export]
macro_rules! arena_types {
    ($macro:path, $args:tt, $tcx:lifetime) => (
        $macro!($args, [
            // HIR types
            [few] hir_krate: rustc_hir::Crate<$tcx>, rustc_hir::Crate<'_x>;
            [] arm: rustc_hir::Arm<$tcx>, rustc_hir::Arm<'_x>;
            [] asm_operand: rustc_hir::InlineAsmOperand<$tcx>, rustc_hir::InlineAsmOperand<'_x>;
            [] asm_template: rustc_ast::ast::InlineAsmTemplatePiece, rustc_ast::ast::InlineAsmTemplatePiece;
            [] attribute: rustc_ast::ast::Attribute, rustc_ast::ast::Attribute;
            [] block: rustc_hir::Block<$tcx>, rustc_hir::Block<'_x>;
            [] bare_fn_ty: rustc_hir::BareFnTy<$tcx>, rustc_hir::BareFnTy<'_x>;
            [few] global_asm: rustc_hir::GlobalAsm, rustc_hir::GlobalAsm;
            [] generic_arg: rustc_hir::GenericArg<$tcx>, rustc_hir::GenericArg<'_x>;
            [] generic_args: rustc_hir::GenericArgs<$tcx>, rustc_hir::GenericArgs<'_x>;
            [] generic_bound: rustc_hir::GenericBound<$tcx>, rustc_hir::GenericBound<'_x>;
            [] generic_param: rustc_hir::GenericParam<$tcx>, rustc_hir::GenericParam<'_x>;
            [] expr: rustc_hir::Expr<$tcx>, rustc_hir::Expr<'_x>;
            [] field: rustc_hir::Field<$tcx>, rustc_hir::Field<'_x>;
            [] field_pat: rustc_hir::FieldPat<$tcx>, rustc_hir::FieldPat<'_x>;
            [] fn_decl: rustc_hir::FnDecl<$tcx>, rustc_hir::FnDecl<'_x>;
            [] foreign_item: rustc_hir::ForeignItem<$tcx>, rustc_hir::ForeignItem<'_x>;
            [] impl_item_ref: rustc_hir::ImplItemRef<$tcx>, rustc_hir::ImplItemRef<'_x>;
            [few] inline_asm: rustc_hir::InlineAsm<$tcx>, rustc_hir::InlineAsm<'_x>;
            [few] llvm_inline_asm: rustc_hir::LlvmInlineAsm<$tcx>, rustc_hir::LlvmInlineAsm<'_x>;
            [] local: rustc_hir::Local<$tcx>, rustc_hir::Local<'_x>;
            [few] macro_def: rustc_hir::MacroDef<$tcx>, rustc_hir::MacroDef<'_x>;
            [] param: rustc_hir::Param<$tcx>, rustc_hir::Param<'_x>;
            [] pat: rustc_hir::Pat<$tcx>, rustc_hir::Pat<'_x>;
            [] path: rustc_hir::Path<$tcx>, rustc_hir::Path<'_x>;
            [] path_segment: rustc_hir::PathSegment<$tcx>, rustc_hir::PathSegment<'_x>;
            [] poly_trait_ref: rustc_hir::PolyTraitRef<$tcx>, rustc_hir::PolyTraitRef<'_x>;
            [] qpath: rustc_hir::QPath<$tcx>, rustc_hir::QPath<'_x>;
            [] stmt: rustc_hir::Stmt<$tcx>, rustc_hir::Stmt<'_x>;
            [] struct_field: rustc_hir::StructField<$tcx>, rustc_hir::StructField<'_x>;
            [] trait_item_ref: rustc_hir::TraitItemRef, rustc_hir::TraitItemRef;
            [] ty: rustc_hir::Ty<$tcx>, rustc_hir::Ty<'_x>;
            [] type_binding: rustc_hir::TypeBinding<$tcx>, rustc_hir::TypeBinding<'_x>;
            [] variant: rustc_hir::Variant<$tcx>, rustc_hir::Variant<'_x>;
            [] where_predicate: rustc_hir::WherePredicate<$tcx>, rustc_hir::WherePredicate<'_x>;
        ], $tcx);
    )
}
