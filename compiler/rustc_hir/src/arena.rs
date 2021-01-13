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
            [few] hir_krate: rustc_hir::Crate<$tcx>,
            [] arm: rustc_hir::Arm<$tcx>,
            [] asm_operand: (rustc_hir::InlineAsmOperand<$tcx>, Span),
            [] asm_template: rustc_ast::InlineAsmTemplatePiece,
            [] attribute: rustc_ast::Attribute,
            [] block: rustc_hir::Block<$tcx>,
            [] bare_fn_ty: rustc_hir::BareFnTy<$tcx>,
            [few] global_asm: rustc_hir::GlobalAsm,
            [] generic_arg: rustc_hir::GenericArg<$tcx>,
            [] generic_args: rustc_hir::GenericArgs<$tcx>,
            [] generic_bound: rustc_hir::GenericBound<$tcx>,
            [] generic_param: rustc_hir::GenericParam<$tcx>,
            [] expr: rustc_hir::Expr<$tcx>,
            [] field: rustc_hir::Field<$tcx>,
            [] field_pat: rustc_hir::FieldPat<$tcx>,
            [] fn_decl: rustc_hir::FnDecl<$tcx>,
            [] foreign_item: rustc_hir::ForeignItem<$tcx>,
            [few] foreign_item_ref: rustc_hir::ForeignItemRef<$tcx>,
            [] impl_item_ref: rustc_hir::ImplItemRef<$tcx>,
            [few] inline_asm: rustc_hir::InlineAsm<$tcx>,
            [few] llvm_inline_asm: rustc_hir::LlvmInlineAsm<$tcx>,
            [] local: rustc_hir::Local<$tcx>,
            [few] macro_def: rustc_hir::MacroDef<$tcx>,
            [] param: rustc_hir::Param<$tcx>,
            [] pat: rustc_hir::Pat<$tcx>,
            [] path: rustc_hir::Path<$tcx>,
            [] path_segment: rustc_hir::PathSegment<$tcx>,
            [] poly_trait_ref: rustc_hir::PolyTraitRef<$tcx>,
            [] qpath: rustc_hir::QPath<$tcx>,
            [] stmt: rustc_hir::Stmt<$tcx>,
            [] struct_field: rustc_hir::StructField<$tcx>,
            [] trait_item_ref: rustc_hir::TraitItemRef,
            [] ty: rustc_hir::Ty<$tcx>,
            [] type_binding: rustc_hir::TypeBinding<$tcx>,
            [] variant: rustc_hir::Variant<$tcx>,
            [] where_predicate: rustc_hir::WherePredicate<$tcx>,
        ], $tcx);
    )
}
