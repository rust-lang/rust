//! The code in this module gathers up all of the inherent impls in
//! the current crate and organizes them in a map. It winds up
//! touching the whole crate and thus must be recomputed completely
//! for any change, but it is very cheap to compute. In practice, most
//! code in the compiler never *directly* requests this map. Instead,
//! it requests the inherent impls specific to some type (via
//! `tcx.inherent_impls(def_id)`). That value, however,
//! is computed by selecting an idea from this table.

use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def_id::{CrateNum, DefId, LocalDefId, LOCAL_CRATE};
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_middle::ty::{self, CrateInherentImpls, TyCtxt};

use rustc_ast as ast;
use rustc_span::Span;

/// On-demand query: yields a map containing all types mapped to their inherent impls.
pub fn crate_inherent_impls(tcx: TyCtxt<'_>, crate_num: CrateNum) -> CrateInherentImpls {
    assert_eq!(crate_num, LOCAL_CRATE);

    let krate = tcx.hir().krate();
    let mut collect = InherentCollect { tcx, impls_map: Default::default() };
    krate.visit_all_item_likes(&mut collect);
    collect.impls_map
}

/// On-demand query: yields a vector of the inherent impls for a specific type.
pub fn inherent_impls(tcx: TyCtxt<'_>, ty_def_id: DefId) -> &[DefId] {
    assert!(ty_def_id.is_local());

    let crate_map = tcx.crate_inherent_impls(ty_def_id.krate);
    match crate_map.inherent_impls.get(&ty_def_id) {
        Some(v) => &v[..],
        None => &[],
    }
}

struct InherentCollect<'tcx> {
    tcx: TyCtxt<'tcx>,
    impls_map: CrateInherentImpls,
}

impl ItemLikeVisitor<'v> for InherentCollect<'tcx> {
    fn visit_item(&mut self, item: &hir::Item<'_>) {
        let (ty, assoc_items) = match item.kind {
            hir::ItemKind::Impl { of_trait: None, ref self_ty, items, .. } => (self_ty, items),
            _ => return,
        };

        let def_id = self.tcx.hir().local_def_id(item.hir_id);
        let self_ty = self.tcx.type_of(def_id);
        let lang_items = self.tcx.lang_items();
        match *self_ty.kind() {
            ty::Adt(def, _) => {
                self.check_def_id(item, def.did);
            }
            ty::Foreign(did) => {
                self.check_def_id(item, did);
            }
            ty::Dynamic(ref data, ..) if data.principal_def_id().is_some() => {
                self.check_def_id(item, data.principal_def_id().unwrap());
            }
            ty::Bool => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.bool_impl(),
                    None,
                    "bool",
                    "bool",
                    item.span,
                    assoc_items,
                );
            }
            ty::Char => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.char_impl(),
                    None,
                    "char",
                    "char",
                    item.span,
                    assoc_items,
                );
            }
            ty::Str => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.str_impl(),
                    lang_items.str_alloc_impl(),
                    "str",
                    "str",
                    item.span,
                    assoc_items,
                );
            }
            ty::Slice(slice_item) if slice_item == self.tcx.types.u8 => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.slice_u8_impl(),
                    lang_items.slice_u8_alloc_impl(),
                    "slice_u8",
                    "[u8]",
                    item.span,
                    assoc_items,
                );
            }
            ty::Slice(_) => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.slice_impl(),
                    lang_items.slice_alloc_impl(),
                    "slice",
                    "[T]",
                    item.span,
                    assoc_items,
                );
            }
            ty::Array(_, _) => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.array_impl(),
                    None,
                    "array",
                    "[T; N]",
                    item.span,
                    assoc_items,
                );
            }
            ty::RawPtr(ty::TypeAndMut { ty: inner, mutbl: hir::Mutability::Not })
                if matches!(inner.kind(), ty::Slice(_)) =>
            {
                self.check_primitive_impl(
                    def_id,
                    lang_items.const_slice_ptr_impl(),
                    None,
                    "const_slice_ptr",
                    "*const [T]",
                    item.span,
                    assoc_items,
                );
            }
            ty::RawPtr(ty::TypeAndMut { ty: inner, mutbl: hir::Mutability::Mut })
                if matches!(inner.kind(), ty::Slice(_)) =>
            {
                self.check_primitive_impl(
                    def_id,
                    lang_items.mut_slice_ptr_impl(),
                    None,
                    "mut_slice_ptr",
                    "*mut [T]",
                    item.span,
                    assoc_items,
                );
            }
            ty::RawPtr(ty::TypeAndMut { ty: _, mutbl: hir::Mutability::Not }) => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.const_ptr_impl(),
                    None,
                    "const_ptr",
                    "*const T",
                    item.span,
                    assoc_items,
                );
            }
            ty::RawPtr(ty::TypeAndMut { ty: _, mutbl: hir::Mutability::Mut }) => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.mut_ptr_impl(),
                    None,
                    "mut_ptr",
                    "*mut T",
                    item.span,
                    assoc_items,
                );
            }
            ty::Int(ast::IntTy::I8) => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.i8_impl(),
                    None,
                    "i8",
                    "i8",
                    item.span,
                    assoc_items,
                );
            }
            ty::Int(ast::IntTy::I16) => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.i16_impl(),
                    None,
                    "i16",
                    "i16",
                    item.span,
                    assoc_items,
                );
            }
            ty::Int(ast::IntTy::I32) => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.i32_impl(),
                    None,
                    "i32",
                    "i32",
                    item.span,
                    assoc_items,
                );
            }
            ty::Int(ast::IntTy::I64) => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.i64_impl(),
                    None,
                    "i64",
                    "i64",
                    item.span,
                    assoc_items,
                );
            }
            ty::Int(ast::IntTy::I128) => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.i128_impl(),
                    None,
                    "i128",
                    "i128",
                    item.span,
                    assoc_items,
                );
            }
            ty::Int(ast::IntTy::Isize) => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.isize_impl(),
                    None,
                    "isize",
                    "isize",
                    item.span,
                    assoc_items,
                );
            }
            ty::Uint(ast::UintTy::U8) => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.u8_impl(),
                    None,
                    "u8",
                    "u8",
                    item.span,
                    assoc_items,
                );
            }
            ty::Uint(ast::UintTy::U16) => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.u16_impl(),
                    None,
                    "u16",
                    "u16",
                    item.span,
                    assoc_items,
                );
            }
            ty::Uint(ast::UintTy::U32) => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.u32_impl(),
                    None,
                    "u32",
                    "u32",
                    item.span,
                    assoc_items,
                );
            }
            ty::Uint(ast::UintTy::U64) => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.u64_impl(),
                    None,
                    "u64",
                    "u64",
                    item.span,
                    assoc_items,
                );
            }
            ty::Uint(ast::UintTy::U128) => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.u128_impl(),
                    None,
                    "u128",
                    "u128",
                    item.span,
                    assoc_items,
                );
            }
            ty::Uint(ast::UintTy::Usize) => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.usize_impl(),
                    None,
                    "usize",
                    "usize",
                    item.span,
                    assoc_items,
                );
            }
            ty::Float(ast::FloatTy::F32) => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.f32_impl(),
                    lang_items.f32_runtime_impl(),
                    "f32",
                    "f32",
                    item.span,
                    assoc_items,
                );
            }
            ty::Float(ast::FloatTy::F64) => {
                self.check_primitive_impl(
                    def_id,
                    lang_items.f64_impl(),
                    lang_items.f64_runtime_impl(),
                    "f64",
                    "f64",
                    item.span,
                    assoc_items,
                );
            }
            ty::Error(_) => {}
            _ => {
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    ty.span,
                    E0118,
                    "no nominal type found for inherent implementation"
                );

                err.span_label(ty.span, "impl requires a nominal type")
                    .note("either implement a trait on it or create a newtype to wrap it instead");

                if let ty::Ref(_, subty, _) = self_ty.kind() {
                    err.note(&format!(
                        "you could also try moving the reference to \
                            uses of `{}` (such as `self`) within the implementation",
                        subty
                    ));
                }

                err.emit();
            }
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem<'_>) {}

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem<'_>) {}

    fn visit_foreign_item(&mut self, _foreign_item: &hir::ForeignItem<'_>) {}
}

impl InherentCollect<'tcx> {
    fn check_def_id(&mut self, item: &hir::Item<'_>, def_id: DefId) {
        if def_id.is_local() {
            // Add the implementation to the mapping from implementation to base
            // type def ID, if there is a base type for this implementation and
            // the implementation does not have any associated traits.
            let impl_def_id = self.tcx.hir().local_def_id(item.hir_id);
            let vec = self.impls_map.inherent_impls.entry(def_id).or_default();
            vec.push(impl_def_id.to_def_id());
        } else {
            struct_span_err!(
                self.tcx.sess,
                item.span,
                E0116,
                "cannot define inherent `impl` for a type outside of the crate \
                              where the type is defined"
            )
            .span_label(item.span, "impl for type defined outside of crate.")
            .note("define and implement a trait or new type instead")
            .emit();
        }
    }

    fn check_primitive_impl(
        &self,
        impl_def_id: LocalDefId,
        lang_def_id: Option<DefId>,
        lang_def_id2: Option<DefId>,
        lang: &str,
        ty: &str,
        span: Span,
        assoc_items: &[hir::ImplItemRef<'_>],
    ) {
        match (lang_def_id, lang_def_id2) {
            (Some(lang_def_id), _) if lang_def_id == impl_def_id.to_def_id() => {
                // OK
            }
            (_, Some(lang_def_id)) if lang_def_id == impl_def_id.to_def_id() => {
                // OK
            }
            _ => {
                let to_implement = if assoc_items.len() == 0 {
                    String::new()
                } else {
                    let plural = assoc_items.len() > 1;
                    let assoc_items_kind = {
                        let item_types = assoc_items.iter().map(|x| x.kind);
                        if item_types.clone().all(|x| x == hir::AssocItemKind::Const) {
                            "constant"
                        } else if item_types
                            .clone()
                            .all(|x| matches! {x, hir::AssocItemKind::Fn{ .. } })
                        {
                            "method"
                        } else {
                            "associated item"
                        }
                    };

                    format!(
                        " to implement {} {}{}",
                        if plural { "these" } else { "this" },
                        assoc_items_kind,
                        if plural { "s" } else { "" }
                    )
                };

                struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0390,
                    "only a single inherent implementation marked with `#[lang = \
                                  \"{}\"]` is allowed for the `{}` primitive",
                    lang,
                    ty
                )
                .help(&format!("consider using a trait{}", to_implement))
                .emit();
            }
        }
    }
}
