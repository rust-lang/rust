use crate::lints::{ ExplicitRangeDiag, TraitImplRangeDiag };
use crate::{LateContext, LateLintPass, LintContext};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{Ty, Binder, FnSig};
use rustc_middle::ty;
use rustc_session::config::CrateType;
use std::iter;

declare_lint! {
    /// The `explicit_range` lint detects uses of `Range`, `RangeInclusive`, 
    /// or `RangeFrom` as parameter types in public APIs.
    /// 
    /// ### Examples
    /// 
    /// ```rust
    /// pub fn takes_range(range: Range<usize>)
    /// 
    /// pub trait Foo {
    ///     fn foo(self, range: Range<u8>) {}
    /// }
    /// impl Foo for Thing {
    ///     fn foo(self, range: Range<u8>) {}
    /// }
    /// 
    /// pub trait Bar
    /// impl Bar for Range<usize>
    /// pub fn bar(b: impl Bar)
    /// 
    /// pub struct Thing
    /// impl Index<Range<usize>> for Thing
    /// ```
    /// 
    /// ### Explanation
    /// 
    /// Gather data for [concerns in RFC 3550].
    /// 
    /// [concerns in RFC 3550](https://github.com/rust-lang/rfcs/pull/3550#issuecomment-1935112286)
    pub(super) EXPLICIT_RANGE,
    Allow,
    "explicit usage of range type in public API"
}

declare_lint_pass!(ExplicitRange => [EXPLICIT_RANGE]);

impl<'tcx> LateLintPass<'tcx> for ExplicitRange {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: hir::intravisit::FnKind<'tcx>,
        decl: &'tcx hir::FnDecl<'tcx>,
        _body: &'tcx hir::Body<'tcx>,
        _span: rustc_span::Span,
        def_id: rustc_span::def_id::LocalDefId
    ) {
        // Only run for libraries
        if cx.tcx.crate_types().iter().any(|&t| matches!(t, CrateType::Executable | CrateType::ProcMacro)) {
            return;
        }

        if !cx.effective_visibilities.is_exported(def_id) {
            return;
        }
        if let hir::intravisit::FnKind::Closure = kind {
            return;
        }

        let ranges = [
            hir::LangItem::Range,
            hir::LangItem::RangeFrom,
            hir::LangItem::RangeInclusiveStruct,
        ].map(|x| cx.tcx.lang_items().get(x).unwrap());

        // Recursively search for instances of `Range` in the type
        fn search_range<'hir, 'tcx>(cx: &LateContext<'tcx>, ranges: &[DefId; 3], hir_ty: hir::Ty<'hir>, ty: Ty<'tcx>) -> Result<(), rustc_span::Span> {
            match (hir_ty.kind, ty.kind()) {
                (hir::TyKind::Path(hir::QPath::Resolved(_, path)), ty::Adt(adt_def, args)) => {
                    // Ignore `param: Self`, which is covered by `trait_impl_range`
                    match path.res {
                        hir::def::Res::Def(_, _) |
                        hir::def::Res::Local(_) => (),
                        // SelfTyParam, SelfTyAlias, etc
                        _ => return Ok(()),
                    }
                    
                    if ranges.contains(&adt_def.did()) {
                        return Err(hir_ty.span);
                    }

                    let hir_args = path.segments.last().unwrap().args().args;

                    if let Some(span) = iter::zip(hir_args, args.iter()).find_map(|(hir_arg, arg)| {
                        if let (hir::GenericArg::Type(&hir_ty), Some(ty)) = (hir_arg, arg.as_type()) {
                            search_range(cx, ranges, hir_ty, ty).err()
                        } else {
                            None
                        }
                    }) {
                        Err(span)
                    } else {
                        Ok(())
                    }
                }
                (hir::TyKind::Array(hir_ty, _), ty::Array(ty, _)) |
                (hir::TyKind::Slice(hir_ty), ty::Slice(ty)) |
                (hir::TyKind::Ptr(hir::MutTy { ty: hir_ty, .. }), ty::RawPtr(ty::TypeAndMut { ty, .. })) |
                (hir::TyKind::Ref(_, hir::MutTy { ty: hir_ty, .. }), ty::Ref(_, ty, _)) => search_range(cx, ranges, *hir_ty, *ty),
                // ty::FnPtr(sig) => sig.skip_binder().inputs_and_output.iter().any(|ty| search_range(cx, ranges, ty)),
                // ty::Dynamic(_, _, _) => todo!(),
                (hir::TyKind::Tup(hir_tys), ty::Tuple(tys)) => {
                    if let Some(span) = iter::zip(hir_tys, tys.iter()).find_map(|(hir_ty, ty)| search_range(cx, ranges, *hir_ty, ty).err()) {
                        Err(span)
                    } else {
                        Ok(())
                    }
                }
                // ty::Alias(_, alias) => search_range(cx, ranges, alias.to_ty(cx.tcx)),
                // ty::Param(param) => param.,

                _ => Ok(()),
            }
        }

        let sig: Binder<'_, FnSig<'_>> = cx.tcx.fn_sig(def_id).instantiate_identity();
        
        let mut inputs = iter::zip(decl.inputs, sig.skip_binder().inputs());
        // Skip implicit `self` arg
        if decl.implicit_self.has_implicit_self() {
            inputs.next();
        }

        for (h, ty) in inputs {
            if let Err(span) = search_range(cx, &ranges, *h, *ty) {
                cx.emit_span_lint(EXPLICIT_RANGE, span, ExplicitRangeDiag)
            }
        }
    }
}

declare_lint! {
    /// The `trait_impl_range` lint detects trait impls involving `Range`, 
    /// `RangeInclusive`, or `RangeFrom` if the trait are used in public APIs.
    /// 
    /// ### Examples
    /// 
    /// ```rust
    /// pub trait Bar
    /// impl Bar for Range<usize>
    /// 
    /// pub struct Thing
    /// impl Index<Range<usize>> for Thing
    /// ```
    /// 
    /// ### Explanation
    /// 
    /// Gather data for [concerns in RFC 3550].
    /// 
    /// [concerns in RFC 3550](https://github.com/rust-lang/rfcs/pull/3550#issuecomment-1935112286)
    pub(super) TRAIT_IMPL_RANGE,
    Allow,
    "public trait impl involving range type"
}

declare_lint_pass!(TraitImplRange => [TRAIT_IMPL_RANGE]);

impl<'tcx> LateLintPass<'tcx> for TraitImplRange {
    fn check_item(
        &mut self,
        cx: &LateContext<'tcx>,
        item: &'tcx hir::Item<'tcx>
    ) {
        // Only run for libraries
        if cx.tcx.crate_types().iter().any(|&t| matches!(t, CrateType::Executable | CrateType::ProcMacro)) {
            return;
        }

        let ranges = [
            hir::LangItem::Range,
            hir::LangItem::RangeFrom,
            hir::LangItem::RangeInclusiveStruct,
        ].map(|x| cx.tcx.lang_items().get(x).unwrap());

        // Recursively search for instances of `Range` in the type
        fn search_range<'hir, 'tcx>(cx: &LateContext<'tcx>, ranges: &[DefId; 3], hir_ty: hir::Ty<'hir>) -> Result<(), rustc_span::Span> {
            match hir_ty.kind {
                hir::TyKind::Path(qpath) => {
                    match cx.qpath_res(&qpath, hir_ty.hir_id) {
                        hir::def::Res::Def(_, def_id) => {
                            if let Some(local_id) = def_id.as_local()
                            && !cx.effective_visibilities.is_exported(local_id) {
                                return Ok(());
                            }

                            if ranges.contains(&def_id) {
                                return Err(hir_ty.span);
                            }
                        }
                        // hir::def::Res::PrimTy(_) => todo!(),
                        // hir::def::Res::SelfTyParam { trait_ } => todo!(),
                        // hir::def::Res::SelfTyAlias { alias_to, forbid_generic, is_trait_impl } => todo!(),
                        // hir::def::Res::SelfCtor(_) => todo!(),
                        // hir::def::Res::Local(_) => todo!(),
                        // hir::def::Res::ToolMod => todo!(),
                        // hir::def::Res::NonMacroAttr(_) => todo!(),
                        // hir::def::Res::Err => todo!(),
                        _ => (),
                    }

                    let hir::QPath::Resolved(_, path) = qpath else { return Ok(()); };

                    for segment in path.segments {
                        for hir_arg in segment.args().args {
                            if let hir::GenericArg::Type(&hir_ty) = hir_arg {
                                search_range(cx, ranges, hir_ty)?;
                            }
                        }
                    }

                    Ok(())
                }
                hir::TyKind::Array(hir_ty, _) |
                hir::TyKind::Slice(hir_ty) |
                hir::TyKind::Ptr(hir::MutTy { ty: hir_ty, .. }) |
                hir::TyKind::Ref(_, hir::MutTy { ty: hir_ty, .. }) => search_range(cx, ranges, *hir_ty),
                // ty::FnPtr(sig) => sig.skip_binder().inputs_and_output.iter().any(|ty| search_range(cx, ranges, ty)),
                // ty::Dynamic(_, _, _) => todo!(),
                hir::TyKind::Tup(hir_tys) => {
                    if let Some(span) = hir_tys.iter().find_map(|hir_ty| search_range(cx, ranges, *hir_ty).err()) {
                        Err(span)
                    } else {
                        Ok(())
                    }
                }
                // ty::Alias(_, alias) => search_range(cx, ranges, alias.to_ty(cx.tcx)),
                // ty::Param(param) => param.,

                _ => Ok(()),
            }
        }

        match item.kind {
            hir::ItemKind::Static(ty, _, _) |
            hir::ItemKind::Const(ty, _, _) |
            hir::ItemKind::TyAlias(ty, _) => {
                if !cx.effective_visibilities.is_exported(item.owner_id.def_id) {
                    return;
                }

                if let Err(span) = search_range(cx, &ranges, *ty) {
                    cx.emit_span_lint(TRAIT_IMPL_RANGE, span, TraitImplRangeDiag)
                }
            },
            // hir::ItemKind::Enum(_, _) => todo!(),
            // hir::ItemKind::Struct(_, _) => todo!(),
            // hir::ItemKind::Union(_, _) => todo!(),
            // hir::ItemKind::Trait(_, _, _, _, _) => todo!(),
            hir::ItemKind::Impl(imp) => {
                let Some(of_trait) = imp.of_trait else { return };
                let Some(trait_def_id) = of_trait.trait_def_id() else { return };
                
                if let Some(trait_local_id) = trait_def_id.as_local() &&
                    !cx.effective_visibilities.is_exported(trait_local_id)
                {
                    // Trait is local but not public
                    return;
                }

                // Skip if private self type
                if let hir::TyKind::Path(qpath) = imp.self_ty.kind
                && let hir::def::Res::Def(_, def_id) = cx.qpath_res(&qpath, imp.self_ty.hir_id)
                && let Some(local_id) = def_id.as_local()
                && !cx.effective_visibilities.is_exported(local_id) {
                    return;
                }

                if let Err(span) = search_range(cx, &ranges, *imp.self_ty) {
                    cx.emit_span_lint(TRAIT_IMPL_RANGE, span, TraitImplRangeDiag);
                }

                for segment in of_trait.path.segments {
                    for hir_arg in segment.args().args {
                        if let hir::GenericArg::Type(&hir_ty) = hir_arg {
                            if let Err(span) = search_range(cx, &ranges, hir_ty) {
                                cx.emit_span_lint(TRAIT_IMPL_RANGE, span, TraitImplRangeDiag);
                            }
                        }
                    }
                }
            },
            
            // hir::ItemKind::ForeignMod { abi, items } => todo!(),
            // hir::ItemKind::ExternCrate(_) => todo!(),
            // hir::ItemKind::Use(_, _) => todo!(),
            // hir::ItemKind::Fn(_, _, _) => todo!(),
            // hir::ItemKind::Macro(_, _) => todo!(),
            // hir::ItemKind::Mod(_) => todo!(),
            // hir::ItemKind::GlobalAsm(_) => todo!(),
            // hir::ItemKind::OpaqueTy(_) => todo!(),
            // hir::ItemKind::TraitAlias(_, _) => todo!(),
            _ => (),
        }
    }
}

declare_lint! {
    /// The `range_syntax` lint detects uses of `a..b`, `a..=b`, or `a..` syntax.
    /// 
    /// ### Examples
    /// 
    /// ```rust
    /// 1..5
    /// 
    /// 0..=255
    /// 
    /// 1..
    /// ```
    /// 
    /// ### Explanation
    /// 
    /// Gather data for [concerns in RFC 3550].
    /// 
    /// [concerns in RFC 3550](https://github.com/rust-lang/rfcs/pull/3550#issuecomment-1935112286)
    pub(super) RANGE_SYNTAX,
    Deny,
    "usage of range syntax"
}

declare_lint! {
    /// The `range_bounds` lint detects uses of the `RangeBounds` trait in public APIs.
    /// This includes in generic bounds or `impl` parameters.
    /// 
    /// ### Examples
    /// 
    /// ```rust
    /// pub fn takes_range_1(range: impl RangeBounds<usize>)
    /// 
    /// pub fn takes_range_2<I, R>(range: R) where R: RangeBounds<I>
    /// 
    /// impl<R: RangeBounds<usize>> Index<R> for Foo
    /// ```
    /// 
    /// ### Explanation
    /// 
    /// Gather data for [concerns in RFC 3550].
    /// 
    /// [concerns in RFC 3550](https://github.com/rust-lang/rfcs/pull/3550#issuecomment-1935112286)
    pub(super) RANGE_BOUNDS,
    Deny,
    "usage of `RangeBounds` trait"
}
