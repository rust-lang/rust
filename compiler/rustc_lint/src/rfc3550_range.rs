use crate::lints::{ExplicitRangeDiag, RangeBoundsDiag, RangeSyntaxDiag, TraitImplRangeDiag};
use crate::{EarlyContext, EarlyLintPass, LateContext, LateLintPass, LintContext};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_session::config::CrateType;
use rustc_span::Symbol;

fn crate_is_library<'tcx>(cx: &LateContext<'tcx>) -> bool {
    for t in cx.tcx.crate_types() {
        if matches!(t, CrateType::Executable | CrateType::ProcMacro) {
            return false;
        }
    }

    true
}

struct TyVisitor<'cx, 'tcx> {
    cx: &'cx LateContext<'tcx>,
    ranges: [DefId; 3],
}

impl<'cx, 'tcx> TyVisitor<'cx, 'tcx> {
    fn new(cx: &'cx LateContext<'tcx>) -> Self {
        let ranges = [
            hir::LangItem::Range,
            hir::LangItem::RangeFrom,
            hir::LangItem::RangeInclusiveStruct,
        ].map(|x| cx.tcx.lang_items().get(x).unwrap());

        Self {
            cx,
            ranges,
        }
    }

    // Recursively search for instances of `Range` in the type
    fn check_ty(&self, hir_ty: hir::Ty<'_>) -> Result<(), rustc_span::Span> {
        match hir_ty.kind {
            hir::TyKind::Path(qpath) => {
                match self.cx.qpath_res(&qpath, hir_ty.hir_id) {
                    hir::def::Res::Def(_, def_id) => {
                        if let Some(local_id) = def_id.as_local()
                        && !self.cx.effective_visibilities.is_exported(local_id) {
                            return Ok(());
                        }

                        if self.ranges.contains(&def_id) {
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
                            self.check_ty(hir_ty)?;
                        }
                    }
                }

                Ok(())
            }
            hir::TyKind::Array(hir_ty, _) |
            hir::TyKind::Slice(hir_ty) |
            hir::TyKind::Ptr(hir::MutTy { ty: hir_ty, .. }) |
            hir::TyKind::Ref(_, hir::MutTy { ty: hir_ty, .. }) => self.check_ty(*hir_ty),
            // ty::FnPtr(sig) => sig.skip_binder().inputs_and_output.iter().any(|ty| check_ty(cx, ranges, ty)),
            // ty::Dynamic(_, _, _) => todo!(),
            hir::TyKind::Tup(hir_tys) => {
                if let Some(span) = hir_tys.iter().find_map(|hir_ty| self.check_ty(*hir_ty).err()) {
                    Err(span)
                } else {
                    Ok(())
                }
            }
            // ty::Alias(_, alias) => check_ty(cx, ranges, alias.to_ty(cx.tcx)),
            // ty::Param(param) => param.,

            _ => Ok(()),
        }
    }
}

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
        if !crate_is_library(cx) {
            return;
        }

        if !cx.effective_visibilities.is_exported(def_id) {
            return;
        }
        if let hir::intravisit::FnKind::Closure = kind {
            return;
        }

        let visitor = TyVisitor::new(cx);
        
        let mut inputs = decl.inputs.iter();
        // Skip implicit `self` arg
        if decl.implicit_self.has_implicit_self() {
            inputs.next();
        }

        for h in inputs {
            if let Err(span) = visitor.check_ty(*h) {
                cx.emit_span_lint(EXPLICIT_RANGE, span, ExplicitRangeDiag)
            }
        }
    }


    fn check_item(
        &mut self,
        cx: &LateContext<'tcx>,
        item: &'tcx hir::Item<'tcx>
    ) {
        // Only run for libraries
        if !crate_is_library(cx) {
            return;
        }
        if !cx.effective_visibilities.is_exported(item.owner_id.def_id) {
            return;
        }

        let visitor = TyVisitor::new(cx);

        match item.kind {
            hir::ItemKind::Static(ty, _, _) |
            hir::ItemKind::Const(ty, _, _) |
            hir::ItemKind::TyAlias(ty, _) => {
                if let Err(span) = visitor.check_ty(*ty) {
                    cx.emit_span_lint(EXPLICIT_RANGE, span, ExplicitRangeDiag)
                }
            },
            hir::ItemKind::Enum(enu, _) => {
                for v in enu.variants {
                    for field in v.data.fields() {
                        if !cx.effective_visibilities.is_exported(field.def_id) {
                            continue;
                        }

                        if let Err(span) = visitor.check_ty(*field.ty) {
                            cx.emit_span_lint(EXPLICIT_RANGE, span, ExplicitRangeDiag)
                        }
                    }
                }
            },
            hir::ItemKind::Struct(data, _) |
            hir::ItemKind::Union(data, _) => {
                for field in data.fields() {
                    if !cx.effective_visibilities.is_exported(field.def_id) {
                        continue;
                    }

                    if let Err(span) = visitor.check_ty(*field.ty) {
                        cx.emit_span_lint(EXPLICIT_RANGE, span, ExplicitRangeDiag)
                    }
                }
            },

            _ => (),
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
        if !crate_is_library(cx) {
            return;
        }

        let visitor = TyVisitor::new(cx);

        match item.kind {
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

                if let Err(span) = visitor.check_ty(*imp.self_ty) {
                    cx.emit_span_lint(TRAIT_IMPL_RANGE, span, TraitImplRangeDiag);
                }

                for segment in of_trait.path.segments {
                    for hir_arg in segment.args().args {
                        if let hir::GenericArg::Type(&hir_ty) = hir_arg {
                            if let Err(span) = visitor.check_ty(hir_ty) {
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
    Allow,
    "usage of range syntax"
}

declare_lint_pass!(RangeSyntax => [RANGE_SYNTAX]);

impl EarlyLintPass for RangeSyntax {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &rustc_ast::Expr) {
        match expr.kind {
            // Range, RangeInclusive - a..b, a..=b
            rustc_ast::ExprKind::Range(Some(_), Some(_), _) |
            // RangeFrom - a..
            rustc_ast::ExprKind::Range(Some(_), None, _) => {
                cx.emit_span_lint(RANGE_SYNTAX, expr.span, RangeSyntaxDiag);
            }
            _ => (),
        }
    }
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
    Allow,
    "usage of `RangeBounds` trait"
}

declare_lint_pass!(RangeBounds => [RANGE_BOUNDS]);

struct BoundsVisitor<'cx, 'tcx> {
    cx: &'cx LateContext<'tcx>,
    range_bounds: [Symbol; 4],
}

impl<'cx, 'tcx> BoundsVisitor<'cx, 'tcx> {
    fn new(cx: &'cx LateContext<'tcx>) -> Self {
        let range_bounds = ["core", "ops", "range", "RangeBounds"].map(|x| Symbol::intern(x));

        Self {
            cx,
            range_bounds,
        }
    }

    fn is_range_bounds(&self, def_id: DefId) -> bool {
        self.cx.match_def_path(def_id, &self.range_bounds)
    }

    fn check_bounds(&self, bounds: hir::GenericBounds<'_>) {
        for bound in bounds {
            let hir::GenericBound::Trait(of_trait, _) = bound else { continue };

            if let Some(def_id) = of_trait.trait_ref.trait_def_id()
            && self.is_range_bounds(def_id) {
                self.cx.emit_span_lint(RANGE_BOUNDS, of_trait.span, RangeBoundsDiag);
            }
        }
    }

    fn check_generics(&self, generics: &'tcx hir::Generics<'_>) {
        for pred in generics.predicates {
            let hir::WherePredicate::BoundPredicate(pred) = pred else { continue }; 

            self.check_bounds(pred.bounds);
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for RangeBounds {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: hir::intravisit::FnKind<'tcx>,
        _decl: &'tcx hir::FnDecl<'tcx>,
        _body: &'tcx hir::Body<'tcx>,
        _span: rustc_span::Span,
        def_id: rustc_span::def_id::LocalDefId
    ) {
        // Only run for libraries
        if !crate_is_library(cx) {
            return;
        }

        if !cx.effective_visibilities.is_exported(def_id) {
            return;
        }
        if let hir::intravisit::FnKind::Closure = kind {
            return;
        }
        
        let Some(generics) = cx.generics else { return };

        let visitor = BoundsVisitor::new(cx);
        visitor.check_generics(generics);
    }

    fn check_item(
        &mut self,
        cx: &LateContext<'tcx>,
        item: &'tcx hir::Item<'tcx>
    ) {
        // Only run for libraries
        if !crate_is_library(cx) {
            return;
        }

        if !cx.effective_visibilities.is_exported(item.owner_id.def_id) {
            return;
        }
        let visitor = BoundsVisitor::new(cx);

        match item.kind {
            hir::ItemKind::Enum(_, generics) |
            hir::ItemKind::Struct(_, generics) |
            hir::ItemKind::Union(_, generics) => {
                visitor.check_generics(generics);
            }
            hir::ItemKind::Trait(_, _, generics, bounds, _) => {
                visitor.check_generics(generics);
                visitor.check_bounds(bounds);
            },
            hir::ItemKind::Impl(imp) => {
                visitor.check_generics(imp.generics);
            },

            _ => (),
        }
    }
}
