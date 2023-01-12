//! Detect autodiff blocks
//!
//! External blocks can be decorated with an autodiff attribute, generating functions
//! during compilation with automatic differentiation from a source function.
//!
//! # Example
//!
//! ```rust
//! fn rosenbrock(a: f32, b: f32, x: f32, y: f32) -> f32 {
//!     let (z, w) = (a-x, y-x*x);
//!
//!     z*z + b*w*w
//! }
//!
//! #[autodiff(rosenbrock, mode = "forward")]
//! extern "C" {
//!     fn dx_rosenbrock(a: f32, b: f32, x: f32, y: f32, d_x: &mut f32);
//!     fn dy_rosenbrock(a: f32, b: f32, x: f32, y: f32, d_y: &mut f32);
//! }
//!

//use rustc_ast::{AttrKind};
use rustc_hir::def_id::LocalDefId;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_hir::{ForeignItem, ForeignItemKind, HirId, ImplItem, Item, ItemKind, TraitItem};
//use rustc_span::symbol::Ident;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_ast::{Lit, LitKind, MetaItem, MetaItemKind, NestedMetaItem, Path};
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::metadata::{DiffItems, DiffMode, DiffActivity, DiffItem};
use rustc_span::{symbol::{sym, Ident}, Span};
use rustc_target::spec::abi::Abi;

#[allow(dead_code)]
struct AutodiffContext<'tcx> {
    tcx: TyCtxt<'tcx>,
    methods: Vec<((Path, LocalDefId), DiffMode, &'tcx ForeignItem<'tcx>, Span)>,
    source_candidates: FxHashMap<(Ident, LocalDefId), HirId>,
}

impl<'tcx> ItemLikeVisitor<'tcx> for AutodiffContext<'tcx> {
    fn visit_item(&mut self, item: &'tcx Item<'tcx>) {
        if let ItemKind::Fn(sig, _, _) = &item.kind {
            if matches!(sig.header.abi, Abi::C { .. }) {
                self.source_candidates.insert(
                    (item.ident, self.tcx.hir().get_parent_item(item.hir_id())),
                    item.hir_id(),
                );
            }

            return;
        }

        // skip all items except extern blocks
        let ItemKind::ForeignMod { items, .. } = item.kind else {
            return
        };

        let id = item.hir_id();
        let attrs = self.tcx.hir().attrs(id);
        let attrs = attrs
            .into_iter()
            .filter(|attr| attr.name_or_empty() == sym::autodiff)
            .collect::<Vec<_>>();

        // check for exactly one autodiff attribute on extern block
        let attr = match &attrs[..] {
            &[] => return,
            &[elm] => elm,
            _ => {
                self.tcx
                    .sess
                    .struct_span_err(item.span, "autodiff attribute can only be applied once")
                    .span_label(item.span, "more than one")
                    .emit();

                return;
            }
        };

        let list = attr.meta_item_list().unwrap_or_default();
        let (path, mode) = match &list[..] {
            &[NestedMetaItem::MetaItem(MetaItem { ref path, kind: MetaItemKind::Word, .. })] => {
                (path, None)
            }
            &[
                NestedMetaItem::MetaItem(MetaItem { ref path, kind: MetaItemKind::Word, .. }),
                NestedMetaItem::MetaItem(MetaItem {
                    path: ref p2,
                    kind: MetaItemKind::NameValue(Lit { kind: LitKind::Str(mode, _), .. }),
                    ..
                }),
            ] if *p2 == sym::mode => (path, Some(mode)),
            _ => {
                self.tcx
                    .sess
                    .struct_span_err(
                        item.span,
                        "autodiff attribute must contain the source function",
                    )
                    .span_label(item.span, "empty argument list")
                    .emit();

                return;
            }
        };

        // parse mode
        let mode = match mode.as_ref().map(|x| x.as_str()) {
            Some("forward") => DiffMode::Forward,
            Some("reverse") => DiffMode::Reverse,
            Some(_) => {
                self.tcx
                    .sess
                    .struct_span_err(item.span, "mode should be either forward or reverse")
                    .span_label(item.span, "invalid mode")
                    .emit();

                return;
            }
            None => DiffMode::Forward,
        };

        let parent_id = self.tcx.hir().get_parent_item(HirId::make_owner(item.def_id));
        self.methods.extend(
            items.into_iter()
                .map(|x| self.tcx.hir().foreign_item(x.id))
                .map(|x| ((path.clone(), parent_id), mode.clone(), x, item.span))
        );

        //dbg!(&parent_id);
        //self.tcx.hir().visit_item_likes_in_module(parent_id, self);
        
        //visit_path(

        //visit_path
        //maybe_resolve_path
    }

    fn visit_trait_item(&mut self, _trait_item: &'tcx TraitItem<'tcx>) {
        // Entry fn is never a trait item.
    }

    fn visit_impl_item(&mut self, _impl_item: &'tcx ImplItem<'tcx>) {
        // Entry fn is never a trait item.
    }

    fn visit_foreign_item(&mut self, _item: &'tcx ForeignItem<'tcx>) {}
}

#[allow(dead_code)]
fn get_autodiff_functions(tcx: TyCtxt<'_>, (): ()) -> DiffItems {
    let mut ctxt =
        AutodiffContext { tcx, methods: Vec::new(), source_candidates: FxHashMap::default() };

    tcx.hir().visit_all_item_likes(&mut ctxt);

    let mut elms = Vec::new();
    for method in ctxt.methods {
        // get source function's def_id
        let ident = method.0.0.segments.last().unwrap().ident;
        let Some(source) = ctxt.source_candidates.get(&(ident, method.0.1)) else {
            tcx
                .sess
                .struct_span_err(
                    method.3,
                    "method not found for autodiff attribute",
                )
                .span_label(method.0.0.span, "unknown method")
                .emit();

            return DiffItems::default();
        };

        //let source = tcx.hir().fn_decl_by_hir_id(*source).unwrap();

        let def_id = tcx.hir().local_def_id(*source).to_def_id();
        let source_args = tcx.fn_arg_names(def_id);

        let fn_args = match method.2.kind {
            ForeignItemKind::Fn(_, args, _) => args,
            _ => unreachable!(),
        };

        // skip beginning of function parameters, make sure they are the same
        if &fn_args[..source_args.len()] != source_args {
            tcx
                .sess
                .struct_span_err(
                    method.0.0.span,
                    "method has to begin with same arguments",
                )
                .span_label(method.0.0.span, "arguments differ")
                .emit();

            return DiffItems::default();
        }

        // create activity from remaining parameters
        let mut activity = vec![DiffActivity::Const; source_args.len()];
        for a in &fn_args[source_args.len()..] {
            let Some(pos) = source_args.iter().position(|x| a.as_str().ends_with(x.as_str())) else {
                tcx
                    .sess
                    .struct_span_err(
                        method.0.0.span,
                        "argument not found",
                    )
                    .span_label(method.0.0.span, "argument not found")
                    .emit();

                return DiffItems::default();
            };

            activity[pos] = DiffActivity::Active;
        }

        elms.push(DiffItem {
            source: def_id,
            target: method.2.ident,
            mode: method.1,
            respect_to: activity
        });
    }

    elms
}

pub fn provide(providers: &mut Providers) {
    providers.autodiff_functions = get_autodiff_functions;
}
