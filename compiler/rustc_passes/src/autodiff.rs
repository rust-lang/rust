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

use rustc_hir::def_id::LocalDefId;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_hir::{ForeignItem, ForeignItemKind, HirId, ImplItem, Item, ItemKind, TraitItem};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_ast::{Lit, LitKind, MetaItem, MetaItemKind, NestedMetaItem, Path};
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::metadata::{DiffItems, DiffMode, DiffActivity, DiffItem};
use rustc_span::{symbol::{sym, Ident}, Span};
use rustc_target::spec::abi::Abi;

#[derive(Clone)]
struct AdTask {
    mode: DiffMode,
    ret: DiffActivity,
    inputs: Vec<DiffActivity>,
}

#[allow(dead_code)]
struct AutodiffContext<'tcx> {
    tcx: TyCtxt<'tcx>,
    methods: Vec<((Path, LocalDefId), AdTask, &'tcx ForeignItem<'tcx>, Span)>,
    source_candidates: FxHashMap<(Ident, LocalDefId), HirId>,
}

impl<'tcx> ItemLikeVisitor<'tcx> for AutodiffContext<'tcx> {
    fn visit_item(&mut self, item: &'tcx Item<'tcx>) {
        if let ItemKind::Fn(sig, _, _) = &item.kind {
            if !matches!(sig.header.abi, Abi::C { .. }) {
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
        //assert!(list.len() > 3);
        let tmp_ret  = &list[2];
        //let tmp_args = &list[3..];

        dbg!(tmp_ret);

        let ret_symbol = match tmp_ret {
            NestedMetaItem::MetaItem(MetaItem {
                path: ref p2,
                kind: MetaItemKind::Word,
                ..
            }) => {
                let id = p2.segments.first().unwrap().ident;
                dbg!(&id);
                id
            },
            _ => {
                self.tcx
                    .sess
                    .struct_span_err(
                        item.span,
                        "autodiff attribute must contain the return activity",
                        )
                    .span_label(item.span, "missing return activity")
                    .emit();

                return;
            },
        };


        let (path, mode) = match &list[0..=1] {
            &[
                NestedMetaItem::MetaItem(MetaItem { ref path, kind: MetaItemKind::Word, .. }),
                NestedMetaItem::MetaItem(MetaItem {
                    path: ref p2,
                    kind: MetaItemKind::NameValue(Lit { kind: LitKind::Str(mode, _), .. }),
                    ..
                }),
            ] if *p2 == sym::mode => (path, mode),
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

        let ret_activity = match ret_symbol.as_str() {
            "None" => DiffActivity::None,
            "Active" => DiffActivity::Active,
            "Const" => DiffActivity::Const,
            "Duplicated" => DiffActivity::Duplicated,
            "DuplicatedNoNeed" => DiffActivity::DuplicatedNoNeed,
            _ => {
                self.tcx
                    .sess
                    .struct_span_err(item.span, "unknown return activity")
                    .span_label(item.span, "invalid return activity")
                    .emit();
                return;
            }
        };

        let mut input_activity: Vec<DiffActivity> = vec![];
        for foo in &list[3..] {
            let foo_symbol = match foo {
                NestedMetaItem::MetaItem(MetaItem {
                    path: ref p2,
                    kind: MetaItemKind::Word,
                    ..
                }) => {
                    let id = p2.segments.first().unwrap().ident;
                    dbg!(&id);
                    id
                },
                _ => {
                    self.tcx
                        .sess
                        .struct_span_err(
                            item.span,
                            "autodiff attribute must contain the return activity",
                            )
                        .span_label(item.span, "missing return activity")
                        .emit();

                    return;
                },
            };
            let act = match foo_symbol.as_str() {
                "None" => DiffActivity::None,
                "Active" => DiffActivity::Active,
                "Const" => DiffActivity::Const,
                "Duplicated" => DiffActivity::Duplicated,
                "DuplicatedNoNeed" => DiffActivity::DuplicatedNoNeed,
                _ => {
                    self.tcx
                        .sess
                        .struct_span_err(item.span, "unknown return activity")
                        .span_label(item.span, "invalid input activity")
                        .emit();
                    return;
                }
            };
            input_activity.push(act);
        }

        // parse mode
        let mode = match mode.as_str() {//map(|x| x.as_str()) {
            "forward" => DiffMode::Forward,
            "reverse" => DiffMode::Reverse,
            _ => {
                self.tcx
                    .sess
                    .struct_span_err(item.span, "mode should be either forward or reverse")
                    .span_label(item.span, "invalid mode")
                    .emit();

                return;
            }
        };

        if mode == DiffMode::Forward {
            if ret_activity == DiffActivity::Active {
                self.tcx
                    .sess
                    .struct_span_err(item.span, "Forward Mode is incompatible with Active ret")
                    .span_label(item.span, "invalid return activity")
                    .emit();
                return;
            }
            if input_activity.iter().filter(|&x| *x == DiffActivity::Active).count() > 0 {
                self.tcx
                    .sess
                    .struct_span_err(item.span, "Forward Mode is incompatible with Active args")
                    .span_label(item.span, "invalid input activity")
                    .emit();
                return;
            }
        }

        if mode == DiffMode::Reverse {
            if ret_activity == DiffActivity::Duplicated || ret_activity == DiffActivity::DuplicatedNoNeed {
                self.tcx
                    .sess
                    .struct_span_err(item.span, "Reverse Mode is only compatible with Active, None, or Const ret")
                    .span_label(item.span, "invalid return activity")
                    .emit();
                return;
            }
        }

        let task = AdTask {
            mode,
            ret: ret_activity,
            inputs: input_activity,
        };


        let parent_id = self.tcx.hir().get_parent_item(HirId::make_owner(item.def_id));
        self.methods.extend(
            items.into_iter()
            .map(|x| self.tcx.hir().foreign_item(x.id))
            .map(|x| ((path.clone(), parent_id), task.clone(), x, item.span))
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

            let _fn_args = match method.2.kind {
                ForeignItemKind::Fn(_, args, _) => args,
                _ => unreachable!(),
            };

            // skip beginning of function parameters, make sure they are the same
            //  if &fn_args[..source_args.len()] != source_args {
            //      tcx
            //          .sess
            //          .struct_span_err(
            //              method.0.0.span,
            //              "method has to begin with same arguments",
            //              )
            //          .span_label(method.0.0.span, "arguments differ")
            //          .emit();

            //      return DiffItems::default();
            //  }
            assert!(method.1.inputs.len() == source_args.len());

            elms.push(DiffItem {
                source: def_id,
                target: method.2.ident.to_string(),
                mode: method.1.mode,
                ret_activity: method.1.ret,
                input_activity: method.1.inputs,
            });
        }

        elms
    }

    pub fn provide(providers: &mut Providers) {
        providers.autodiff_functions = get_autodiff_functions;
    }
