use rustc::hir::*;
use rustc::hir::map::*;
use rustc::hir::intravisit::FnKind;
use rustc::lint::*;
use rustc::ty::{self, RegionKind, TypeFoldable};
use rustc::traits;
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::mem_categorization as mc;
use rustc_target::spec::abi::Abi;
use syntax::ast::NodeId;
use syntax_pos::Span;
use syntax::errors::DiagnosticBuilder;
use utils::{get_trait_def_id, implements_trait, in_macro, is_copy, is_self, match_type, multispan_sugg, paths,
            snippet, snippet_opt, span_lint_and_then};
use utils::ptr::get_spans;
use std::collections::{HashMap, HashSet};
use std::borrow::Cow;

/// **What it does:** Checks for functions taking arguments by value, but not
/// consuming them in its
/// body.
///
/// **Why is this bad?** Taking arguments by reference is more flexible and can
/// sometimes avoid
/// unnecessary allocations.
///
/// **Known problems:**
/// * This lint suggests taking an argument by reference,
/// however sometimes it is better to let users decide the argument type
/// (by using `Borrow` trait, for example), depending on how the function is used.
///
/// **Example:**
/// ```rust
/// fn foo(v: Vec<i32>) {
///     assert_eq!(v.len(), 42);
/// }
/// // should be
/// fn foo(v: &[i32]) {
///     assert_eq!(v.len(), 42);
/// }
/// ```
declare_clippy_lint! {
    pub NEEDLESS_PASS_BY_VALUE,
    style,
    "functions taking arguments by value, but not consuming them in its body"
}

pub struct NeedlessPassByValue;

impl LintPass for NeedlessPassByValue {
    fn get_lints(&self) -> LintArray {
        lint_array![NEEDLESS_PASS_BY_VALUE]
    }
}

macro_rules! need {
    ($e: expr) => { if let Some(x) = $e { x } else { return; } };
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NeedlessPassByValue {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl,
        body: &'tcx Body,
        span: Span,
        node_id: NodeId,
    ) {
        if in_macro(span) {
            return;
        }

        match kind {
            FnKind::ItemFn(.., abi, _, attrs) => {
                if abi != Abi::Rust {
                    return;
                }
                for a in attrs {
                    if_chain! {
                        if a.meta_item_list().is_some();
                        if let Some(name) = a.name();
                        if name == "proc_macro_derive";
                        then {
                            return;
                        }
                    }
                }
            },
            FnKind::Method(..) => (),
            _ => return,
        }

        // Exclude non-inherent impls
        if let Some(NodeItem(item)) = cx.tcx.hir.find(cx.tcx.hir.get_parent_node(node_id)) {
            if matches!(item.node, ItemImpl(_, _, _, _, Some(_), _, _) |
                ItemTrait(..))
            {
                return;
            }
        }

        // Allow `Borrow` or functions to be taken by value
        let borrow_trait = need!(get_trait_def_id(cx, &paths::BORROW_TRAIT));
        let whitelisted_traits = [
            need!(cx.tcx.lang_items().fn_trait()),
            need!(cx.tcx.lang_items().fn_once_trait()),
            need!(cx.tcx.lang_items().fn_mut_trait()),
            need!(get_trait_def_id(cx, &paths::RANGE_ARGUMENT_TRAIT))
        ];

        let sized_trait = need!(cx.tcx.lang_items().sized_trait());

        let fn_def_id = cx.tcx.hir.local_def_id(node_id);

        let preds = traits::elaborate_predicates(cx.tcx, cx.param_env.caller_bounds.to_vec())
            .filter(|p| !p.is_global())
            .filter_map(|pred| {
                if let ty::Predicate::Trait(poly_trait_ref) = pred {
                    if poly_trait_ref.def_id() == sized_trait || poly_trait_ref.skip_binder().has_escaping_regions() {
                        return None;
                    }
                    Some(poly_trait_ref)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        // Collect moved variables and spans which will need dereferencings from the
        // function body.
        let MovedVariablesCtxt {
            moved_vars,
            spans_need_deref,
            ..
        } = {
            let mut ctx = MovedVariablesCtxt::new(cx);
            let region_scope_tree = &cx.tcx.region_scope_tree(fn_def_id);
            euv::ExprUseVisitor::new(&mut ctx, cx.tcx, cx.param_env, region_scope_tree, cx.tables, None)
                .consume_body(body);
            ctx
        };

        let fn_sig = cx.tcx.fn_sig(fn_def_id);
        let fn_sig = cx.tcx.erase_late_bound_regions(&fn_sig);

        for (idx, ((input, &ty), arg)) in decl.inputs
            .iter()
            .zip(fn_sig.inputs())
            .zip(&body.arguments)
            .enumerate()
        {
            // All spans generated from a proc-macro invocation are the same...
            if span == input.span {
                return;
            }

            // Ignore `self`s.
            if idx == 0 {
                if let PatKind::Binding(_, _, name, ..) = arg.pat.node {
                    if name.node.as_str() == "self" {
                        continue;
                    }
                }
            }

            // * Exclude a type that is specifically bounded by `Borrow`.
            // * Exclude a type whose reference also fulfills its bound.
            //   (e.g. `std::convert::AsRef`, `serde::Serialize`)
            let (implements_borrow_trait, all_borrowable_trait) = {
                let preds = preds
                    .iter()
                    .filter(|t| t.skip_binder().self_ty() == ty)
                    .collect::<Vec<_>>();

                (
                    preds.iter().any(|t| t.def_id() == borrow_trait),
                    !preds.is_empty() && preds.iter().all(|t| {
                        implements_trait(
                            cx,
                            cx.tcx.mk_imm_ref(&RegionKind::ReErased, ty),
                            t.def_id(),
                            &t.skip_binder().input_types().skip(1).collect::<Vec<_>>(),
                        )
                    }),
                )
            };

            if_chain! {
                if !is_self(arg);
                if !ty.is_mutable_pointer();
                if !is_copy(cx, ty);
                if !whitelisted_traits.iter().any(|&t| implements_trait(cx, ty, t, &[]));
                if !implements_borrow_trait;
                if !all_borrowable_trait;

                if let PatKind::Binding(mode, canonical_id, ..) = arg.pat.node;
                if !moved_vars.contains(&canonical_id);
                then {
                    if mode == BindingAnnotation::Mutable || mode == BindingAnnotation::RefMut {
                        continue;
                    }

                    // Dereference suggestion
                    let sugg = |db: &mut DiagnosticBuilder| {
                        if let ty::TypeVariants::TyAdt(def, ..) = ty.sty {
                            if let Some(span) = cx.tcx.hir.span_if_local(def.did) {
                                let param_env = ty::ParamEnv::empty();
                                if param_env.can_type_implement_copy(cx.tcx, ty, span).is_ok() {
                                    db.span_help(span, "consider marking this type as Copy");
                                }
                            }
                        }

                        let deref_span = spans_need_deref.get(&canonical_id);
                        if_chain! {
                            if match_type(cx, ty, &paths::VEC);
                            if let Some(clone_spans) =
                                get_spans(cx, Some(body.id()), idx, &[("clone", ".to_owned()")]);
                            if let TyPath(QPath::Resolved(_, ref path)) = input.node;
                            if let Some(elem_ty) = path.segments.iter()
                                .find(|seg| seg.name == "Vec")
                                .and_then(|ps| ps.parameters.as_ref())
                                .map(|params| &params.types[0]);
                            then {
                                let slice_ty = format!("&[{}]", snippet(cx, elem_ty.span, "_"));
                                db.span_suggestion(input.span,
                                                "consider changing the type to",
                                                slice_ty);

                                for (span, suggestion) in clone_spans {
                                    db.span_suggestion(
                                        span,
                                        &snippet_opt(cx, span)
                                            .map_or(
                                                "change the call to".into(),
                                                |x| Cow::from(format!("change `{}` to", x)),
                                            ),
                                        suggestion.into()
                                    );
                                }

                                // cannot be destructured, no need for `*` suggestion
                                assert!(deref_span.is_none());
                                return;
                            }
                        }

                        if match_type(cx, ty, &paths::STRING) {
                            if let Some(clone_spans) =
                                get_spans(cx, Some(body.id()), idx, &[("clone", ".to_string()"), ("as_str", "")]) {
                                db.span_suggestion(input.span, "consider changing the type to", "&str".to_string());

                                for (span, suggestion) in clone_spans {
                                    db.span_suggestion(
                                        span,
                                        &snippet_opt(cx, span)
                                            .map_or(
                                                "change the call to".into(),
                                                |x| Cow::from(format!("change `{}` to", x))
                                            ),
                                        suggestion.into(),
                                    );
                                }

                                assert!(deref_span.is_none());
                                return;
                            }
                        }

                        let mut spans = vec![(input.span, format!("&{}", snippet(cx, input.span, "_")))];

                        // Suggests adding `*` to dereference the added reference.
                        if let Some(deref_span) = deref_span {
                            spans.extend(
                                deref_span
                                    .iter()
                                    .cloned()
                                    .map(|span| (span, format!("*{}", snippet(cx, span, "<expr>")))),
                            );
                            spans.sort_by_key(|&(span, _)| span);
                        }
                        multispan_sugg(db, "consider taking a reference instead".to_string(), spans);
                    };

                    span_lint_and_then(
                        cx,
                        NEEDLESS_PASS_BY_VALUE,
                        input.span,
                        "this argument is passed by value, but not consumed in the function body",
                        sugg,
                    );
                }
            }
        }
    }
}

struct MovedVariablesCtxt<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>,
    moved_vars: HashSet<NodeId>,
    /// Spans which need to be prefixed with `*` for dereferencing the
    /// suggested additional reference.
    spans_need_deref: HashMap<NodeId, HashSet<Span>>,
}

impl<'a, 'tcx> MovedVariablesCtxt<'a, 'tcx> {
    fn new(cx: &'a LateContext<'a, 'tcx>) -> Self {
        Self {
            cx,
            moved_vars: HashSet::new(),
            spans_need_deref: HashMap::new(),
        }
    }

    fn move_common(&mut self, _consume_id: NodeId, _span: Span, cmt: mc::cmt<'tcx>) {
        let cmt = unwrap_downcast_or_interior(cmt);

        if let mc::Categorization::Local(vid) = cmt.cat {
            self.moved_vars.insert(vid);
        }
    }

    fn non_moving_pat(&mut self, matched_pat: &Pat, cmt: mc::cmt<'tcx>) {
        let cmt = unwrap_downcast_or_interior(cmt);

        if let mc::Categorization::Local(vid) = cmt.cat {
            let mut id = matched_pat.id;
            loop {
                let parent = self.cx.tcx.hir.get_parent_node(id);
                if id == parent {
                    // no parent
                    return;
                }
                id = parent;

                if let Some(node) = self.cx.tcx.hir.find(id) {
                    match node {
                        map::Node::NodeExpr(e) => {
                            // `match` and `if let`
                            if let ExprMatch(ref c, ..) = e.node {
                                self.spans_need_deref
                                    .entry(vid)
                                    .or_insert_with(HashSet::new)
                                    .insert(c.span);
                            }
                        },

                        map::Node::NodeStmt(s) => {
                            // `let <pat> = x;`
                            if_chain! {
                                if let StmtDecl(ref decl, _) = s.node;
                                if let DeclLocal(ref local) = decl.node;
                                then {
                                    self.spans_need_deref
                                        .entry(vid)
                                        .or_insert_with(HashSet::new)
                                        .insert(local.init
                                            .as_ref()
                                            .map(|e| e.span)
                                            .expect("`let` stmt without init aren't caught by match_pat"));
                                }
                            }
                        },

                        _ => {},
                    }
                }
            }
        }
    }
}

impl<'a, 'tcx> euv::Delegate<'tcx> for MovedVariablesCtxt<'a, 'tcx> {
    fn consume(&mut self, consume_id: NodeId, consume_span: Span, cmt: mc::cmt<'tcx>, mode: euv::ConsumeMode) {
        if let euv::ConsumeMode::Move(_) = mode {
            self.move_common(consume_id, consume_span, cmt);
        }
    }

    fn matched_pat(&mut self, matched_pat: &Pat, cmt: mc::cmt<'tcx>, mode: euv::MatchMode) {
        if let euv::MatchMode::MovingMatch = mode {
            self.move_common(matched_pat.id, matched_pat.span, cmt);
        } else {
            self.non_moving_pat(matched_pat, cmt);
        }
    }

    fn consume_pat(&mut self, consume_pat: &Pat, cmt: mc::cmt<'tcx>, mode: euv::ConsumeMode) {
        if let euv::ConsumeMode::Move(_) = mode {
            self.move_common(consume_pat.id, consume_pat.span, cmt);
        }
    }

    fn borrow(&mut self, _: NodeId, _: Span, _: mc::cmt<'tcx>, _: ty::Region, _: ty::BorrowKind, _: euv::LoanCause) {}

    fn mutate(&mut self, _: NodeId, _: Span, _: mc::cmt<'tcx>, _: euv::MutateMode) {}

    fn decl_without_init(&mut self, _: NodeId, _: Span) {}
}


fn unwrap_downcast_or_interior(mut cmt: mc::cmt) -> mc::cmt {
    loop {
        match cmt.cat.clone() {
            mc::Categorization::Downcast(c, _) | mc::Categorization::Interior(c, _) => {
                cmt = c;
            },
            _ => return cmt,
        }
    }
}
