// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Verifies that the types and values of const and static items
// are safe. The rules enforced by this module are:
//
// - For each *mutable* static item, it checks that its **type**:
//     - doesn't have a destructor
//     - doesn't own a box
//
// - For each *immutable* static item, it checks that its **value**:
//       - doesn't own a box
//       - doesn't contain a struct literal or a call to an enum variant / struct constructor where
//           - the type of the struct/enum has a dtor
//
// Rules Enforced Elsewhere:
// - It's not possible to take the address of a static item with unsafe interior. This is enforced
// by borrowck::gather_loans

use rustc::ty::cast::CastKind;
use rustc::hir::def::{Def, CtorKind};
use rustc::hir::def_id::DefId;
use rustc::hir::map::blocks::FnLikeNode;
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::mem_categorization as mc;
use rustc::middle::mem_categorization::Categorization;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::query::Providers;
use rustc::ty::subst::Substs;
use rustc::util::nodemap::{ItemLocalSet, NodeSet};
use rustc::hir;
use rustc_data_structures::sync::Lrc;
use syntax::ast;
use syntax::attr;
use syntax_pos::{Span, DUMMY_SP};
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        rvalue_promotable_map,
        const_is_rvalue_promotable_to_static,
        ..*providers
    };
}

pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    for &body_id in &tcx.hir.krate().body_ids {
        let def_id = tcx.hir.body_owner_def_id(body_id);
        tcx.const_is_rvalue_promotable_to_static(def_id);
    }
    tcx.sess.abort_if_errors();
}

fn const_is_rvalue_promotable_to_static<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                                  def_id: DefId)
                                                  -> bool
{
    assert!(def_id.is_local());

    let node_id = tcx.hir.as_local_node_id(def_id)
                     .expect("rvalue_promotable_map invoked with non-local def-id");
    let body_id = tcx.hir.body_owned_by(node_id);
    let body_hir_id = tcx.hir.node_to_hir_id(body_id.node_id);
    tcx.rvalue_promotable_map(def_id).contains(&body_hir_id.local_id)
}

fn rvalue_promotable_map<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                   def_id: DefId)
                                   -> Lrc<ItemLocalSet>
{
    let outer_def_id = tcx.closure_base_def_id(def_id);
    if outer_def_id != def_id {
        return tcx.rvalue_promotable_map(outer_def_id);
    }

    let mut visitor = CheckCrateVisitor {
        tcx,
        tables: &ty::TypeckTables::empty(None),
        in_fn: false,
        in_static: false,
        promotable: false,
        mut_rvalue_borrows: NodeSet(),
        param_env: ty::ParamEnv::empty(),
        identity_substs: Substs::empty(),
        result: ItemLocalSet(),
    };

    // `def_id` should be a `Body` owner
    let node_id = tcx.hir.as_local_node_id(def_id)
                     .expect("rvalue_promotable_map invoked with non-local def-id");
    let body_id = tcx.hir.body_owned_by(node_id);
    visitor.visit_nested_body(body_id);

    Lrc::new(visitor.result)
}

struct CheckCrateVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    in_fn: bool,
    in_static: bool,
    promotable: bool,
    mut_rvalue_borrows: NodeSet,
    param_env: ty::ParamEnv<'tcx>,
    identity_substs: &'tcx Substs<'tcx>,
    tables: &'a ty::TypeckTables<'tcx>,
    result: ItemLocalSet,
}

impl<'a, 'gcx> CheckCrateVisitor<'a, 'gcx> {
    // Returns true iff all the values of the type are promotable.
    fn type_has_only_promotable_values(&mut self, ty: Ty<'gcx>) -> bool {
        ty.is_freeze(self.tcx, self.param_env, DUMMY_SP) &&
        !ty.needs_drop(self.tcx, self.param_env)
    }

    fn handle_const_fn_call(&mut self, def_id: DefId, ret_ty: Ty<'gcx>, span: Span) {
        self.promotable &= self.type_has_only_promotable_values(ret_ty);

        self.promotable &= if let Some(fn_id) = self.tcx.hir.as_local_node_id(def_id) {
            FnLikeNode::from_node(self.tcx.hir.get(fn_id)).map_or(false, |fn_like| {
                fn_like.constness() == hir::Constness::Const
            })
        } else {
            self.tcx.is_const_fn(def_id)
        };

        if let Some(&attr::Stability {
            rustc_const_unstable: Some(attr::RustcConstUnstable {
                feature: ref feature_name
            }),
        .. }) = self.tcx.lookup_stability(def_id) {
            self.promotable &=
                // feature-gate is enabled,
                self.tcx.features()
                    .declared_lib_features
                    .iter()
                    .any(|&(ref sym, _)| sym == feature_name) ||

                // this comes from a crate with the feature-gate enabled,
                !def_id.is_local() ||

                // this comes from a macro that has #[allow_internal_unstable]
                span.allows_unstable();
        }
    }

    /// While the `ExprUseVisitor` walks, we will identify which
    /// expressions are borrowed, and insert their ids into this
    /// table. Actually, we insert the "borrow-id", which is normally
    /// the id of the expession being borrowed: but in the case of
    /// `ref mut` borrows, the `id` of the pattern is
    /// inserted. Therefore later we remove that entry from the table
    /// and transfer it over to the value being matched. This will
    /// then prevent said value from being promoted.
    fn remove_mut_rvalue_borrow(&mut self, pat: &hir::Pat) -> bool {
        let mut any_removed = false;
        pat.walk(|p| {
            any_removed |= self.mut_rvalue_borrows.remove(&p.id);
            true
        });
        any_removed
    }
}

impl<'a, 'tcx> Visitor<'tcx> for CheckCrateVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        // note that we *do* visit nested bodies, because we override `visit_nested_body` below
        NestedVisitorMap::None
    }

    fn visit_nested_body(&mut self, body_id: hir::BodyId) {
        let item_id = self.tcx.hir.body_owner(body_id);
        let item_def_id = self.tcx.hir.local_def_id(item_id);

        let outer_in_fn = self.in_fn;
        let outer_tables = self.tables;
        let outer_param_env = self.param_env;
        let outer_identity_substs = self.identity_substs;

        self.in_fn = false;
        self.in_static = false;

        match self.tcx.hir.body_owner_kind(item_id) {
            hir::BodyOwnerKind::Fn => self.in_fn = true,
            hir::BodyOwnerKind::Static(_) => self.in_static = true,
            _ => {}
        };


        self.tables = self.tcx.typeck_tables_of(item_def_id);
        self.param_env = self.tcx.param_env(item_def_id);
        self.identity_substs = Substs::identity_for_item(self.tcx, item_def_id);

        let body = self.tcx.hir.body(body_id);

        let tcx = self.tcx;
        let param_env = self.param_env;
        let region_scope_tree = self.tcx.region_scope_tree(item_def_id);
        euv::ExprUseVisitor::new(self, tcx, param_env, &region_scope_tree, self.tables, None)
            .consume_body(body);

        self.visit_body(body);

        self.in_fn = outer_in_fn;
        self.tables = outer_tables;
        self.param_env = outer_param_env;
        self.identity_substs = outer_identity_substs;
    }

    fn visit_stmt(&mut self, stmt: &'tcx hir::Stmt) {
        match stmt.node {
            hir::StmtDecl(ref decl, _) => {
                match &decl.node {
                    hir::DeclLocal(local) => {
                        self.promotable = false;

                        if self.remove_mut_rvalue_borrow(&local.pat) {
                            if let Some(init) = &local.init {
                                self.mut_rvalue_borrows.insert(init.id);
                            }
                        }
                    }
                    // Item statements are allowed
                    hir::DeclItem(_) => {}
                }
            }
            hir::StmtExpr(..) |
            hir::StmtSemi(..) => {
                self.promotable = false;
            }
        }
        intravisit::walk_stmt(self, stmt);
    }

    fn visit_expr(&mut self, ex: &'tcx hir::Expr) {
        let outer = self.promotable;
        self.promotable = true;

        let node_ty = self.tables.node_id_to_type(ex.hir_id);
        check_expr(self, ex, node_ty);
        check_adjustments(self, ex);

        if let hir::ExprMatch(ref discr, ref arms, _) = ex.node {
            // Compute the most demanding borrow from all the arms'
            // patterns and set that on the discriminator.
            let mut mut_borrow = false;
            for pat in arms.iter().flat_map(|arm| &arm.pats) {
                mut_borrow = self.remove_mut_rvalue_borrow(pat);
            }
            if mut_borrow {
                self.mut_rvalue_borrows.insert(discr.id);
            }
        }

        intravisit::walk_expr(self, ex);

        // Handle borrows on (or inside the autorefs of) this expression.
        if self.mut_rvalue_borrows.remove(&ex.id) {
            self.promotable = false;
        }

        if self.promotable {
            self.result.insert(ex.hir_id.local_id);
        }
        self.promotable &= outer;
    }
}

/// This function is used to enforce the constraints on
/// const/static items. It walks through the *value*
/// of the item walking down the expression and evaluating
/// every nested expression. If the expression is not part
/// of a const/static item, it is qualified for promotion
/// instead of producing errors.
fn check_expr<'a, 'tcx>(v: &mut CheckCrateVisitor<'a, 'tcx>, e: &hir::Expr, node_ty: Ty<'tcx>) {
    match node_ty.sty {
        ty::TyAdt(def, _) if def.has_dtor(v.tcx) => {
            v.promotable = false;
        }
        _ => {}
    }

    match e.node {
        hir::ExprUnary(..) |
        hir::ExprBinary(..) |
        hir::ExprIndex(..) if v.tables.is_method_call(e) => {
            v.promotable = false;
        }
        hir::ExprBox(_) => {
            v.promotable = false;
        }
        hir::ExprUnary(op, _) => {
            if op == hir::UnDeref {
                v.promotable = false;
            }
        }
        hir::ExprBinary(op, ref lhs, _) => {
            match v.tables.node_id_to_type(lhs.hir_id).sty {
                ty::TyRawPtr(_) => {
                    assert!(op.node == hir::BiEq || op.node == hir::BiNe ||
                            op.node == hir::BiLe || op.node == hir::BiLt ||
                            op.node == hir::BiGe || op.node == hir::BiGt);

                    v.promotable = false;
                }
                _ => {}
            }
        }
        hir::ExprCast(ref from, _) => {
            debug!("Checking const cast(id={})", from.id);
            match v.tables.cast_kinds().get(from.hir_id) {
                None => v.tcx.sess.delay_span_bug(e.span, "no kind for cast"),
                Some(&CastKind::PtrAddrCast) | Some(&CastKind::FnPtrAddrCast) => {
                    v.promotable = false;
                }
                _ => {}
            }
        }
        hir::ExprPath(ref qpath) => {
            let def = v.tables.qpath_def(qpath, e.hir_id);
            match def {
                Def::VariantCtor(..) | Def::StructCtor(..) |
                Def::Fn(..) | Def::Method(..) =>  {}

                // References to a static that are themselves within a static
                // are inherently promotable with the exception
                //  of "#[thread_local]" statics, which may not
                // outlive the current function
                Def::Static(did, _) => {

                    if v.in_static {
                        let mut thread_local = false;

                        for attr in &v.tcx.get_attrs(did)[..] {
                            if attr.check_name("thread_local") {
                                debug!("Reference to Static(id={:?}) is unpromotable \
                                       due to a #[thread_local] attribute", did);
                                v.promotable = false;
                                thread_local = true;
                                break;
                            }
                        }

                        if !thread_local {
                            debug!("Allowing promotion of reference to Static(id={:?})", did);
                        }
                    } else {
                        debug!("Reference to Static(id={:?}) is unpromotable as it is not \
                               referenced from a static", did);
                        v.promotable = false;

                    }
                }

                Def::Const(did) |
                Def::AssociatedConst(did) => {
                    let promotable = if v.tcx.trait_of_item(did).is_some() {
                        // Don't peek inside trait associated constants.
                        false
                    } else {
                        v.tcx.at(e.span).const_is_rvalue_promotable_to_static(did)
                    };

                    // Just in case the type is more specific than the definition,
                    // e.g. impl associated const with type parameters, check it.
                    // Also, trait associated consts are relaxed by this.
                    v.promotable &= promotable || v.type_has_only_promotable_values(node_ty);
                }

                _ => {
                    v.promotable = false;
                }
            }
        }
        hir::ExprCall(ref callee, _) => {
            let mut callee = &**callee;
            loop {
                callee = match callee.node {
                    hir::ExprBlock(ref block, _) => match block.expr {
                        Some(ref tail) => &tail,
                        None => break
                    },
                    _ => break
                };
            }
            // The callee is an arbitrary expression, it doesn't necessarily have a definition.
            let def = if let hir::ExprPath(ref qpath) = callee.node {
                v.tables.qpath_def(qpath, callee.hir_id)
            } else {
                Def::Err
            };
            match def {
                Def::StructCtor(_, CtorKind::Fn) |
                Def::VariantCtor(_, CtorKind::Fn) => {}
                Def::Fn(did) => {
                    v.handle_const_fn_call(did, node_ty, e.span)
                }
                Def::Method(did) => {
                    match v.tcx.associated_item(did).container {
                        ty::ImplContainer(_) => {
                            v.handle_const_fn_call(did, node_ty, e.span)
                        }
                        ty::TraitContainer(_) => v.promotable = false
                    }
                }
                _ => v.promotable = false
            }
        }
        hir::ExprMethodCall(..) => {
            if let Some(def) = v.tables.type_dependent_defs().get(e.hir_id) {
                let def_id = def.def_id();
                match v.tcx.associated_item(def_id).container {
                    ty::ImplContainer(_) => v.handle_const_fn_call(def_id, node_ty, e.span),
                    ty::TraitContainer(_) => v.promotable = false
                }
            } else {
                v.tcx.sess.delay_span_bug(e.span, "no type-dependent def for method call");
            }
        }
        hir::ExprStruct(..) => {
            if let ty::TyAdt(adt, ..) = v.tables.expr_ty(e).sty {
                // unsafe_cell_type doesn't necessarily exist with no_core
                if Some(adt.did) == v.tcx.lang_items().unsafe_cell_type() {
                    v.promotable = false;
                }
            }
        }

        hir::ExprLit(_) |
        hir::ExprAddrOf(..) |
        hir::ExprRepeat(..) => {}

        hir::ExprClosure(..) => {
            // Paths in constant contexts cannot refer to local variables,
            // as there are none, and thus closures can't have upvars there.
            if v.tcx.with_freevars(e.id, |fv| !fv.is_empty()) {
                v.promotable = false;
            }
        }

        hir::ExprField(ref expr, _) => {
            if let Some(def) = v.tables.expr_ty(expr).ty_adt_def() {
                if def.is_union() {
                    v.promotable = false
                }
            }
        }

        hir::ExprBlock(..) |
        hir::ExprIndex(..) |
        hir::ExprArray(_) |
        hir::ExprType(..) |
        hir::ExprTup(..) => {}

        // Conditional control flow (possible to implement).
        hir::ExprMatch(..) |
        hir::ExprIf(..) |

        // Loops (not very meaningful in constants).
        hir::ExprWhile(..) |
        hir::ExprLoop(..) |

        // More control flow (also not very meaningful).
        hir::ExprBreak(..) |
        hir::ExprContinue(_) |
        hir::ExprRet(_) |

        // Generator expressions
        hir::ExprYield(_) |

        // Expressions with side-effects.
        hir::ExprAssign(..) |
        hir::ExprAssignOp(..) |
        hir::ExprInlineAsm(..) => {
            v.promotable = false;
        }
    }
}

/// Check the adjustments of an expression
fn check_adjustments<'a, 'tcx>(v: &mut CheckCrateVisitor<'a, 'tcx>, e: &hir::Expr) {
    use rustc::ty::adjustment::*;

    let mut adjustments = v.tables.expr_adjustments(e).iter().peekable();
    while let Some(adjustment) = adjustments.next() {
        match adjustment.kind {
            Adjust::NeverToAny |
            Adjust::ReifyFnPointer |
            Adjust::UnsafeFnPointer |
            Adjust::ClosureFnPointer |
            Adjust::MutToConstPointer |
            Adjust::Borrow(_) |
            Adjust::Unsize => {}

            Adjust::Deref(_) => {
                if let Some(next_adjustment) = adjustments.peek() {
                    if let Adjust::Borrow(_) = next_adjustment.kind {
                        continue;
                    }
                }
                v.promotable = false;
                break;
            }
        }
    }
}

impl<'a, 'gcx, 'tcx> euv::Delegate<'tcx> for CheckCrateVisitor<'a, 'gcx> {
    fn consume(&mut self,
               _consume_id: ast::NodeId,
               _consume_span: Span,
               _cmt: &mc::cmt_,
               _mode: euv::ConsumeMode) {}

    fn borrow(&mut self,
              borrow_id: ast::NodeId,
              _borrow_span: Span,
              cmt: &mc::cmt_<'tcx>,
              _loan_region: ty::Region<'tcx>,
              bk: ty::BorrowKind,
              loan_cause: euv::LoanCause) {
        debug!(
            "borrow(borrow_id={:?}, cmt={:?}, bk={:?}, loan_cause={:?})",
            borrow_id,
            cmt,
            bk,
            loan_cause,
        );

        // Kind of hacky, but we allow Unsafe coercions in constants.
        // These occur when we convert a &T or *T to a *U, as well as
        // when making a thin pointer (e.g., `*T`) into a fat pointer
        // (e.g., `*Trait`).
        match loan_cause {
            euv::LoanCause::AutoUnsafe => {
                return;
            }
            _ => {}
        }

        let mut cur = cmt;
        loop {
            match cur.cat {
                Categorization::Rvalue(..) => {
                    if loan_cause == euv::MatchDiscriminant {
                        // Ignore the dummy immutable borrow created by EUV.
                        break;
                    }
                    if bk.to_mutbl_lossy() == hir::MutMutable {
                        self.mut_rvalue_borrows.insert(borrow_id);
                    }
                    break;
                }
                Categorization::StaticItem => {
                    break;
                }
                Categorization::Deref(ref cmt, _) |
                Categorization::Downcast(ref cmt, _) |
                Categorization::Interior(ref cmt, _) => {
                    cur = cmt;
                }

                Categorization::Upvar(..) |
                Categorization::Local(..) => break,
            }
        }
    }

    fn decl_without_init(&mut self, _id: ast::NodeId, _span: Span) {}
    fn mutate(&mut self,
              _assignment_id: ast::NodeId,
              _assignment_span: Span,
              _assignee_cmt: &mc::cmt_,
              _mode: euv::MutateMode) {
    }

    fn matched_pat(&mut self, _: &hir::Pat, _: &mc::cmt_, _: euv::MatchMode) {}

    fn consume_pat(&mut self, _consume_pat: &hir::Pat, _cmt: &mc::cmt_, _mode: euv::ConsumeMode) {}
}
