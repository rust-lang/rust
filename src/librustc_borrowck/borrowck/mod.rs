//! See The Book chapter on the borrow checker for more details.

#![allow(non_camel_case_types)]

pub use LoanPathKind::*;
pub use LoanPathElem::*;
pub use bckerr_code::*;
pub use AliasableViolationKind::*;
pub use MovedValueUseKind::*;

use InteriorKind::*;

use rustc::hir::HirId;
use rustc::hir::Node;
use rustc::hir::map::blocks::FnLikeNode;
use rustc::cfg;
use rustc::middle::borrowck::{BorrowCheckResult, SignalledError};
use rustc::hir::def_id::{DefId, LocalDefId};
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::mem_categorization as mc;
use rustc::middle::mem_categorization::Categorization;
use rustc::middle::mem_categorization::ImmutabilityBlame;
use rustc::middle::region;
use rustc::middle::free_region::RegionRelations;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::query::Providers;
use rustc_mir::util::borrowck_errors::{BorrowckErrors, Origin};
use rustc_mir::util::suggest_ref_mut;
use rustc::util::nodemap::FxHashSet;

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::fmt;
use std::rc::Rc;
use std::hash::{Hash, Hasher};
use syntax::source_map::CompilerDesugaringKind;
use syntax_pos::{MultiSpan, Span};
use errors::{Applicability, DiagnosticBuilder, DiagnosticId};
use log::debug;

use rustc::hir;

use crate::dataflow::{DataFlowContext, BitwiseOperator, DataFlowOperator, KillFrom};

pub mod check_loans;

pub mod gather_loans;

pub mod move_data;

#[derive(Clone, Copy)]
pub struct LoanDataFlowOperator;

pub type LoanDataFlow<'tcx> = DataFlowContext<'tcx, LoanDataFlowOperator>;

pub fn check_crate(tcx: TyCtxt<'_>) {
    tcx.par_body_owners(|body_owner_def_id| {
        tcx.ensure().borrowck(body_owner_def_id);
    });
}

pub fn provide(providers: &mut Providers<'_>) {
    *providers = Providers {
        borrowck,
        ..*providers
    };
}

/// Collection of conclusions determined via borrow checker analyses.
pub struct AnalysisData<'tcx> {
    pub all_loans: Vec<Loan<'tcx>>,
    pub loans: DataFlowContext<'tcx, LoanDataFlowOperator>,
    pub move_data: move_data::FlowedMoveData<'tcx>,
}

fn borrowck(tcx: TyCtxt<'_>, owner_def_id: DefId) -> &BorrowCheckResult {
    assert!(tcx.use_ast_borrowck() || tcx.migrate_borrowck());

    debug!("borrowck(body_owner_def_id={:?})", owner_def_id);

    let owner_id = tcx.hir().as_local_hir_id(owner_def_id).unwrap();

    match tcx.hir().get(owner_id) {
        Node::Ctor(..) => {
            // We get invoked with anything that has MIR, but some of
            // those things (notably the synthesized constructors from
            // tuple structs/variants) do not have an associated body
            // and do not need borrowchecking.
            return tcx.arena.alloc(BorrowCheckResult {
                used_mut_nodes: Default::default(),
                signalled_any_error: SignalledError::NoErrorsSeen,
            })
        }
        _ => { }
    }

    let body_id = tcx.hir().body_owned_by(owner_id);
    let tables = tcx.typeck_tables_of(owner_def_id);
    let region_scope_tree = tcx.region_scope_tree(owner_def_id);
    let body = tcx.hir().body(body_id);
    let mut bccx = BorrowckCtxt {
        tcx,
        tables,
        region_scope_tree,
        owner_def_id,
        body,
        used_mut_nodes: Default::default(),
        signalled_any_error: Cell::new(SignalledError::NoErrorsSeen),
    };

    // Eventually, borrowck will always read the MIR, but at the
    // moment we do not. So, for now, we always force MIR to be
    // constructed for a given fn, since this may result in errors
    // being reported and we want that to happen.
    //
    // Note that `mir_validated` is a "stealable" result; the
    // thief, `optimized_mir()`, forces borrowck, so we know that
    // is not yet stolen.
    tcx.ensure().mir_validated(owner_def_id);

    // option dance because you can't capture an uninitialized variable
    // by mut-ref.
    let mut cfg = None;
    if let Some(AnalysisData { all_loans,
                               loans: loan_dfcx,
                               move_data: flowed_moves }) =
        build_borrowck_dataflow_data(&mut bccx, false, body_id,
                                     |bccx| {
                                         cfg = Some(cfg::CFG::new(bccx.tcx, &body));
                                         cfg.as_mut().unwrap()
                                     })
    {
        check_loans::check_loans(&mut bccx, &loan_dfcx, &flowed_moves, &all_loans, body);
    }

    tcx.arena.alloc(BorrowCheckResult {
        used_mut_nodes: bccx.used_mut_nodes.into_inner(),
        signalled_any_error: bccx.signalled_any_error.into_inner(),
    })
}

fn build_borrowck_dataflow_data<'a, 'c, 'tcx, F>(
    this: &mut BorrowckCtxt<'a, 'tcx>,
    force_analysis: bool,
    body_id: hir::BodyId,
    get_cfg: F,
) -> Option<AnalysisData<'tcx>>
where
    F: FnOnce(&mut BorrowckCtxt<'a, 'tcx>) -> &'c cfg::CFG,
{
    // Check the body of fn items.
    let (all_loans, move_data) =
        gather_loans::gather_loans_in_fn(this, body_id);

    if !force_analysis && move_data.is_empty() && all_loans.is_empty() {
        // large arrays of data inserted as constants can take a lot of
        // time and memory to borrow-check - see issue #36799. However,
        // they don't have places, so no borrow-check is actually needed.
        // Recognize that case and skip borrow-checking.
        debug!("skipping loan propagation for {:?} because of no loans", body_id);
        return None;
    } else {
        debug!("propagating loans in {:?}", body_id);
    }

    let cfg = get_cfg(this);
    let mut loan_dfcx =
        DataFlowContext::new(this.tcx,
                             "borrowck",
                             Some(this.body),
                             cfg,
                             LoanDataFlowOperator,
                             all_loans.len());
    for (loan_idx, loan) in all_loans.iter().enumerate() {
        loan_dfcx.add_gen(loan.gen_scope.item_local_id(), loan_idx);
        loan_dfcx.add_kill(KillFrom::ScopeEnd,
                           loan.kill_scope.item_local_id(),
                           loan_idx);
    }
    loan_dfcx.add_kills_from_flow_exits(cfg);
    loan_dfcx.propagate(cfg, this.body);

    let flowed_moves = move_data::FlowedMoveData::new(move_data,
                                                      this,
                                                      cfg,
                                                      this.body);

    Some(AnalysisData { all_loans,
                        loans: loan_dfcx,
                        move_data:flowed_moves })
}

/// Accessor for introspective clients inspecting `AnalysisData` and
/// the `BorrowckCtxt` itself , e.g., the flowgraph visualizer.
pub fn build_borrowck_dataflow_data_for_fn<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    body_id: hir::BodyId,
    cfg: &cfg::CFG,
) -> (BorrowckCtxt<'a, 'tcx>, AnalysisData<'tcx>) {
    let owner_id = tcx.hir().body_owner(body_id);
    let owner_def_id = tcx.hir().local_def_id_from_hir_id(owner_id);
    let tables = tcx.typeck_tables_of(owner_def_id);
    let region_scope_tree = tcx.region_scope_tree(owner_def_id);
    let body = tcx.hir().body(body_id);
    let mut bccx = BorrowckCtxt {
        tcx,
        tables,
        region_scope_tree,
        owner_def_id,
        body,
        used_mut_nodes: Default::default(),
        signalled_any_error: Cell::new(SignalledError::NoErrorsSeen),
    };

    let dataflow_data = build_borrowck_dataflow_data(&mut bccx, true, body_id, |_| cfg);
    (bccx, dataflow_data.unwrap())
}

// ----------------------------------------------------------------------
// Type definitions

pub struct BorrowckCtxt<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,

    // tables for the current thing we are checking; set to
    // Some in `borrowck_fn` and cleared later
    tables: &'a ty::TypeckTables<'tcx>,

    region_scope_tree: &'tcx region::ScopeTree,

    owner_def_id: DefId,

    body: &'tcx hir::Body,

    used_mut_nodes: RefCell<FxHashSet<HirId>>,

    signalled_any_error: Cell<SignalledError>,
}

impl BorrowckCtxt<'_, 'tcx> {
    fn signal_error(&self) {
        self.signalled_any_error.set(SignalledError::SawSomeError);
    }
}

impl BorrowckErrors<'a> for &'a BorrowckCtxt<'_, 'tcx> {
    fn struct_span_err_with_code<S: Into<MultiSpan>>(self,
                                                     sp: S,
                                                     msg: &str,
                                                     code: DiagnosticId)
                                                     -> DiagnosticBuilder<'a>
    {
        self.tcx.sess.struct_span_err_with_code(sp, msg, code)
    }

    fn struct_span_err<S: Into<MultiSpan>>(self,
                                           sp: S,
                                           msg: &str)
                                           -> DiagnosticBuilder<'a>
    {
        self.tcx.sess.struct_span_err(sp, msg)
    }

    fn cancel_if_wrong_origin(self,
                              mut diag: DiagnosticBuilder<'a>,
                              o: Origin)
                              -> DiagnosticBuilder<'a>
    {
        if !o.should_emit_errors(self.tcx.borrowck_mode()) {
            self.tcx.sess.diagnostic().cancel(&mut diag);
        }
        diag
    }
}

///////////////////////////////////////////////////////////////////////////
// Loans and loan paths

/// Record of a loan that was issued.
pub struct Loan<'tcx> {
    index: usize,
    loan_path: Rc<LoanPath<'tcx>>,
    kind: ty::BorrowKind,
    restricted_paths: Vec<Rc<LoanPath<'tcx>>>,

    /// gen_scope indicates where loan is introduced. Typically the
    /// loan is introduced at the point of the borrow, but in some
    /// cases, notably method arguments, the loan may be introduced
    /// only later, once it comes into scope. See also
    /// `GatherLoanCtxt::compute_gen_scope`.
    gen_scope: region::Scope,

    /// kill_scope indicates when the loan goes out of scope. This is
    /// either when the lifetime expires or when the local variable
    /// which roots the loan-path goes out of scope, whichever happens
    /// faster. See also `GatherLoanCtxt::compute_kill_scope`.
    kill_scope: region::Scope,
    span: Span,
    cause: euv::LoanCause,
}

impl<'tcx> Loan<'tcx> {
    pub fn loan_path(&self) -> Rc<LoanPath<'tcx>> {
        self.loan_path.clone()
    }
}

#[derive(Eq)]
pub struct LoanPath<'tcx> {
    kind: LoanPathKind<'tcx>,
    ty: Ty<'tcx>,
}

impl<'tcx> PartialEq for LoanPath<'tcx> {
    fn eq(&self, that: &LoanPath<'tcx>) -> bool {
        self.kind == that.kind
    }
}

impl<'tcx> Hash for LoanPath<'tcx> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
    }
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub enum LoanPathKind<'tcx> {
    LpVar(hir::HirId),                          // `x` in README.md
    LpUpvar(ty::UpvarId),                       // `x` captured by-value into closure
    LpDowncast(Rc<LoanPath<'tcx>>, DefId), // `x` downcast to particular enum variant
    LpExtend(Rc<LoanPath<'tcx>>, mc::MutabilityCategory, LoanPathElem<'tcx>)
}

impl<'tcx> LoanPath<'tcx> {
    fn new(kind: LoanPathKind<'tcx>, ty: Ty<'tcx>) -> LoanPath<'tcx> {
        LoanPath { kind: kind, ty: ty }
    }

    fn to_type(&self) -> Ty<'tcx> { self.ty }

    fn has_downcast(&self) -> bool {
        match self.kind {
            LpDowncast(_, _) => true,
            LpExtend(ref lp, _, LpInterior(_, _)) => {
                lp.has_downcast()
            }
            _ => false,
        }
    }
}

// FIXME (pnkfelix): See discussion here
// https://github.com/pnkfelix/rust/commit/
//     b2b39e8700e37ad32b486b9a8409b50a8a53aa51#commitcomment-7892003
const DOWNCAST_PRINTED_OPERATOR: &'static str = " as ";

// A local, "cleaned" version of `mc::InteriorKind` that drops
// information that is not relevant to loan-path analysis. (In
// particular, the distinction between how precisely an array-element
// is tracked is irrelevant here.)
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum InteriorKind {
    InteriorField(mc::FieldIndex),
    InteriorElement,
}

trait ToInteriorKind { fn cleaned(self) -> InteriorKind; }
impl ToInteriorKind for mc::InteriorKind {
    fn cleaned(self) -> InteriorKind {
        match self {
            mc::InteriorField(name) => InteriorField(name),
            mc::InteriorElement(_) => InteriorElement,
        }
    }
}

// This can be:
// - a pointer dereference (`*P` in README.md)
// - a field reference, with an optional definition of the containing
//   enum variant (`P.f` in README.md)
// `DefId` is present when the field is part of struct that is in
// a variant of an enum. For instance in:
// `enum E { X { foo: u32 }, Y { foo: u32 }}`
// each `foo` is qualified by the definitition id of the variant (`X` or `Y`).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum LoanPathElem<'tcx> {
    LpDeref(mc::PointerKind<'tcx>),
    LpInterior(Option<DefId>, InteriorKind),
}

fn closure_to_block(closure_id: LocalDefId, tcx: TyCtxt<'_>) -> HirId {
    let closure_id = tcx.hir().local_def_id_to_hir_id(closure_id);
    match tcx.hir().get(closure_id) {
        Node::Expr(expr) => match expr.node {
            hir::ExprKind::Closure(.., body_id, _, _) => {
                body_id.hir_id
            }
            _ => {
                bug!("encountered non-closure id: {}", closure_id)
            }
        },
        _ => bug!("encountered non-expr id: {}", closure_id)
    }
}

impl LoanPath<'tcx> {
    pub fn kill_scope(&self, bccx: &BorrowckCtxt<'_, 'tcx>) -> region::Scope {
        match self.kind {
            LpVar(hir_id) => {
                bccx.region_scope_tree.var_scope(hir_id.local_id)
            }
            LpUpvar(upvar_id) => {
                let block_id = closure_to_block(upvar_id.closure_expr_id, bccx.tcx);
                region::Scope { id: block_id.local_id, data: region::ScopeData::Node }
            }
            LpDowncast(ref base, _) |
            LpExtend(ref base, ..) => base.kill_scope(bccx),
        }
    }

    fn has_fork(&self, other: &LoanPath<'tcx>) -> bool {
        match (&self.kind, &other.kind) {
            (&LpExtend(ref base, _, LpInterior(opt_variant_id, id)),
             &LpExtend(ref base2, _, LpInterior(opt_variant_id2, id2))) =>
                if id == id2 && opt_variant_id == opt_variant_id2 {
                    base.has_fork(&base2)
                } else {
                    true
                },
            (&LpExtend(ref base, _, LpDeref(_)), _) => base.has_fork(other),
            (_, &LpExtend(ref base, _, LpDeref(_))) => self.has_fork(&base),
            _ => false,
        }
    }

    fn depth(&self) -> usize {
        match self.kind {
            LpExtend(ref base, _, LpDeref(_)) => base.depth(),
            LpExtend(ref base, _, LpInterior(..)) => base.depth() + 1,
            _ => 0,
        }
    }

    fn common(&self, other: &LoanPath<'tcx>) -> Option<LoanPath<'tcx>> {
        match (&self.kind, &other.kind) {
            (&LpExtend(ref base, a, LpInterior(opt_variant_id, id)),
             &LpExtend(ref base2, _, LpInterior(opt_variant_id2, id2))) => {
                if id == id2 && opt_variant_id == opt_variant_id2 {
                    base.common(&base2).map(|x| {
                        let xd = x.depth();
                        if base.depth() == xd && base2.depth() == xd {
                            LoanPath {
                                kind: LpExtend(Rc::new(x), a, LpInterior(opt_variant_id, id)),
                                ty: self.ty,
                            }
                        } else {
                            x
                        }
                    })
                } else {
                    base.common(&base2)
                }
            }
            (&LpExtend(ref base, _, LpDeref(_)), _) => base.common(other),
            (_, &LpExtend(ref other, _, LpDeref(_))) => self.common(&other),
            (&LpVar(id), &LpVar(id2)) => {
                if id == id2 {
                    Some(LoanPath { kind: LpVar(id), ty: self.ty })
                } else {
                    None
                }
            }
            (&LpUpvar(id), &LpUpvar(id2)) => {
                if id == id2 {
                    Some(LoanPath { kind: LpUpvar(id), ty: self.ty })
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

// Avoid "cannot borrow immutable field `self.x` as mutable" as that implies that a field *can* be
// mutable independently of the struct it belongs to. (#35937)
pub fn opt_loan_path_is_field<'tcx>(cmt: &mc::cmt_<'tcx>) -> (Option<Rc<LoanPath<'tcx>>>, bool) {
    let new_lp = |v: LoanPathKind<'tcx>| Rc::new(LoanPath::new(v, cmt.ty));

    match cmt.cat {
        Categorization::Rvalue(..) |
        Categorization::ThreadLocal(..) |
        Categorization::StaticItem => {
            (None, false)
        }

        Categorization::Local(id) => {
            (Some(new_lp(LpVar(id))), false)
        }

        Categorization::Upvar(mc::Upvar { id, .. }) => {
            (Some(new_lp(LpUpvar(id))), false)
        }

        Categorization::Deref(ref cmt_base, pk) => {
            let lp = opt_loan_path_is_field(cmt_base);
            (lp.0.map(|lp| {
                new_lp(LpExtend(lp, cmt.mutbl, LpDeref(pk)))
            }), lp.1)
        }

        Categorization::Interior(ref cmt_base, ik) => {
            (opt_loan_path(cmt_base).map(|lp| {
                let opt_variant_id = match cmt_base.cat {
                    Categorization::Downcast(_, did) =>  Some(did),
                    _ => None
                };
                new_lp(LpExtend(lp, cmt.mutbl, LpInterior(opt_variant_id, ik.cleaned())))
            }), true)
        }

        Categorization::Downcast(ref cmt_base, variant_def_id) => {
            let lp = opt_loan_path_is_field(cmt_base);
            (lp.0.map(|lp| {
                new_lp(LpDowncast(lp, variant_def_id))
            }), lp.1)
        }
    }
}

/// Computes the `LoanPath` (if any) for a `cmt`.
/// Note that this logic is somewhat duplicated in
/// the method `compute()` found in `gather_loans::restrictions`,
/// which allows it to share common loan path pieces as it
/// traverses the CMT.
pub fn opt_loan_path<'tcx>(cmt: &mc::cmt_<'tcx>) -> Option<Rc<LoanPath<'tcx>>> {
    opt_loan_path_is_field(cmt).0
}

///////////////////////////////////////////////////////////////////////////
// Errors

// Errors that can occur
#[derive(Debug, PartialEq)]
pub enum bckerr_code<'tcx> {
    err_mutbl,
    /// superscope, subscope, loan cause
    err_out_of_scope(ty::Region<'tcx>, ty::Region<'tcx>, euv::LoanCause),
    err_borrowed_pointer_too_short(ty::Region<'tcx>, ty::Region<'tcx>), // loan, ptr
}

// Combination of an error code and the categorization of the expression
// that caused it
#[derive(Debug, PartialEq)]
pub struct BckError<'c, 'tcx> {
    span: Span,
    cause: AliasableViolationKind,
    cmt: &'c mc::cmt_<'tcx>,
    code: bckerr_code<'tcx>
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AliasableViolationKind {
    MutabilityViolation,
    BorrowViolation(euv::LoanCause)
}

#[derive(Copy, Clone, Debug)]
pub enum MovedValueUseKind {
    MovedInUse,
    MovedInCapture,
}

///////////////////////////////////////////////////////////////////////////
// Misc

impl BorrowckCtxt<'_, 'tcx> {
    pub fn is_subregion_of(&self,
                           r_sub: ty::Region<'tcx>,
                           r_sup: ty::Region<'tcx>)
                           -> bool
    {
        let region_rels = RegionRelations::new(self.tcx,
                                               self.owner_def_id,
                                               &self.region_scope_tree,
                                               &self.tables.free_region_map);
        region_rels.is_subregion_of(r_sub, r_sup)
    }

    pub fn report(&self, err: BckError<'a, 'tcx>) {
        // Catch and handle some particular cases.
        match (&err.code, &err.cause) {
            (&err_out_of_scope(&ty::ReScope(_), &ty::ReStatic, _),
             &BorrowViolation(euv::ClosureCapture(span))) |
            (&err_out_of_scope(&ty::ReScope(_), &ty::ReEarlyBound(..), _),
             &BorrowViolation(euv::ClosureCapture(span))) |
            (&err_out_of_scope(&ty::ReScope(_), &ty::ReFree(..), _),
             &BorrowViolation(euv::ClosureCapture(span))) => {
                return self.report_out_of_scope_escaping_closure_capture(&err, span);
            }
            _ => { }
        }

        self.report_bckerr(&err);
    }

    pub fn report_use_of_moved_value(&self,
                                     use_span: Span,
                                     use_kind: MovedValueUseKind,
                                     lp: &LoanPath<'tcx>,
                                     the_move: &move_data::Move,
                                     moved_lp: &LoanPath<'tcx>) {
        let (verb, verb_participle) = match use_kind {
            MovedInUse => ("use", "used"),
            MovedInCapture => ("capture", "captured"),
        };

        let (_ol, _moved_lp_msg, mut err, need_note) = match the_move.kind {
            move_data::Declared => {
                // If this is an uninitialized variable, just emit a simple warning
                // and return.
                self.cannot_act_on_uninitialized_variable(use_span,
                                                          verb,
                                                          &self.loan_path_to_string(lp),
                                                          Origin::Ast)
                    .span_label(use_span, format!("use of possibly uninitialized `{}`",
                                                  self.loan_path_to_string(lp)))
                    .emit();
                self.signal_error();
                return;
            }
            _ => {
                // If moved_lp is something like `x.a`, and lp is something like `x.b`, we would
                // normally generate a rather confusing message:
                //
                //     error: use of moved value: `x.b`
                //     note: `x.a` moved here...
                //
                // What we want to do instead is get the 'common ancestor' of the two moves and
                // use that for most of the message instead, giving is something like this:
                //
                //     error: use of moved value: `x`
                //     note: `x` moved here (through moving `x.a`)...

                let common = moved_lp.common(lp);
                let has_common = common.is_some();
                let has_fork = moved_lp.has_fork(lp);
                let (nl, ol, moved_lp_msg) =
                    if has_fork && has_common {
                        let nl = self.loan_path_to_string(&common.unwrap());
                        let ol = nl.clone();
                        let moved_lp_msg = format!(" (through moving `{}`)",
                                                   self.loan_path_to_string(moved_lp));
                        (nl, ol, moved_lp_msg)
                    } else {
                        (self.loan_path_to_string(lp),
                         self.loan_path_to_string(moved_lp),
                         String::new())
                    };

                let partial = moved_lp.depth() > lp.depth();
                let msg = if !has_fork && partial { "partially " }
                          else if has_fork && !has_common { "collaterally "}
                else { "" };
                let mut err = self.cannot_act_on_moved_value(use_span,
                                                             verb,
                                                             msg,
                                                             Some(nl),
                                                             Origin::Ast);
                let need_note = match lp.ty.sty {
                    ty::Closure(id, _) => {
                        let hir_id = self.tcx.hir().as_local_hir_id(id).unwrap();
                        if let Some((span, name)) = self.tables.closure_kind_origins().get(hir_id) {
                            err.span_note(*span, &format!(
                                "closure cannot be invoked more than once because \
                                it moves the variable `{}` out of its environment",
                                name
                            ));
                            false
                        } else {
                            true
                        }
                    }
                    _ => true,
                };
                (ol, moved_lp_msg, err, need_note)
            }
        };

        // Get type of value and span where it was previously
        // moved.
        let hir_id = hir::HirId {
            owner: self.body.value.hir_id.owner,
            local_id: the_move.id
        };
        let (move_span, move_note) = match the_move.kind {
            move_data::Declared => {
                unreachable!();
            }

            move_data::MoveExpr |
            move_data::MovePat => (self.tcx.hir().span(hir_id), ""),

            move_data::Captured =>
                (match self.tcx.hir().expect_expr(hir_id).node {
                    hir::ExprKind::Closure(.., fn_decl_span, _) => fn_decl_span,
                    ref r => bug!("Captured({:?}) maps to non-closure: {:?}",
                                  the_move.id, r),
                }, " (into closure)"),
        };

        // Annotate the use and the move in the span. Watch out for
        // the case where the use and the move are the same. This
        // means the use is in a loop.
        err = if use_span == move_span {
            err.span_label(
                use_span,
                format!("value moved{} here in previous iteration of loop",
                         move_note));
            err
        } else {
            err.span_label(use_span, format!("value {} here after move", verb_participle));
            err.span_label(move_span, format!("value moved{} here", move_note));
            err
        };

        if need_note {
            err.note(&format!(
                "move occurs because {} has type `{}`, which does not implement the `Copy` trait",
                if moved_lp.has_downcast() {
                    "the value".to_string()
                } else {
                    format!("`{}`", self.loan_path_to_string(moved_lp))
                },
                moved_lp.ty));
        }
        if let (Some(CompilerDesugaringKind::ForLoop), Ok(snippet)) = (
            move_span.compiler_desugaring_kind(),
            self.tcx.sess.source_map().span_to_snippet(move_span),
         ) {
            if !snippet.starts_with("&") {
                err.span_suggestion(
                    move_span,
                    "consider borrowing this to avoid moving it into the for loop",
                    format!("&{}", snippet),
                    Applicability::MaybeIncorrect,
                );
            }
        }

        // Note: we used to suggest adding a `ref binding` or calling
        // `clone` but those suggestions have been removed because
        // they are often not what you actually want to do, and were
        // not considered particularly helpful.

        err.emit();
        self.signal_error();
    }

    pub fn report_partial_reinitialization_of_uninitialized_structure(
            &self,
            span: Span,
            lp: &LoanPath<'tcx>) {
        self.cannot_partially_reinit_an_uninit_struct(span,
                                                      &self.loan_path_to_string(lp),
                                                      Origin::Ast)
            .emit();
        self.signal_error();
    }

    pub fn report_reassigned_immutable_variable(&self,
                                                span: Span,
                                                lp: &LoanPath<'tcx>,
                                                assign:
                                                &move_data::Assignment) {
        let mut err = self.cannot_reassign_immutable(span,
                                                     &self.loan_path_to_string(lp),
                                                     false,
                                                     Origin::Ast);
        err.span_label(span, "cannot assign twice to immutable variable");
        if span != assign.span {
            err.span_label(assign.span, format!("first assignment to `{}`",
                                                self.loan_path_to_string(lp)));
        }
        err.emit();
        self.signal_error();
    }

    fn report_bckerr(&self, err: &BckError<'a, 'tcx>) {
        let error_span = err.span.clone();

        match err.code {
            err_mutbl => {
                let descr: Cow<'static, str> = match err.cmt.note {
                    mc::NoteClosureEnv(_) | mc::NoteUpvarRef(_) => {
                        self.cmt_to_cow_str(&err.cmt)
                    }
                    _ => match opt_loan_path_is_field(&err.cmt) {
                        (None, true) => {
                            format!("{} of {} binding",
                                    self.cmt_to_cow_str(&err.cmt),
                                    err.cmt.mutbl.to_user_str()).into()

                        }
                        (None, false) => {
                            format!("{} {}",
                                    err.cmt.mutbl.to_user_str(),
                                    self.cmt_to_cow_str(&err.cmt)).into()

                        }
                        (Some(lp), true) => {
                            format!("{} `{}` of {} binding",
                                    self.cmt_to_cow_str(&err.cmt),
                                    self.loan_path_to_string(&lp),
                                    err.cmt.mutbl.to_user_str()).into()
                        }
                        (Some(lp), false) => {
                            format!("{} {} `{}`",
                                    err.cmt.mutbl.to_user_str(),
                                    self.cmt_to_cow_str(&err.cmt),
                                    self.loan_path_to_string(&lp)).into()
                        }
                    }
                };

                let mut db = match err.cause {
                    MutabilityViolation => {
                        let mut db = self.cannot_assign(error_span, &descr, Origin::Ast);
                        if let mc::NoteClosureEnv(upvar_id) = err.cmt.note {
                            let hir_id = upvar_id.var_path.hir_id;
                            let sp = self.tcx.hir().span(hir_id);
                            let fn_closure_msg = "`Fn` closures cannot capture their enclosing \
                                                  environment for modifications";
                            match (self.tcx.sess.source_map().span_to_snippet(sp), &err.cmt.cat) {
                                (_, &Categorization::Upvar(mc::Upvar {
                                    kind: ty::ClosureKind::Fn, ..
                                })) => {
                                    db.note(fn_closure_msg);
                                    // we should point at the cause for this closure being
                                    // identified as `Fn` (like in signature of method this
                                    // closure was passed into)
                                }
                                (Ok(ref snippet), ref cat) => {
                                    let msg = &format!("consider making `{}` mutable", snippet);
                                    let suggestion = format!("mut {}", snippet);

                                    if let &Categorization::Deref(ref cmt, _) = cat {
                                        if let Categorization::Upvar(mc::Upvar {
                                            kind: ty::ClosureKind::Fn, ..
                                        }) = cmt.cat {
                                            db.note(fn_closure_msg);
                                        } else {
                                            db.span_suggestion(
                                                sp,
                                                msg,
                                                suggestion,
                                                Applicability::Unspecified,
                                            );
                                        }
                                    } else {
                                        db.span_suggestion(
                                            sp,
                                            msg,
                                            suggestion,
                                            Applicability::Unspecified,
                                        );
                                    }
                                }
                                _ => {
                                    db.span_help(sp, "consider making this binding mutable");
                                }
                            }
                        }

                        db
                    }
                    BorrowViolation(euv::ClosureCapture(_)) => {
                        self.closure_cannot_assign_to_borrowed(error_span, &descr, Origin::Ast)
                    }
                    BorrowViolation(euv::OverloadedOperator) |
                    BorrowViolation(euv::AddrOf) |
                    BorrowViolation(euv::RefBinding) |
                    BorrowViolation(euv::AutoRef) |
                    BorrowViolation(euv::AutoUnsafe) |
                    BorrowViolation(euv::ForLoop) |
                    BorrowViolation(euv::MatchDiscriminant) => {
                        self.cannot_borrow_path_as_mutable(error_span, &descr, Origin::Ast)
                    }
                    BorrowViolation(euv::ClosureInvocation) => {
                        span_bug!(err.span, "err_mutbl with a closure invocation");
                    }
                };

                // We add a special note about `IndexMut`, if the source of this error
                // is the fact that `Index` is implemented, but `IndexMut` is not. Needing
                // to implement two traits for "one operator" is not very intuitive for
                // many programmers.
                if err.cmt.note == mc::NoteIndex {
                    let node = self.tcx.hir().get(err.cmt.hir_id);

                    // This pattern probably always matches.
                    if let Node::Expr(
                        hir::Expr { node: hir::ExprKind::Index(lhs, _), ..}
                    ) = node {
                        let ty = self.tables.expr_ty(lhs);

                        db.help(&format!(
                            "trait `IndexMut` is required to modify indexed content, but \
                             it is not implemented for `{}`",
                            ty
                        ));
                    }
                }

                self.note_and_explain_mutbl_error(&mut db, &err, &error_span);
                self.note_immutability_blame(
                    &mut db,
                    err.cmt.immutability_blame(),
                    err.cmt.hir_id
                );
                db.emit();
                self.signal_error();
            }
            err_out_of_scope(super_scope, sub_scope, cause) => {
                let msg = match opt_loan_path(&err.cmt) {
                    None => "borrowed value".to_string(),
                    Some(lp) => {
                        format!("`{}`", self.loan_path_to_string(&lp))
                    }
                };

                let mut db = self.path_does_not_live_long_enough(error_span, &msg, Origin::Ast);
                let value_kind = match err.cmt.cat {
                    mc::Categorization::Rvalue(..) => "temporary value",
                    _ => "borrowed value",
                };

                let is_closure = match cause {
                    euv::ClosureCapture(s) => {
                        // The primary span starts out as the closure creation point.
                        // Change the primary span here to highlight the use of the variable
                        // in the closure, because it seems more natural. Highlight
                        // closure creation point as a secondary span.
                        match db.span.primary_span() {
                            Some(primary) => {
                                db.span = MultiSpan::from_span(s);
                                db.span_label(primary, "capture occurs here");
                                db.span_label(s, format!("{} does not live long enough",
                                                         value_kind));
                                true
                            }
                            None => false
                        }
                    }
                    _ => {
                        db.span_label(error_span, format!("{} does not live long enough",
                                                          value_kind));
                        false
                    }
                };

                let sub_span = self.region_end_span(sub_scope);
                let super_span = self.region_end_span(super_scope);

                match (sub_span, super_span) {
                    (Some(s1), Some(s2)) if s1 == s2 => {
                        if !is_closure {
                            let msg = match opt_loan_path(&err.cmt) {
                                None => value_kind.to_string(),
                                Some(lp) => {
                                    format!("`{}`", self.loan_path_to_string(&lp))
                                }
                            };
                            db.span_label(s1,
                                          format!("{} dropped here while still borrowed", msg));
                        } else {
                            db.span_label(s1, format!("{} dropped before borrower", value_kind));
                        }
                        db.note("values in a scope are dropped in the opposite order \
                                they are created");
                    }
                    (Some(s1), Some(s2)) if !is_closure => {
                        let msg = match opt_loan_path(&err.cmt) {
                            None => value_kind.to_string(),
                            Some(lp) => {
                                format!("`{}`", self.loan_path_to_string(&lp))
                            }
                        };
                        db.span_label(s2, format!("{} dropped here while still borrowed", msg));
                        db.span_label(s1, format!("{} needs to live until here", value_kind));
                    }
                    _ => {
                        match sub_span {
                            Some(s) => {
                                db.span_label(s, format!("{} needs to live until here",
                                                          value_kind));
                            }
                            None => {
                                self.tcx.note_and_explain_region(
                                    &self.region_scope_tree,
                                    &mut db,
                                    "borrowed value must be valid for ",
                                    sub_scope,
                                    "...");
                            }
                        }
                        match super_span {
                            Some(s) => {
                                db.span_label(s, format!("{} only lives until here", value_kind));
                            }
                            None => {
                                self.tcx.note_and_explain_region(
                                    &self.region_scope_tree,
                                    &mut db,
                                    "...but borrowed value is only valid for ",
                                    super_scope,
                                    "");
                            }
                        }
                    }
                }

                if let ty::ReScope(scope) = *super_scope {
                    let hir_id = scope.hir_id(&self.region_scope_tree);
                    match self.tcx.hir().find(hir_id) {
                        Some(Node::Stmt(_)) => {
                            if *sub_scope != ty::ReStatic {
                                db.note("consider using a `let` binding to increase its lifetime");
                            }

                        }
                        _ => {}
                    }
                }

                db.emit();
                self.signal_error();
            }
            err_borrowed_pointer_too_short(loan_scope, ptr_scope) => {
                let descr = self.cmt_to_path_or_string(err.cmt);
                let mut db = self.lifetime_too_short_for_reborrow(error_span, &descr, Origin::Ast);
                let descr: Cow<'static, str> = match opt_loan_path(&err.cmt) {
                    Some(lp) => {
                        format!("`{}`", self.loan_path_to_string(&lp)).into()
                    }
                    None => self.cmt_to_cow_str(&err.cmt)
                };
                self.tcx.note_and_explain_region(
                    &self.region_scope_tree,
                    &mut db,
                    &format!("{} would have to be valid for ",
                            descr),
                    loan_scope,
                    "...");
                self.tcx.note_and_explain_region(
                    &self.region_scope_tree,
                    &mut db,
                    &format!("...but {} is only valid for ", descr),
                    ptr_scope,
                    "");

                db.emit();
                self.signal_error();
            }
        }
    }

    pub fn report_aliasability_violation(&self,
                                         span: Span,
                                         kind: AliasableViolationKind,
                                         cause: mc::AliasableReason,
                                         cmt: &mc::cmt_<'tcx>) {
        let mut is_closure = false;
        let prefix = match kind {
            MutabilityViolation => {
                "cannot assign to data"
            }
            BorrowViolation(euv::ClosureCapture(_)) |
            BorrowViolation(euv::OverloadedOperator) |
            BorrowViolation(euv::AddrOf) |
            BorrowViolation(euv::AutoRef) |
            BorrowViolation(euv::AutoUnsafe) |
            BorrowViolation(euv::RefBinding) |
            BorrowViolation(euv::MatchDiscriminant) => {
                "cannot borrow data mutably"
            }
            BorrowViolation(euv::ClosureInvocation) => {
                is_closure = true;
                "closure invocation"
            }

            BorrowViolation(euv::ForLoop) => {
                "`for` loop"
            }
        };

        match cause {
            mc::AliasableStaticMut => {
                // This path cannot occur. `static mut X` is not checked
                // for aliasability violations.
                span_bug!(span, "aliasability violation for static mut `{}`", prefix)
            }
            mc::AliasableStatic | mc::AliasableBorrowed => {}
        };
        let blame = cmt.immutability_blame();
        let mut err = match blame {
            Some(ImmutabilityBlame::ClosureEnv(id)) => {
                // FIXME: the distinction between these 2 messages looks wrong.
                let help_msg = if let BorrowViolation(euv::ClosureCapture(_)) = kind {
                    // The aliasability violation with closure captures can
                    // happen for nested closures, so we know the enclosing
                    // closure incorrectly accepts an `Fn` while it needs to
                    // be `FnMut`.
                    "consider changing this to accept closures that implement `FnMut`"

                } else {
                    "consider changing this closure to take self by mutable reference"
                };
                let hir_id = self.tcx.hir().local_def_id_to_hir_id(id);
                let help_span = self.tcx.hir().span(hir_id);
                self.cannot_act_on_capture_in_sharable_fn(span,
                                                          prefix,
                                                          (help_span, help_msg),
                                                          Origin::Ast)
            }
            _ =>  {
                self.cannot_assign_into_immutable_reference(span, prefix,
                                                            Origin::Ast)
            }
        };
        self.note_immutability_blame(
            &mut err,
            blame,
            cmt.hir_id
        );

        if is_closure {
            err.help("closures behind references must be called via `&mut`");
        }
        err.emit();
        self.signal_error();
    }

    /// Given a type, if it is an immutable reference, return a suggestion to make it mutable
    fn suggest_mut_for_immutable(&self, pty: &hir::Ty, is_implicit_self: bool) -> Option<String> {
        // Check whether the argument is an immutable reference
        debug!("suggest_mut_for_immutable({:?}, {:?})", pty, is_implicit_self);
        if let hir::TyKind::Rptr(lifetime, hir::MutTy {
            mutbl: hir::Mutability::MutImmutable,
            ref ty
        }) = pty.node {
            // Account for existing lifetimes when generating the message
            let pointee_snippet = match self.tcx.sess.source_map().span_to_snippet(ty.span) {
                Ok(snippet) => snippet,
                _ => return None
            };

            let lifetime_snippet = if !lifetime.is_elided() {
                format!("{} ", match self.tcx.sess.source_map().span_to_snippet(lifetime.span) {
                    Ok(lifetime_snippet) => lifetime_snippet,
                    _ => return None
                })
            } else {
                String::new()
            };
            Some(format!("use `&{}mut {}` here to make mutable",
                         lifetime_snippet,
                         if is_implicit_self { "self" } else { &*pointee_snippet }))
        } else {
            None
        }
    }

    fn local_binding_mode(&self, hir_id: hir::HirId) -> ty::BindingMode {
        let pat = match self.tcx.hir().get(hir_id) {
            Node::Binding(pat) => pat,
            node => bug!("bad node for local: {:?}", node)
        };

        match pat.node {
            hir::PatKind::Binding(..) => {
                *self.tables
                     .pat_binding_modes()
                     .get(pat.hir_id)
                     .expect("missing binding mode")
            }
            _ => bug!("local is not a binding: {:?}", pat)
        }
    }

    fn local_ty(&self, hir_id: hir::HirId) -> (Option<&hir::Ty>, bool) {
        let parent = self.tcx.hir().get_parent_node(hir_id);
        let parent_node = self.tcx.hir().get(parent);

        // The parent node is like a fn
        if let Some(fn_like) = FnLikeNode::from_node(parent_node) {
            // `nid`'s parent's `Body`
            let fn_body = self.tcx.hir().body(fn_like.body());
            // Get the position of `node_id` in the arguments list
            let arg_pos = fn_body.arguments.iter().position(|arg| arg.pat.hir_id == hir_id);
            if let Some(i) = arg_pos {
                // The argument's `Ty`
                (Some(&fn_like.decl().inputs[i]),
                 i == 0 && fn_like.decl().implicit_self.has_implicit_self())
            } else {
                (None, false)
            }
        } else {
            (None, false)
        }
    }

    fn note_immutability_blame(&self,
                               db: &mut DiagnosticBuilder<'_>,
                               blame: Option<ImmutabilityBlame<'_>>,
                               error_hir_id: hir::HirId) {
        match blame {
            None => {}
            Some(ImmutabilityBlame::ClosureEnv(_)) => {}
            Some(ImmutabilityBlame::ImmLocal(hir_id)) => {
                self.note_immutable_local(db, error_hir_id, hir_id)
            }
            Some(ImmutabilityBlame::LocalDeref(hir_id)) => {
                match self.local_binding_mode(hir_id) {
                    ty::BindByReference(..) => {
                        let let_span = self.tcx.hir().span(hir_id);
                        let suggestion = suggest_ref_mut(self.tcx, let_span);
                        if let Some(replace_str) = suggestion {
                            db.span_suggestion(
                                let_span,
                                "use a mutable reference instead",
                                replace_str,
                                // I believe this can be machine applicable,
                                // but if there are multiple attempted uses of an immutable
                                // reference, I don't know how rustfix handles it, it might
                                // attempt fixing them multiple times.
                                //                              @estebank
                                Applicability::Unspecified,
                            );
                        }
                    }
                    ty::BindByValue(..) => {
                        if let (Some(local_ty), is_implicit_self) = self.local_ty(hir_id) {
                            if let Some(msg) =
                                 self.suggest_mut_for_immutable(local_ty, is_implicit_self) {
                                db.span_label(local_ty.span, msg);
                            }
                        }
                    }
                }
            }
            Some(ImmutabilityBlame::AdtFieldDeref(_, field)) => {
                let hir_id = match self.tcx.hir().as_local_hir_id(field.did) {
                    Some(hir_id) => hir_id,
                    None => return
                };

                if let Node::Field(ref field) = self.tcx.hir().get(hir_id) {
                    if let Some(msg) = self.suggest_mut_for_immutable(&field.ty, false) {
                        db.span_label(field.ty.span, msg);
                    }
                }
            }
        }
    }

     // Suggest a fix when trying to mutably borrow an immutable local
     // binding: either to make the binding mutable (if its type is
     // not a mutable reference) or to avoid borrowing altogether
    fn note_immutable_local(&self,
                            db: &mut DiagnosticBuilder<'_>,
                            borrowed_hir_id: hir::HirId,
                            binding_hir_id: hir::HirId) {
        let let_span = self.tcx.hir().span(binding_hir_id);
        if let ty::BindByValue(..) = self.local_binding_mode(binding_hir_id) {
            if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(let_span) {
                let (ty, is_implicit_self) = self.local_ty(binding_hir_id);
                if is_implicit_self && snippet != "self" {
                    // avoid suggesting `mut &self`.
                    return
                }
                if let Some(&hir::TyKind::Rptr(
                    _,
                    hir::MutTy {
                        mutbl: hir::MutMutable,
                        ..
                    },
                )) = ty.map(|t| &t.node)
                {
                    let borrow_expr_id = self.tcx.hir().get_parent_node(borrowed_hir_id);
                    db.span_suggestion(
                        self.tcx.hir().span(borrow_expr_id),
                        "consider removing the `&mut`, as it is an \
                        immutable binding to a mutable reference",
                        snippet,
                        Applicability::MachineApplicable,
                    );
                } else {
                    db.span_suggestion(
                        let_span,
                        "make this binding mutable",
                        format!("mut {}", snippet),
                        Applicability::MachineApplicable,
                    );
                }
            }
        }
    }

    fn report_out_of_scope_escaping_closure_capture(&self,
                                                    err: &BckError<'a, 'tcx>,
                                                    capture_span: Span)
    {
        let cmt_path_or_string = self.cmt_to_path_or_string(&err.cmt);

        let suggestion =
            match self.tcx.sess.source_map().span_to_snippet(err.span) {
                Ok(string) => format!("move {}", string),
                Err(_) => "move |<args>| <body>".to_string()
            };

        self.cannot_capture_in_long_lived_closure(err.span,
                                                  &cmt_path_or_string,
                                                  capture_span,
                                                  Origin::Ast)
            .span_suggestion(
                 err.span,
                 &format!("to force the closure to take ownership of {} \
                           (and any other referenced variables), \
                           use the `move` keyword",
                           cmt_path_or_string),
                 suggestion,
                 Applicability::MachineApplicable,
            )
            .emit();
        self.signal_error();
    }

    fn region_end_span(&self, region: ty::Region<'tcx>) -> Option<Span> {
        match *region {
            ty::ReScope(scope) => {
                Some(self.tcx.sess.source_map().end_point(
                        scope.span(self.tcx, &self.region_scope_tree)))
            }
            _ => None
        }
    }

    fn note_and_explain_mutbl_error(&self, db: &mut DiagnosticBuilder<'_>, err: &BckError<'a, 'tcx>,
                                    error_span: &Span) {
        match err.cmt.note {
            mc::NoteClosureEnv(upvar_id) | mc::NoteUpvarRef(upvar_id) => {
                // If this is an `Fn` closure, it simply can't mutate upvars.
                // If it's an `FnMut` closure, the original variable was declared immutable.
                // We need to determine which is the case here.
                let kind = match err.cmt.upvar_cat().unwrap() {
                    Categorization::Upvar(mc::Upvar { kind, .. }) => kind,
                    _ => bug!()
                };
                if *kind == ty::ClosureKind::Fn {
                    let closure_hir_id =
                        self.tcx.hir().local_def_id_to_hir_id(upvar_id.closure_expr_id);
                    db.span_help(self.tcx.hir().span(closure_hir_id),
                                 "consider changing this closure to take \
                                  self by mutable reference");
                }
            }
            _ => {
                if let Categorization::Deref(..) = err.cmt.cat {
                    db.span_label(*error_span, "cannot borrow as mutable");
                } else if let Categorization::Local(local_id) = err.cmt.cat {
                    let span = self.tcx.hir().span(local_id);
                    if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
                        if snippet.starts_with("ref mut ") || snippet.starts_with("&mut ") {
                            db.span_label(*error_span, "cannot reborrow mutably");
                            db.span_label(*error_span, "try removing `&mut` here");
                        } else {
                            db.span_label(*error_span, "cannot borrow mutably");
                        }
                    } else {
                        db.span_label(*error_span, "cannot borrow mutably");
                    }
                } else if let Categorization::Interior(ref cmt, _) = err.cmt.cat {
                    if let mc::MutabilityCategory::McImmutable = cmt.mutbl {
                        db.span_label(*error_span,
                                      "cannot mutably borrow field of immutable binding");
                    }
                }
            }
        }
    }
    pub fn append_loan_path_to_string(&self,
                                      loan_path: &LoanPath<'tcx>,
                                      out: &mut String) {
        match loan_path.kind {
            LpUpvar(ty::UpvarId { var_path: ty::UpvarPath { hir_id: id }, closure_expr_id: _ }) => {
                out.push_str(&self.tcx.hir().name(id).as_str());
            }
            LpVar(id) => {
                out.push_str(&self.tcx.hir().name(id).as_str());
            }

            LpDowncast(ref lp_base, variant_def_id) => {
                out.push('(');
                self.append_loan_path_to_string(&lp_base, out);
                out.push_str(DOWNCAST_PRINTED_OPERATOR);
                out.push_str(&self.tcx.def_path_str(variant_def_id));
                out.push(')');
            }

            LpExtend(ref lp_base, _, LpInterior(_, InteriorField(mc::FieldIndex(_, info)))) => {
                self.append_autoderefd_loan_path_to_string(&lp_base, out);
                out.push('.');
                out.push_str(&info.as_str());
            }

            LpExtend(ref lp_base, _, LpInterior(_, InteriorElement)) => {
                self.append_autoderefd_loan_path_to_string(&lp_base, out);
                out.push_str("[..]");
            }

            LpExtend(ref lp_base, _, LpDeref(_)) => {
                out.push('*');
                self.append_loan_path_to_string(&lp_base, out);
            }
        }
    }

    pub fn append_autoderefd_loan_path_to_string(&self,
                                                 loan_path: &LoanPath<'tcx>,
                                                 out: &mut String) {
        match loan_path.kind {
            LpExtend(ref lp_base, _, LpDeref(_)) => {
                // For a path like `(*x).f` or `(*x)[3]`, autoderef
                // rules would normally allow users to omit the `*x`.
                // So just serialize such paths to `x.f` or x[3]` respectively.
                self.append_autoderefd_loan_path_to_string(&lp_base, out)
            }

            LpDowncast(ref lp_base, variant_def_id) => {
                out.push('(');
                self.append_autoderefd_loan_path_to_string(&lp_base, out);
                out.push_str(DOWNCAST_PRINTED_OPERATOR);
                out.push_str(&self.tcx.def_path_str(variant_def_id));
                out.push(')');
            }

            LpVar(..) | LpUpvar(..) | LpExtend(.., LpInterior(..)) => {
                self.append_loan_path_to_string(loan_path, out)
            }
        }
    }

    pub fn loan_path_to_string(&self, loan_path: &LoanPath<'tcx>) -> String {
        let mut result = String::new();
        self.append_loan_path_to_string(loan_path, &mut result);
        result
    }

    pub fn cmt_to_cow_str(&self, cmt: &mc::cmt_<'tcx>) -> Cow<'static, str> {
        cmt.descriptive_string(self.tcx)
    }

    pub fn cmt_to_path_or_string(&self, cmt: &mc::cmt_<'tcx>) -> String {
        match opt_loan_path(cmt) {
            Some(lp) => format!("`{}`", self.loan_path_to_string(&lp)),
            None => self.cmt_to_cow_str(cmt).into_owned(),
        }
    }
}

impl BitwiseOperator for LoanDataFlowOperator {
    #[inline]
    fn join(&self, succ: usize, pred: usize) -> usize {
        succ | pred // loans from both preds are in scope
    }
}

impl DataFlowOperator for LoanDataFlowOperator {
    #[inline]
    fn initial_value(&self) -> bool {
        false // no loans in scope by default
    }
}

impl fmt::Debug for InteriorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            InteriorField(mc::FieldIndex(_, info)) => write!(f, "{}", info),
            InteriorElement => write!(f, "[]"),
        }
    }
}

impl<'tcx> fmt::Debug for Loan<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Loan_{}({:?}, {:?}, {:?}-{:?}, {:?})",
               self.index,
               self.loan_path,
               self.kind,
               self.gen_scope,
               self.kill_scope,
               self.restricted_paths)
    }
}

impl<'tcx> fmt::Debug for LoanPath<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            LpVar(id) => {
                write!(f, "$({})", ty::tls::with(|tcx| tcx.hir().node_to_string(id)))
            }

            LpUpvar(ty::UpvarId{ var_path: ty::UpvarPath {hir_id: var_id}, closure_expr_id }) => {
                let s = ty::tls::with(|tcx| {
                    tcx.hir().node_to_string(var_id)
                });
                write!(f, "$({} captured by id={:?})", s, closure_expr_id)
            }

            LpDowncast(ref lp, variant_def_id) => {
                let variant_str = if variant_def_id.is_local() {
                    ty::tls::with(|tcx| tcx.def_path_str(variant_def_id))
                } else {
                    format!("{:?}", variant_def_id)
                };
                write!(f, "({:?}{}{})", lp, DOWNCAST_PRINTED_OPERATOR, variant_str)
            }

            LpExtend(ref lp, _, LpDeref(_)) => {
                write!(f, "{:?}.*", lp)
            }

            LpExtend(ref lp, _, LpInterior(_, ref interior)) => {
                write!(f, "{:?}.{:?}", lp, interior)
            }
        }
    }
}

impl<'tcx> fmt::Display for LoanPath<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            LpVar(id) => {
                write!(f, "$({})", ty::tls::with(|tcx| tcx.hir().hir_to_user_string(id)))
            }

            LpUpvar(ty::UpvarId{ var_path: ty::UpvarPath { hir_id }, closure_expr_id: _ }) => {
                let s = ty::tls::with(|tcx| {
                    tcx.hir().node_to_string(hir_id)
                });
                write!(f, "$({} captured by closure)", s)
            }

            LpDowncast(ref lp, variant_def_id) => {
                let variant_str = if variant_def_id.is_local() {
                    ty::tls::with(|tcx| tcx.def_path_str(variant_def_id))
                } else {
                    format!("{:?}", variant_def_id)
                };
                write!(f, "({}{}{})", lp, DOWNCAST_PRINTED_OPERATOR, variant_str)
            }

            LpExtend(ref lp, _, LpDeref(_)) => {
                write!(f, "{}.*", lp)
            }

            LpExtend(ref lp, _, LpInterior(_, ref interior)) => {
                write!(f, "{}.{:?}", lp, interior)
            }
        }
    }
}
