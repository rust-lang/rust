//! See The Book chapter on the borrow checker for more details.

#![allow(non_camel_case_types)]

pub use LoanPathKind::*;
pub use LoanPathElem::*;

use InteriorKind::*;

use rustc::hir::HirId;
use rustc::hir::Node;
use rustc::cfg;
use rustc::middle::borrowck::{BorrowCheckResult, SignalledError};
use rustc::hir::def_id::{DefId, LocalDefId};
use rustc::middle::mem_categorization as mc;
use rustc::middle::mem_categorization::Categorization;
use rustc::middle::region;
use rustc::middle::free_region::RegionRelations;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::query::Providers;

use std::borrow::Cow;
use std::cell::{Cell};
use std::fmt;
use std::rc::Rc;
use std::hash::{Hash, Hasher};
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
        signalled_any_error: bccx.signalled_any_error.into_inner(),
    })
}

fn build_borrowck_dataflow_data<'a, 'c, 'tcx, F>(this: &mut BorrowckCtxt<'a, 'tcx>,
                                                 force_analysis: bool,
                                                 body_id: hir::BodyId,
                                                 get_cfg: F)
                                                 -> Option<AnalysisData<'tcx>>
    where F: FnOnce(&mut BorrowckCtxt<'a, 'tcx>) -> &'c cfg::CFG
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
    cfg: &cfg::CFG)
    -> (BorrowckCtxt<'a, 'tcx>, AnalysisData<'tcx>)
{
    let owner_id = tcx.hir().body_owner(body_id);
    let owner_def_id = tcx.hir().local_def_id(owner_id);
    let tables = tcx.typeck_tables_of(owner_def_id);
    let region_scope_tree = tcx.region_scope_tree(owner_def_id);
    let body = tcx.hir().body(body_id);
    let mut bccx = BorrowckCtxt {
        tcx,
        tables,
        region_scope_tree,
        owner_def_id,
        body,
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

    signalled_any_error: Cell<SignalledError>,
}


impl<'a, 'tcx: 'a> BorrowckCtxt<'a, 'tcx> {
    fn signal_error(&self) {
        self.signalled_any_error.set(SignalledError::SawSomeError);
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

impl<'a, 'tcx> LoanPath<'tcx> {
    pub fn kill_scope(&self, bccx: &BorrowckCtxt<'a, 'tcx>) -> region::Scope {
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
// Misc

impl<'a, 'tcx> BorrowckCtxt<'a, 'tcx> {
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
