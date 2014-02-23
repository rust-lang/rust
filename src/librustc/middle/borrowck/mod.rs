// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! See doc.rs for a thorough explanation of the borrow checker */

#[allow(non_camel_case_types)];

use mc = middle::mem_categorization;
use middle::ty;
use middle::typeck;
use middle::moves;
use middle::dataflow::DataFlowContext;
use middle::dataflow::DataFlowOperator;
use util::ppaux::{note_and_explain_region, Repr, UserString};

use std::cell::{Cell, RefCell};
use collections::HashMap;
use std::ops::{BitOr, BitAnd};
use std::result::{Result};
use syntax::ast;
use syntax::ast_map;
use syntax::ast_util;
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::visit;
use syntax::visit::{Visitor, FnKind};
use syntax::ast::{FnDecl, Block, NodeId};

macro_rules! if_ok(
    ($inp: expr) => (
        match $inp {
            Ok(v) => { v }
            Err(e) => { return Err(e); }
        }
    )
)

pub mod doc;

pub mod check_loans;

pub mod gather_loans;

pub mod move_data;

pub struct LoanDataFlowOperator;

/// FIXME(pcwalton): Should just be #[deriving(Clone)], but that doesn't work
/// yet on unit structs.
impl Clone for LoanDataFlowOperator {
    fn clone(&self) -> LoanDataFlowOperator {
        LoanDataFlowOperator
    }
}

pub type LoanDataFlow = DataFlowContext<LoanDataFlowOperator>;

impl Visitor<()> for BorrowckCtxt {
    fn visit_fn(&mut self, fk: &FnKind, fd: &FnDecl,
                b: &Block, s: Span, n: NodeId, _: ()) {
        borrowck_fn(self, fk, fd, b, s, n);
    }
}

pub fn check_crate(tcx: ty::ctxt,
                   method_map: typeck::method_map,
                   moves_map: moves::MovesMap,
                   moved_variables_set: moves::MovedVariablesSet,
                   capture_map: moves::CaptureMap,
                   krate: &ast::Crate)
                   -> root_map {
    let mut bccx = BorrowckCtxt {
        tcx: tcx,
        method_map: method_map,
        moves_map: moves_map,
        moved_variables_set: moved_variables_set,
        capture_map: capture_map,
        root_map: root_map(),
        stats: @BorrowStats {
            loaned_paths_same: Cell::new(0),
            loaned_paths_imm: Cell::new(0),
            stable_paths: Cell::new(0),
            guaranteed_paths: Cell::new(0),
        }
    };
    let bccx = &mut bccx;

    visit::walk_crate(bccx, krate, ());

    if tcx.sess.borrowck_stats() {
        println!("--- borrowck stats ---");
        println!("paths requiring guarantees: {}",
                 bccx.stats.guaranteed_paths.get());
        println!("paths requiring loans     : {}",
                 make_stat(bccx, bccx.stats.loaned_paths_same.get()));
        println!("paths requiring imm loans : {}",
                 make_stat(bccx, bccx.stats.loaned_paths_imm.get()));
        println!("stable paths              : {}",
                 make_stat(bccx, bccx.stats.stable_paths.get()));
    }

    return bccx.root_map;

    fn make_stat(bccx: &mut BorrowckCtxt, stat: uint) -> ~str {
        let stat_f = stat as f64;
        let total = bccx.stats.guaranteed_paths.get() as f64;
        format!("{} ({:.0f}%)", stat  , stat_f * 100.0 / total)
    }
}

fn borrowck_fn(this: &mut BorrowckCtxt,
               fk: &FnKind,
               decl: &ast::FnDecl,
               body: &ast::Block,
               sp: Span,
               id: ast::NodeId) {
    debug!("borrowck_fn(id={})", id);

    // Check the body of fn items.
    let (id_range, all_loans, move_data) =
        gather_loans::gather_loans(this, decl, body);
    let all_loans = all_loans.borrow();
    let mut loan_dfcx =
        DataFlowContext::new(this.tcx,
                             this.method_map,
                             LoanDataFlowOperator,
                             id_range,
                             all_loans.get().len());
    for (loan_idx, loan) in all_loans.get().iter().enumerate() {
        loan_dfcx.add_gen(loan.gen_scope, loan_idx);
        loan_dfcx.add_kill(loan.kill_scope, loan_idx);
    }
    loan_dfcx.propagate(body);

    let flowed_moves = move_data::FlowedMoveData::new(move_data,
                                                      this.tcx,
                                                      this.method_map,
                                                      id_range,
                                                      body);

    check_loans::check_loans(this, &loan_dfcx, flowed_moves,
                             *all_loans.get(), body);

    visit::walk_fn(this, fk, decl, body, sp, id, ());
}

// ----------------------------------------------------------------------
// Type definitions

pub struct BorrowckCtxt {
    tcx: ty::ctxt,
    method_map: typeck::method_map,
    moves_map: moves::MovesMap,
    moved_variables_set: moves::MovedVariablesSet,
    capture_map: moves::CaptureMap,
    root_map: root_map,

    // Statistics:
    stats: @BorrowStats
}

pub struct BorrowStats {
    loaned_paths_same: Cell<uint>,
    loaned_paths_imm: Cell<uint>,
    stable_paths: Cell<uint>,
    guaranteed_paths: Cell<uint>,
}

// The keys to the root map combine the `id` of the deref expression
// with the number of types that it is *autodereferenced*. So, for
// example, imagine I have a variable `x: @@@T` and an expression
// `(*x).f`.  This will have 3 derefs, one explicit and then two
// autoderefs. These are the relevant `root_map_key` values that could
// appear:
//
//    {id:*x, derefs:0} --> roots `x` (type: @@@T, due to explicit deref)
//    {id:*x, derefs:1} --> roots `*x` (type: @@T, due to autoderef #1)
//    {id:*x, derefs:2} --> roots `**x` (type: @T, due to autoderef #2)
//
// Note that there is no entry with derefs:3---the type of that expression
// is T, which is not a box.
#[deriving(Eq, Hash)]
pub struct root_map_key {
    id: ast::NodeId,
    derefs: uint
}

pub type BckResult<T> = Result<T, BckError>;

#[deriving(Eq)]
pub enum PartialTotal {
    Partial,   // Loan affects some portion
    Total      // Loan affects entire path
}

///////////////////////////////////////////////////////////////////////////
// Loans and loan paths

/// Record of a loan that was issued.
pub struct Loan {
    index: uint,
    loan_path: @LoanPath,
    cmt: mc::cmt,
    kind: ty::BorrowKind,
    restrictions: ~[Restriction],
    gen_scope: ast::NodeId,
    kill_scope: ast::NodeId,
    span: Span,
    cause: LoanCause,
}

#[deriving(Eq)]
pub enum LoanCause {
    ClosureCapture(Span),
    AddrOf,
    AutoRef,
    RefBinding,
}

#[deriving(Eq, Hash)]
pub enum LoanPath {
    LpVar(ast::NodeId),               // `x` in doc.rs
    LpExtend(@LoanPath, mc::MutabilityCategory, LoanPathElem)
}

#[deriving(Eq, Hash)]
pub enum LoanPathElem {
    LpDeref(mc::PointerKind),    // `*LV` in doc.rs
    LpInterior(mc::InteriorKind) // `LV.f` in doc.rs
}

impl LoanPath {
    pub fn node_id(&self) -> ast::NodeId {
        match *self {
            LpVar(local_id) => local_id,
            LpExtend(base, _, _) => base.node_id()
        }
    }
}

pub fn opt_loan_path(cmt: mc::cmt) -> Option<@LoanPath> {
    //! Computes the `LoanPath` (if any) for a `cmt`.
    //! Note that this logic is somewhat duplicated in
    //! the method `compute()` found in `gather_loans::restrictions`,
    //! which allows it to share common loan path pieces as it
    //! traverses the CMT.

    match cmt.cat {
        mc::cat_rvalue(..) |
        mc::cat_static_item |
        mc::cat_copied_upvar(_) => {
            None
        }

        mc::cat_local(id) |
        mc::cat_arg(id) |
        mc::cat_upvar(ty::UpvarId {var_id: id, ..}, _) => {
            Some(@LpVar(id))
        }

        mc::cat_deref(cmt_base, _, pk) => {
            opt_loan_path(cmt_base).map(|lp| {
                @LpExtend(lp, cmt.mutbl, LpDeref(pk))
            })
        }

        mc::cat_interior(cmt_base, ik) => {
            opt_loan_path(cmt_base).map(|lp| {
                @LpExtend(lp, cmt.mutbl, LpInterior(ik))
            })
        }

        mc::cat_downcast(cmt_base) |
        mc::cat_discr(cmt_base, _) => {
            opt_loan_path(cmt_base)
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Restrictions
//
// Borrowing an lvalue often results in *restrictions* that limit what
// can be done with this lvalue during the scope of the loan:
//
// - `RESTR_MUTATE`: The lvalue may not be modified or `&mut` borrowed.
// - `RESTR_FREEZE`: `&` borrows of the lvalue are forbidden.
//
// In addition, no value which is restricted may be moved. Therefore,
// restrictions are meaningful even if the RestrictionSet is empty,
// because the restriction against moves is implied.

pub struct Restriction {
    loan_path: @LoanPath,
    set: RestrictionSet
}

#[deriving(Eq)]
pub struct RestrictionSet {
    bits: u32
}

pub static RESTR_EMPTY: RestrictionSet  = RestrictionSet {bits: 0b0000};
pub static RESTR_MUTATE: RestrictionSet = RestrictionSet {bits: 0b0001};
pub static RESTR_FREEZE: RestrictionSet = RestrictionSet {bits: 0b0010};

impl RestrictionSet {
    pub fn intersects(&self, restr: RestrictionSet) -> bool {
        (self.bits & restr.bits) != 0
    }

    pub fn contains_all(&self, restr: RestrictionSet) -> bool {
        (self.bits & restr.bits) == restr.bits
    }
}

impl BitOr<RestrictionSet,RestrictionSet> for RestrictionSet {
    fn bitor(&self, rhs: &RestrictionSet) -> RestrictionSet {
        RestrictionSet {bits: self.bits | rhs.bits}
    }
}

impl BitAnd<RestrictionSet,RestrictionSet> for RestrictionSet {
    fn bitand(&self, rhs: &RestrictionSet) -> RestrictionSet {
        RestrictionSet {bits: self.bits & rhs.bits}
    }
}

impl Repr for RestrictionSet {
    fn repr(&self, _tcx: ty::ctxt) -> ~str {
        format!("RestrictionSet(0x{:x})", self.bits as uint)
    }
}

///////////////////////////////////////////////////////////////////////////
// Rooting of managed boxes
//
// When we borrow the interior of a managed box, it is sometimes
// necessary to *root* the box, meaning to stash a copy of the box
// somewhere that the garbage collector will find it. This ensures
// that the box is not collected for the lifetime of the borrow.
//
// As part of this rooting, we sometimes also freeze the box at
// runtime, meaning that we dynamically detect when the box is
// borrowed in incompatible ways.
//
// Both of these actions are driven through the `root_map`, which maps
// from a node to the dynamic rooting action that should be taken when
// that node executes. The node is identified through a
// `root_map_key`, which pairs a node-id and a deref count---the
// problem is that sometimes the box that needs to be rooted is only
// uncovered after a certain number of auto-derefs.

pub struct RootInfo {
    scope: ast::NodeId,
}

pub type root_map = @RefCell<HashMap<root_map_key, RootInfo>>;

pub fn root_map() -> root_map {
    return @RefCell::new(HashMap::new());
}

///////////////////////////////////////////////////////////////////////////
// Errors

// Errors that can occur
#[deriving(Eq)]
pub enum bckerr_code {
    err_mutbl,
    err_out_of_root_scope(ty::Region, ty::Region), // superscope, subscope
    err_out_of_scope(ty::Region, ty::Region), // superscope, subscope
    err_borrowed_pointer_too_short(
        ty::Region, ty::Region, RestrictionSet), // loan, ptr
}

// Combination of an error code and the categorization of the expression
// that caused it
#[deriving(Eq)]
pub struct BckError {
    span: Span,
    cause: LoanCause,
    cmt: mc::cmt,
    code: bckerr_code
}

pub enum AliasableViolationKind {
    MutabilityViolation,
    BorrowViolation(LoanCause)
}

pub enum MovedValueUseKind {
    MovedInUse,
    MovedInCapture,
}

///////////////////////////////////////////////////////////////////////////
// Misc

impl BorrowckCtxt {
    pub fn is_subregion_of(&self, r_sub: ty::Region, r_sup: ty::Region)
                           -> bool {
        self.tcx.region_maps.is_subregion_of(r_sub, r_sup)
    }

    pub fn is_subscope_of(&self, r_sub: ast::NodeId, r_sup: ast::NodeId)
                          -> bool {
        self.tcx.region_maps.is_subscope_of(r_sub, r_sup)
    }

    pub fn is_move(&self, id: ast::NodeId) -> bool {
        let moves_map = self.moves_map.borrow();
        moves_map.get().contains(&id)
    }

    pub fn mc(&self) -> mc::MemCategorizationContext<TcxTyper> {
        mc::MemCategorizationContext {
            typer: TcxTyper {
                tcx: self.tcx,
                method_map: self.method_map
            }
        }
    }

    pub fn cat_expr(&self, expr: &ast::Expr) -> mc::cmt {
        match self.mc().cat_expr(expr) {
            Ok(c) => c,
            Err(()) => {
                self.tcx.sess.span_bug(expr.span, "error in mem categorization");
            }
        }
    }

    pub fn cat_expr_unadjusted(&self, expr: &ast::Expr) -> mc::cmt {
        match self.mc().cat_expr_unadjusted(expr) {
            Ok(c) => c,
            Err(()) => {
                self.tcx.sess.span_bug(expr.span, "error in mem categorization");
            }
        }
    }

    pub fn cat_expr_autoderefd(&self,
                               expr: &ast::Expr,
                               adj: &ty::AutoAdjustment)
                               -> mc::cmt {
        let r = match *adj {
            ty::AutoAddEnv(..) | ty::AutoObject(..) => {
                // no autoderefs
                self.mc().cat_expr_unadjusted(expr)
            }

            ty::AutoDerefRef(
                ty::AutoDerefRef {
                    autoderefs: autoderefs, ..}) => {
                self.mc().cat_expr_autoderefd(expr, autoderefs)
            }
        };

        match r {
            Ok(c) => c,
            Err(()) => {
                self.tcx.sess.span_bug(expr.span,
                                       "error in mem categorization");
            }
        }
    }

    pub fn cat_def(&self,
                   id: ast::NodeId,
                   span: Span,
                   ty: ty::t,
                   def: ast::Def)
                   -> mc::cmt {
        match self.mc().cat_def(id, span, ty, def) {
            Ok(c) => c,
            Err(()) => {
                self.tcx.sess.span_bug(span, "error in mem categorization");
            }
        }
    }

    pub fn cat_captured_var(&self,
                            id: ast::NodeId,
                            span: Span,
                            captured_var: &moves::CaptureVar) -> mc::cmt {
        // Create the cmt for the variable being borrowed, from the
        // caller's perspective
        let var_id = ast_util::def_id_of_def(captured_var.def).node;
        let var_ty = ty::node_id_to_type(self.tcx, var_id);
        self.cat_def(id, span, var_ty, captured_var.def)
    }

    pub fn cat_discr(&self, cmt: mc::cmt, match_id: ast::NodeId) -> mc::cmt {
        @mc::cmt_ {cat:mc::cat_discr(cmt, match_id),
                   mutbl:cmt.mutbl.inherit(),
                   ..*cmt}
    }

    pub fn cat_pattern(&self,
                       cmt: mc::cmt,
                       pat: @ast::Pat,
                       op: |mc::cmt, &ast::Pat|) {
        let r = self.mc().cat_pattern(cmt, pat, |_,x,y| op(x,y));
        assert!(r.is_ok());
    }

    pub fn report(&self, err: BckError) {
        self.span_err(
            err.span,
            self.bckerr_to_str(err));
        self.note_and_explain_bckerr(err);
    }

    pub fn report_use_of_moved_value(&self,
                                     use_span: Span,
                                     use_kind: MovedValueUseKind,
                                     lp: &LoanPath,
                                     move: &move_data::Move,
                                     moved_lp: @LoanPath) {
        let verb = match use_kind {
            MovedInUse => "use",
            MovedInCapture => "capture",
        };

        match move.kind {
            move_data::Declared => {
                self.tcx.sess.span_err(
                    use_span,
                    format!("{} of possibly uninitialized value: `{}`",
                         verb,
                         self.loan_path_to_str(lp)));
            }
            _ => {
                let partially = if lp == moved_lp {""} else {"partially "};
                self.tcx.sess.span_err(
                    use_span,
                    format!("{} of {}moved value: `{}`",
                         verb,
                         partially,
                         self.loan_path_to_str(lp)));
            }
        }

        match move.kind {
            move_data::Declared => {}

            move_data::MoveExpr => {
                let (expr_ty, expr_span) = match self.tcx.map.find(move.id) {
                    Some(ast_map::NodeExpr(expr)) => {
                        (ty::expr_ty_adjusted(self.tcx, expr), expr.span)
                    }
                    r => self.tcx.sess.bug(format!("MoveExpr({:?}) maps to {:?}, not Expr",
                                                   move.id, r))
                };
                let suggestion = move_suggestion(self.tcx, expr_ty,
                        "moved by default (use `copy` to override)");
                self.tcx.sess.span_note(
                    expr_span,
                    format!("`{}` moved here because it has type `{}`, which is {}",
                         self.loan_path_to_str(moved_lp),
                         expr_ty.user_string(self.tcx), suggestion));
            }

            move_data::MovePat => {
                let pat_ty = ty::node_id_to_type(self.tcx, move.id);
                self.tcx.sess.span_note(self.tcx.map.span(move.id),
                    format!("`{}` moved here because it has type `{}`, \
                          which is moved by default (use `ref` to override)",
                         self.loan_path_to_str(moved_lp),
                         pat_ty.user_string(self.tcx)));
            }

            move_data::Captured => {
                let (expr_ty, expr_span) = match self.tcx.map.find(move.id) {
                    Some(ast_map::NodeExpr(expr)) => {
                        (ty::expr_ty_adjusted(self.tcx, expr), expr.span)
                    }
                    r => self.tcx.sess.bug(format!("Captured({:?}) maps to {:?}, not Expr",
                                                   move.id, r))
                };
                let suggestion = move_suggestion(self.tcx, expr_ty,
                        "moved by default (make a copy and \
                         capture that instead to override)");
                self.tcx.sess.span_note(
                    expr_span,
                    format!("`{}` moved into closure environment here because it \
                          has type `{}`, which is {}",
                         self.loan_path_to_str(moved_lp),
                         expr_ty.user_string(self.tcx), suggestion));
            }
        }

        fn move_suggestion(tcx: ty::ctxt, ty: ty::t, default_msg: &'static str)
                          -> &'static str {
            match ty::get(ty).sty {
                ty::ty_closure(ref cty) if cty.sigil == ast::BorrowedSigil =>
                    "a non-copyable stack closure (capture it in a new closure, \
                     e.g. `|x| f(x)`, to override)",
                _ if ty::type_moves_by_default(tcx, ty) =>
                    "non-copyable (perhaps you meant to use clone()?)",
                _ => default_msg,
            }
        }
    }

    pub fn report_reassigned_immutable_variable(&self,
                                                span: Span,
                                                lp: &LoanPath,
                                                assign:
                                                &move_data::Assignment) {
        self.tcx.sess.span_err(
            span,
            format!("re-assignment of immutable variable `{}`",
                 self.loan_path_to_str(lp)));
        self.tcx.sess.span_note(
            assign.span,
            format!("prior assignment occurs here"));
    }

    pub fn span_err(&self, s: Span, m: &str) {
        self.tcx.sess.span_err(s, m);
    }

    pub fn span_note(&self, s: Span, m: &str) {
        self.tcx.sess.span_note(s, m);
    }

    pub fn span_end_note(&self, s: Span, m: &str) {
        self.tcx.sess.span_end_note(s, m);
    }

    pub fn bckerr_to_str(&self, err: BckError) -> ~str {
        match err.code {
            err_mutbl => {
                let descr = match opt_loan_path(err.cmt) {
                    None => format!("{} {}",
                                    err.cmt.mutbl.to_user_str(),
                                    self.cmt_to_str(err.cmt)),
                    Some(lp) => format!("{} {} `{}`",
                                        err.cmt.mutbl.to_user_str(),
                                        self.cmt_to_str(err.cmt),
                                        self.loan_path_to_str(lp)),
                };

                match err.cause {
                    ClosureCapture(_) => {
                        format!("closure cannot assign to {}", descr)
                    }
                    AddrOf | RefBinding | AutoRef => {
                        format!("cannot borrow {} as mutable", descr)
                    }
                }
            }
            err_out_of_root_scope(..) => {
                format!("cannot root managed value long enough")
            }
            err_out_of_scope(..) => {
                let msg = match opt_loan_path(err.cmt) {
                    None => format!("borrowed value"),
                    Some(lp) => format!("`{}`", self.loan_path_to_str(lp)),
                };
                format!("{} does not live long enough", msg)
            }
            err_borrowed_pointer_too_short(..) => {
                let descr = match opt_loan_path(err.cmt) {
                    Some(lp) => format!("`{}`", self.loan_path_to_str(lp)),
                    None => self.cmt_to_str(err.cmt),
                };

                format!("lifetime of {} is too short to guarantee \
                        its contents can be safely reborrowed",
                        descr)
            }
        }
    }

    pub fn report_aliasability_violation(&self,
                                         span: Span,
                                         kind: AliasableViolationKind,
                                         cause: mc::AliasableReason) {
        let prefix = match kind {
            MutabilityViolation => {
                "cannot assign to data"
            }
            BorrowViolation(ClosureCapture(_)) => {
                // I don't think we can get aliasability violations
                // with closure captures, so no need to come up with a
                // good error message. The reason this cannot happen
                // is because we only capture local variables in
                // closures, and those are never aliasable.
                self.tcx.sess.span_bug(
                    span,
                    "aliasability violation with closure");
            }
            BorrowViolation(AddrOf) |
            BorrowViolation(AutoRef) |
            BorrowViolation(RefBinding) => {
                "cannot borrow data mutably"
            }
        };

        match cause {
            mc::AliasableOther => {
                self.tcx.sess.span_err(
                    span,
                    format!("{} in an aliasable location", prefix));
            }
            mc::AliasableStatic |
            mc::AliasableStaticMut => {
                self.tcx.sess.span_err(
                    span,
                    format!("{} in a static location", prefix));
            }
            mc::AliasableManaged => {
                self.tcx.sess.span_err(
                    span,
                    format!("{} in a `@` pointer", prefix));
            }
            mc::AliasableBorrowed => {
                self.tcx.sess.span_err(
                    span,
                    format!("{} in a `&` reference", prefix));
            }
        }
    }

    pub fn note_and_explain_bckerr(&self, err: BckError) {
        let code = err.code;
        match code {
            err_mutbl(..) => { }

            err_out_of_root_scope(super_scope, sub_scope) => {
                note_and_explain_region(
                    self.tcx,
                    "managed value would have to be rooted for ",
                    sub_scope,
                    "...");
                note_and_explain_region(
                    self.tcx,
                    "...but can only be rooted for ",
                    super_scope,
                    "");
            }

            err_out_of_scope(super_scope, sub_scope) => {
                note_and_explain_region(
                    self.tcx,
                    "reference must be valid for ",
                    sub_scope,
                    "...");
                note_and_explain_region(
                    self.tcx,
                    "...but borrowed value is only valid for ",
                    super_scope,
                    "");
            }

            err_borrowed_pointer_too_short(loan_scope, ptr_scope, _) => {
                let descr = match opt_loan_path(err.cmt) {
                    Some(lp) => format!("`{}`", self.loan_path_to_str(lp)),
                    None => self.cmt_to_str(err.cmt),
                };
                note_and_explain_region(
                    self.tcx,
                    format!("{} would have to be valid for ", descr),
                    loan_scope,
                    "...");
                note_and_explain_region(
                    self.tcx,
                    format!("...but {} is only valid for ", descr),
                    ptr_scope,
                    "");
            }
        }
    }

    pub fn append_loan_path_to_str(&self,
                                   loan_path: &LoanPath,
                                   out: &mut ~str) {
        match *loan_path {
            LpVar(id) => {
                out.push_str(ty::local_var_name_str(self.tcx, id).get());
            }

            LpExtend(lp_base, _, LpInterior(mc::InteriorField(fname))) => {
                self.append_autoderefd_loan_path_to_str(lp_base, out);
                match fname {
                    mc::NamedField(fname) => {
                        out.push_char('.');
                        out.push_str(token::get_name(fname).get());
                    }
                    mc::PositionalField(idx) => {
                        out.push_char('#'); // invent a notation here
                        out.push_str(idx.to_str());
                    }
                }
            }

            LpExtend(lp_base, _, LpInterior(mc::InteriorElement(_))) => {
                self.append_autoderefd_loan_path_to_str(lp_base, out);
                out.push_str("[..]");
            }

            LpExtend(lp_base, _, LpDeref(_)) => {
                out.push_char('*');
                self.append_loan_path_to_str(lp_base, out);
            }
        }
    }

    pub fn append_autoderefd_loan_path_to_str(&self,
                                              loan_path: &LoanPath,
                                              out: &mut ~str) {
        match *loan_path {
            LpExtend(lp_base, _, LpDeref(_)) => {
                // For a path like `(*x).f` or `(*x)[3]`, autoderef
                // rules would normally allow users to omit the `*x`.
                // So just serialize such paths to `x.f` or x[3]` respectively.
                self.append_autoderefd_loan_path_to_str(lp_base, out)
            }

            LpVar(..) | LpExtend(_, _, LpInterior(..)) => {
                self.append_loan_path_to_str(loan_path, out)
            }
        }
    }

    pub fn loan_path_to_str(&self, loan_path: &LoanPath) -> ~str {
        let mut result = ~"";
        self.append_loan_path_to_str(loan_path, &mut result);
        result
    }

    pub fn cmt_to_str(&self, cmt: mc::cmt) -> ~str {
        self.mc().cmt_to_str(cmt)
    }

    pub fn mut_to_str(&self, mutbl: ast::Mutability) -> ~str {
        self.mc().mut_to_str(mutbl)
    }

    pub fn mut_to_keyword(&self, mutbl: ast::Mutability) -> &'static str {
        match mutbl {
            ast::MutImmutable => "",
            ast::MutMutable => "mut",
        }
    }
}

impl DataFlowOperator for LoanDataFlowOperator {
    #[inline]
    fn initial_value(&self) -> bool {
        false // no loans in scope by default
    }

    #[inline]
    fn join(&self, succ: uint, pred: uint) -> uint {
        succ | pred // loans from both preds are in scope
    }
}

impl Repr for Loan {
    fn repr(&self, tcx: ty::ctxt) -> ~str {
        format!("Loan_{:?}({}, {:?}, {:?}-{:?}, {})",
             self.index,
             self.loan_path.repr(tcx),
             self.kind,
             self.gen_scope,
             self.kill_scope,
             self.restrictions.repr(tcx))
    }
}

impl Repr for Restriction {
    fn repr(&self, tcx: ty::ctxt) -> ~str {
        format!("Restriction({}, {:x})",
             self.loan_path.repr(tcx),
             self.set.bits as uint)
    }
}

impl Repr for LoanPath {
    fn repr(&self, tcx: ty::ctxt) -> ~str {
        match self {
            &LpVar(id) => {
                format!("$({})", tcx.map.node_to_str(id))
            }

            &LpExtend(lp, _, LpDeref(_)) => {
                format!("{}.*", lp.repr(tcx))
            }

            &LpExtend(lp, _, LpInterior(ref interior)) => {
                format!("{}.{}", lp.repr(tcx), interior.repr(tcx))
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////

struct TcxTyper {
    tcx: ty::ctxt,
    method_map: typeck::method_map,
}

impl mc::Typer for TcxTyper {
    fn tcx(&self) -> ty::ctxt {
        self.tcx
    }

    fn node_ty(&mut self, id: ast::NodeId) -> mc::McResult<ty::t> {
        Ok(ty::node_id_to_type(self.tcx, id))
    }

    fn adjustment(&mut self, id: ast::NodeId) -> Option<@ty::AutoAdjustment> {
        let adjustments = self.tcx.adjustments.borrow();
        adjustments.get().find_copy(&id)
    }

    fn is_method_call(&mut self, id: ast::NodeId) -> bool {
        let method_map = self.method_map.borrow();
        method_map.get().contains_key(&id)
    }

    fn temporary_scope(&mut self, id: ast::NodeId) -> Option<ast::NodeId> {
        self.tcx.region_maps.temporary_scope(id)
    }

    fn upvar_borrow(&mut self, id: ty::UpvarId) -> ty::UpvarBorrow {
        let upvar_borrow_map = self.tcx.upvar_borrow_map.borrow();
        upvar_borrow_map.get().get_copy(&id)
    }
}
