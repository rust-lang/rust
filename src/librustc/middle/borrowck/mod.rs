// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! See doc.rs for a thorough explanation of the borrow checker */

use mc = middle::mem_categorization;
use middle::ty;
use middle::typeck;
use middle::moves;
use middle::dataflow::DataFlowContext;
use middle::dataflow::DataFlowOperator;
use util::common::stmt_set;
use util::ppaux::{note_and_explain_region, Repr};

#[cfg(stage0)]
use core; // NOTE: this can be removed after the next snapshot
use core::hashmap::{HashSet, HashMap};
use core::io;
use core::result::{Result};
use core::ops::{BitOr, BitAnd};
use syntax::ast;
use syntax::ast_map;
use syntax::visit;
use syntax::codemap::span;

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

#[path="gather_loans/mod.rs"]
pub mod gather_loans;

pub struct LoanDataFlowOperator;
pub type LoanDataFlow = DataFlowContext<LoanDataFlowOperator>;

pub fn check_crate(
    tcx: ty::ctxt,
    method_map: typeck::method_map,
    moves_map: moves::MovesMap,
    moved_variables_set: moves::MovedVariablesSet,
    capture_map: moves::CaptureMap,
    crate: @ast::crate) -> (root_map, write_guard_map)
{
    let bccx = @BorrowckCtxt {
        tcx: tcx,
        method_map: method_map,
        moves_map: moves_map,
        moved_variables_set: moved_variables_set,
        capture_map: capture_map,
        root_map: root_map(),
        loan_map: @mut HashMap::new(),
        write_guard_map: @mut HashSet::new(),
        stmt_map: @mut HashSet::new(),
        stats: @mut BorrowStats {
            loaned_paths_same: 0,
            loaned_paths_imm: 0,
            stable_paths: 0,
            req_pure_paths: 0,
            guaranteed_paths: 0,
        }
    };

    let v = visit::mk_vt(@visit::Visitor {visit_fn: borrowck_fn,
                                          ..*visit::default_visitor()});
    visit::visit_crate(crate, bccx, v);

    if tcx.sess.borrowck_stats() {
        io::println(~"--- borrowck stats ---");
        io::println(fmt!("paths requiring guarantees: %u",
                        bccx.stats.guaranteed_paths));
        io::println(fmt!("paths requiring loans     : %s",
                         make_stat(bccx, bccx.stats.loaned_paths_same)));
        io::println(fmt!("paths requiring imm loans : %s",
                         make_stat(bccx, bccx.stats.loaned_paths_imm)));
        io::println(fmt!("stable paths              : %s",
                         make_stat(bccx, bccx.stats.stable_paths)));
        io::println(fmt!("paths requiring purity    : %s",
                         make_stat(bccx, bccx.stats.req_pure_paths)));
    }

    return (bccx.root_map, bccx.write_guard_map);

    fn make_stat(bccx: &BorrowckCtxt, stat: uint) -> ~str {
        let stat_f = stat as float;
        let total = bccx.stats.guaranteed_paths as float;
        fmt!("%u (%.0f%%)", stat  , stat_f * 100f / total)
    }
}

fn borrowck_fn(fk: &visit::fn_kind,
               decl: &ast::fn_decl,
               body: &ast::blk,
               sp: span,
               id: ast::node_id,
               this: @BorrowckCtxt,
               v: visit::vt<@BorrowckCtxt>) {
    match fk {
        &visit::fk_anon(*) |
        &visit::fk_fn_block(*) => {
            // Closures are checked as part of their containing fn item.
        }

        &visit::fk_item_fn(*) |
        &visit::fk_method(*) => {
            debug!("borrowck_fn(id=%?)", id);

            // Check the body of fn items.
            let (id_range, all_loans) =
                gather_loans::gather_loans(this, body);
            let all_loans: &~[Loan] = &*all_loans; // FIXME(#5074)
            let mut dfcx =
                DataFlowContext::new(this.tcx,
                                     this.method_map,
                                     LoanDataFlowOperator,
                                     id_range,
                                     all_loans.len());
            for all_loans.eachi |loan_idx, loan| {
                dfcx.add_gen(loan.gen_scope, loan_idx);
                dfcx.add_kill(loan.kill_scope, loan_idx);
            }
            dfcx.propagate(body);
            check_loans::check_loans(this, &dfcx, *all_loans, body);
        }
    }

    visit::visit_fn(fk, decl, body, sp, id, this, v);
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
    loan_map: LoanMap,
    write_guard_map: write_guard_map,
    stmt_map: stmt_set,

    // Statistics:
    stats: @mut BorrowStats
}

pub struct BorrowStats {
    loaned_paths_same: uint,
    loaned_paths_imm: uint,
    stable_paths: uint,
    req_pure_paths: uint,
    guaranteed_paths: uint
}

pub type LoanMap = @mut HashMap<ast::node_id, @Loan>;

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
//
// Note that implicit dereferences also occur with indexing of `@[]`,
// `@str`, etc.  The same rules apply. So, for example, given a
// variable `x` of type `@[@[...]]`, if I have an instance of the
// expression `x[0]` which is then auto-slice'd, there would be two
// potential entries in the root map, both with the id of the `x[0]`
// expression. The entry with `derefs==0` refers to the deref of `x`
// used as part of evaluating `x[0]`. The entry with `derefs==1`
// refers to the deref of the `x[0]` that occurs as part of the
// auto-slice.
#[deriving(Eq, IterBytes)]
pub struct root_map_key {
    id: ast::node_id,
    derefs: uint
}

// A set containing IDs of expressions of gc'd type that need to have a write
// guard.
pub type write_guard_map = @mut HashSet<root_map_key>;

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
    mutbl: ast::mutability,
    restrictions: ~[Restriction],
    gen_scope: ast::node_id,
    kill_scope: ast::node_id,
    span: span,
}

#[deriving(Eq)]
pub enum LoanPath {
    LpVar(ast::node_id),               // `x` in doc.rs
    LpExtend(@LoanPath, mc::MutabilityCategory, LoanPathElem)
}

#[deriving(Eq)]
pub enum LoanPathElem {
    LpDeref,                      // `*LV` in doc.rs
    LpInterior(mc::interior_kind) // `LV.f` in doc.rs
}

pub impl LoanPath {
    fn node_id(&self) -> ast::node_id {
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
        mc::cat_rvalue |
        mc::cat_static_item |
        mc::cat_copied_upvar(_) |
        mc::cat_implicit_self => {
            None
        }

        mc::cat_local(id) |
        mc::cat_arg(id) |
        mc::cat_self(id) => {
            Some(@LpVar(id))
        }

        mc::cat_deref(cmt_base, _, _) => {
            opt_loan_path(cmt_base).map(
                |&lp| @LpExtend(lp, cmt.mutbl, LpDeref))
        }

        mc::cat_interior(cmt_base, ik) => {
            opt_loan_path(cmt_base).map(
                |&lp| @LpExtend(lp, cmt.mutbl, LpInterior(ik)))
        }

        mc::cat_stack_upvar(cmt_base) |
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
// - `RESTR_MUTATE`: The lvalue may not be modified and mutable pointers to
//                   the value cannot be created.
// - `RESTR_FREEZE`: Immutable pointers to the value cannot be created.
// - `RESTR_ALIAS`: The lvalue may not be aliased in any way.
//
// In addition, no value which is restricted may be moved. Therefore,
// restrictions are meaningful even if the RestrictionSet is empty,
// because the restriction against moves is implied.

pub struct Restriction {
    loan_path: @LoanPath,
    set: RestrictionSet
}

pub struct RestrictionSet {
    bits: u32
}

pub static RESTR_EMPTY: RestrictionSet  = RestrictionSet {bits: 0b000};
pub static RESTR_MUTATE: RestrictionSet = RestrictionSet {bits: 0b001};
pub static RESTR_FREEZE: RestrictionSet = RestrictionSet {bits: 0b010};
pub static RESTR_ALIAS: RestrictionSet  = RestrictionSet {bits: 0b100};

pub impl RestrictionSet {
    fn intersects(&self, restr: RestrictionSet) -> bool {
        (self.bits & restr.bits) != 0
    }

    fn contains_all(&self, restr: RestrictionSet) -> bool {
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
    scope: ast::node_id,
    freeze: Option<DynaFreezeKind> // Some() if we should freeze box at runtime
}

pub type root_map = @mut HashMap<root_map_key, RootInfo>;

pub fn root_map() -> root_map {
    return @mut HashMap::new();
}

pub enum DynaFreezeKind {
    DynaImm,
    DynaMut
}

impl ToStr for DynaFreezeKind {
    fn to_str(&self) -> ~str {
        match *self {
            DynaMut => ~"mutable",
            DynaImm => ~"immutable"
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Errors

// Errors that can occur
#[deriving(Eq)]
pub enum bckerr_code {
    err_mutbl(ast::mutability),
    err_out_of_root_scope(ty::Region, ty::Region), // superscope, subscope
    err_out_of_scope(ty::Region, ty::Region), // superscope, subscope
    err_freeze_aliasable_const
}

// Combination of an error code and the categorization of the expression
// that caused it
#[deriving(Eq)]
pub struct BckError {
    span: span,
    cmt: mc::cmt,
    code: bckerr_code
}

pub enum AliasableViolationKind {
    MutabilityViolation,
    BorrowViolation
}

///////////////////////////////////////////////////////////////////////////
// Misc

pub impl BorrowckCtxt {
    fn is_subregion_of(&self, r_sub: ty::Region, r_sup: ty::Region) -> bool {
        self.tcx.region_maps.is_subregion_of(r_sub, r_sup)
    }

    fn is_subscope_of(&self, r_sub: ast::node_id, r_sup: ast::node_id) -> bool {
        self.tcx.region_maps.is_subscope_of(r_sub, r_sup)
    }

    fn cat_expr(&self, expr: @ast::expr) -> mc::cmt {
        mc::cat_expr(self.tcx, self.method_map, expr)
    }

    fn cat_expr_unadjusted(&self, expr: @ast::expr) -> mc::cmt {
        mc::cat_expr_unadjusted(self.tcx, self.method_map, expr)
    }

    fn cat_expr_autoderefd(&self, expr: @ast::expr,
                           adj: @ty::AutoAdjustment) -> mc::cmt {
        match *adj {
            ty::AutoAddEnv(*) => {
                // no autoderefs
                mc::cat_expr_unadjusted(self.tcx, self.method_map, expr)
            }

            ty::AutoDerefRef(
                ty::AutoDerefRef {
                    autoderefs: autoderefs, _}) => {
                mc::cat_expr_autoderefd(self.tcx, self.method_map, expr,
                                        autoderefs)
            }
        }
    }

    fn cat_def(&self,
               id: ast::node_id,
               span: span,
               ty: ty::t,
               def: ast::def) -> mc::cmt {
        mc::cat_def(self.tcx, self.method_map, id, span, ty, def)
    }

    fn cat_discr(&self, cmt: mc::cmt, match_id: ast::node_id) -> mc::cmt {
        @mc::cmt_ {cat:mc::cat_discr(cmt, match_id),
                   mutbl:cmt.mutbl.inherit(),
                   ..*cmt}
    }

    fn mc_ctxt(&self) -> mc::mem_categorization_ctxt {
        mc::mem_categorization_ctxt {tcx: self.tcx,
                                 method_map: self.method_map}
    }

    fn cat_pattern(&self,
                   cmt: mc::cmt,
                   pat: @ast::pat,
                   op: &fn(mc::cmt, @ast::pat)) {
        let mc = self.mc_ctxt();
        mc.cat_pattern(cmt, pat, op);
    }

    fn report(&self, err: BckError) {
        self.span_err(
            err.span,
            self.bckerr_to_str(err));
        self.note_and_explain_bckerr(err);
    }

    fn span_err(&self, s: span, m: &str) {
        self.tcx.sess.span_err(s, m);
    }

    fn span_note(&self, s: span, m: &str) {
        self.tcx.sess.span_note(s, m);
    }

    fn bckerr_to_str(&self, err: BckError) -> ~str {
        match err.code {
            err_mutbl(lk) => {
                fmt!("cannot borrow %s %s as %s",
                     err.cmt.mutbl.to_user_str(),
                     self.cmt_to_str(err.cmt),
                     self.mut_to_str(lk))
            }
            err_out_of_root_scope(*) => {
                fmt!("cannot root managed value long enough")
            }
            err_out_of_scope(*) => {
                fmt!("borrowed value does not live long enough")
            }
            err_freeze_aliasable_const => {
                // Means that the user borrowed a ~T or enum value
                // residing in &const or @const pointer.  Terrible
                // error message, but then &const and @const are
                // supposed to be going away.
                fmt!("unsafe borrow of aliasable, const value")
            }
        }
    }

    fn report_aliasability_violation(&self,
                                     span: span,
                                     kind: AliasableViolationKind,
                                     cause: mc::AliasableReason) {
        let prefix = match kind {
            MutabilityViolation => "cannot assign to an `&mut`",
            BorrowViolation => "cannot borrow an `&mut`"
        };

        match cause {
            mc::AliasableOther => {
                self.tcx.sess.span_err(
                    span,
                    fmt!("%s in an aliasable location", prefix));
            }
            mc::AliasableManaged(ast::m_mutbl) => {
                // FIXME(#6269) reborrow @mut to &mut
                self.tcx.sess.span_err(
                    span,
                    fmt!("%s in a `@mut` pointer; \
                          try borrowing as `&mut` first", prefix));
            }
            mc::AliasableManaged(m) => {
                self.tcx.sess.span_err(
                    span,
                    fmt!("%s in a `@%s` pointer; \
                          try an `@mut` instead",
                         prefix,
                         self.mut_to_keyword(m)));
            }
            mc::AliasableBorrowed(m) => {
                self.tcx.sess.span_err(
                    span,
                    fmt!("%s in a `&%s` pointer; \
                          try an `&mut` instead",
                         prefix,
                         self.mut_to_keyword(m)));
            }
        }
    }

    fn note_and_explain_bckerr(&self, err: BckError) {
        let code = err.code;
        match code {
            err_mutbl(*) | err_freeze_aliasable_const(*) => {}

            err_out_of_root_scope(super_scope, sub_scope) => {
                note_and_explain_region(
                    self.tcx,
                    ~"managed value would have to be rooted for ",
                    sub_scope,
                    ~"...");
                note_and_explain_region(
                    self.tcx,
                    ~"...but can only be rooted for ",
                    super_scope,
                    ~"");
            }

            err_out_of_scope(super_scope, sub_scope) => {
                note_and_explain_region(
                    self.tcx,
                    ~"borrowed pointer must be valid for ",
                    sub_scope,
                    ~"...");
                note_and_explain_region(
                    self.tcx,
                    ~"...but borrowed value is only valid for ",
                    super_scope,
                    ~"");
          }
        }
    }

    fn append_loan_path_to_str_from_interior(&self,
                                             loan_path: &LoanPath,
                                             out: &mut ~str) {
        match *loan_path {
            LpExtend(_, _, LpDeref) => {
                str::push_char(out, '(');
                self.append_loan_path_to_str(loan_path, out);
                str::push_char(out, ')');
            }
            LpExtend(_, _, LpInterior(_)) |
            LpVar(_) => {
                self.append_loan_path_to_str(loan_path, out);
            }
        }
    }

    fn append_loan_path_to_str(&self, loan_path: &LoanPath, out: &mut ~str) {
        match *loan_path {
            LpVar(id) => {
                match self.tcx.items.find(&id) {
                    Some(&ast_map::node_local(ident)) => {
                        str::push_str(out, *self.tcx.sess.intr().get(ident));
                    }
                    r => {
                        self.tcx.sess.bug(
                            fmt!("Loan path LpVar(%?) maps to %?, not local",
                                 id, r));
                    }
                }
            }

            LpExtend(lp_base, _, LpInterior(mc::interior_field(fld))) => {
                self.append_loan_path_to_str_from_interior(lp_base, out);
                str::push_char(out, '.');
                str::push_str(out, *self.tcx.sess.intr().get(fld));
            }

            LpExtend(lp_base, _, LpInterior(mc::interior_index(*))) => {
                self.append_loan_path_to_str_from_interior(lp_base, out);
                str::push_str(out, "[]");
            }

            LpExtend(lp_base, _, LpInterior(mc::interior_tuple)) |
            LpExtend(lp_base, _, LpInterior(mc::interior_anon_field)) |
            LpExtend(lp_base, _, LpInterior(mc::interior_variant(_))) => {
                self.append_loan_path_to_str_from_interior(lp_base, out);
                str::push_str(out, ".(tuple)");
            }

            LpExtend(lp_base, _, LpDeref) => {
                str::push_char(out, '*');
                self.append_loan_path_to_str(lp_base, out);
            }
        }
    }

    fn loan_path_to_str(&self, loan_path: &LoanPath) -> ~str {
        let mut result = ~"";
        self.append_loan_path_to_str(loan_path, &mut result);
        result
    }

    fn cmt_to_str(&self, cmt: mc::cmt) -> ~str {
        let mc = &mc::mem_categorization_ctxt {tcx: self.tcx,
                                               method_map: self.method_map};
        mc.cmt_to_str(cmt)
    }

    fn mut_to_str(&self, mutbl: ast::mutability) -> ~str {
        let mc = &mc::mem_categorization_ctxt {tcx: self.tcx,
                                               method_map: self.method_map};
        mc.mut_to_str(mutbl)
    }

    fn mut_to_keyword(&self, mutbl: ast::mutability) -> &'static str {
        match mutbl {
            ast::m_imm => "",
            ast::m_const => "const",
            ast::m_mutbl => "mut"
        }
    }
}

impl DataFlowOperator for LoanDataFlowOperator {
    #[inline(always)]
    fn initial_value(&self) -> bool {
        false // no loans in scope by default
    }

    #[inline(always)]
    fn join(&self, succ: uint, pred: uint) -> uint {
        succ | pred // loans from both preds are in scope
    }

    #[inline(always)]
    fn walk_closures(&self) -> bool {
        true
    }
}

impl Repr for Loan {
    fn repr(&self, tcx: ty::ctxt) -> ~str {
        fmt!("Loan_%?(%s, %?, %?-%?, %s)",
             self.index,
             self.loan_path.repr(tcx),
             self.mutbl,
             self.gen_scope,
             self.kill_scope,
             self.restrictions.repr(tcx))
    }
}

impl Repr for Restriction {
    fn repr(&self, tcx: ty::ctxt) -> ~str {
        fmt!("Restriction(%s, %x)",
             self.loan_path.repr(tcx),
             self.set.bits as uint)
    }
}

impl Repr for LoanPath {
    fn repr(&self, tcx: ty::ctxt) -> ~str {
        match self {
            &LpVar(id) => {
                fmt!("$(%?)", id)
            }

            &LpExtend(lp, _, LpDeref) => {
                fmt!("%s.*", lp.repr(tcx))
            }

            &LpExtend(lp, _, LpInterior(ref interior)) => {
                fmt!("%s.%s", lp.repr(tcx), interior.repr(tcx))
            }
        }
    }
}
