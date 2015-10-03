// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Compilation of match statements
//!
//! I will endeavor to explain the code as best I can.  I have only a loose
//! understanding of some parts of it.
//!
//! ## Matching
//!
//! The basic state of the code is maintained in an array `m` of `Match`
//! objects.  Each `Match` describes some list of patterns, all of which must
//! match against the current list of values.  If those patterns match, then
//! the arm listed in the match is the correct arm.  A given arm may have
//! multiple corresponding match entries, one for each alternative that
//! remains.  As we proceed these sets of matches are adjusted by the various
//! `enter_XXX()` functions, each of which adjusts the set of options given
//! some information about the value which has been matched.
//!
//! So, initially, there is one value and N matches, each of which have one
//! constituent pattern.  N here is usually the number of arms but may be
//! greater, if some arms have multiple alternatives.  For example, here:
//!
//!     enum Foo { A, B(int), C(usize, usize) }
//!     match foo {
//!         A => ...,
//!         B(x) => ...,
//!         C(1, 2) => ...,
//!         C(_) => ...
//!     }
//!
//! The value would be `foo`.  There would be four matches, each of which
//! contains one pattern (and, in one case, a guard).  We could collect the
//! various options and then compile the code for the case where `foo` is an
//! `A`, a `B`, and a `C`.  When we generate the code for `C`, we would (1)
//! drop the two matches that do not match a `C` and (2) expand the other two
//! into two patterns each.  In the first case, the two patterns would be `1`
//! and `2`, and the in the second case the _ pattern would be expanded into
//! `_` and `_`.  The two values are of course the arguments to `C`.
//!
//! Here is a quick guide to the various functions:
//!
//! - `compile_submatch()`: The main workhouse.  It takes a list of values and
//!   a list of matches and finds the various possibilities that could occur.
//!
//! - `enter_XXX()`: modifies the list of matches based on some information
//!   about the value that has been matched.  For example,
//!   `enter_rec_or_struct()` adjusts the values given that a record or struct
//!   has been matched.  This is an infallible pattern, so *all* of the matches
//!   must be either wildcards or record/struct patterns.  `enter_opt()`
//!   handles the fallible cases, and it is correspondingly more complex.
//!
//! ## Bindings
//!
//! We store information about the bound variables for each arm as part of the
//! per-arm `ArmData` struct.  There is a mapping from identifiers to
//! `BindingInfo` structs.  These structs contain the mode/id/type of the
//! binding, but they also contain an LLVM value which points at an alloca
//! called `llmatch`. For by value bindings that are Copy, we also create
//! an extra alloca that we copy the matched value to so that any changes
//! we do to our copy is not reflected in the original and vice-versa.
//! We don't do this if it's a move since the original value can't be used
//! and thus allowing us to cheat in not creating an extra alloca.
//!
//! The `llmatch` binding always stores a pointer into the value being matched
//! which points at the data for the binding.  If the value being matched has
//! type `T`, then, `llmatch` will point at an alloca of type `T*` (and hence
//! `llmatch` has type `T**`).  So, if you have a pattern like:
//!
//!    let a: A = ...;
//!    let b: B = ...;
//!    match (a, b) { (ref c, d) => { ... } }
//!
//! For `c` and `d`, we would generate allocas of type `C*` and `D*`
//! respectively.  These are called the `llmatch`.  As we match, when we come
//! up against an identifier, we store the current pointer into the
//! corresponding alloca.
//!
//! Once a pattern is completely matched, and assuming that there is no guard
//! pattern, we will branch to a block that leads to the body itself.  For any
//! by-value bindings, this block will first load the ptr from `llmatch` (the
//! one of type `D*`) and then load a second time to get the actual value (the
//! one of type `D`). For by ref bindings, the value of the local variable is
//! simply the first alloca.
//!
//! So, for the example above, we would generate a setup kind of like this:
//!
//!        +-------+
//!        | Entry |
//!        +-------+
//!            |
//!        +--------------------------------------------+
//!        | llmatch_c = (addr of first half of tuple)  |
//!        | llmatch_d = (addr of second half of tuple) |
//!        +--------------------------------------------+
//!            |
//!        +--------------------------------------+
//!        | *llbinding_d = **llmatch_d           |
//!        +--------------------------------------+
//!
//! If there is a guard, the situation is slightly different, because we must
//! execute the guard code.  Moreover, we need to do so once for each of the
//! alternatives that lead to the arm, because if the guard fails, they may
//! have different points from which to continue the search. Therefore, in that
//! case, we generate code that looks more like:
//!
//!        +-------+
//!        | Entry |
//!        +-------+
//!            |
//!        +-------------------------------------------+
//!        | llmatch_c = (addr of first half of tuple) |
//!        | llmatch_d = (addr of first half of tuple) |
//!        +-------------------------------------------+
//!            |
//!        +-------------------------------------------------+
//!        | *llbinding_d = **llmatch_d                      |
//!        | check condition                                 |
//!        | if false { goto next case }                     |
//!        | if true { goto body }                           |
//!        +-------------------------------------------------+
//!
//! The handling for the cleanups is a bit... sensitive.  Basically, the body
//! is the one that invokes `add_clean()` for each binding.  During the guard
//! evaluation, we add temporary cleanups and revoke them after the guard is
//! evaluated (it could fail, after all). Note that guards and moves are
//! just plain incompatible.
//!
//! Some relevant helper functions that manage bindings:
//! - `create_bindings_map()`
//! - `insert_lllocals()`
//!
//!
//! ## Notes on vector pattern matching.
//!
//! Vector pattern matching is surprisingly tricky. The problem is that
//! the structure of the vector isn't fully known, and slice matches
//! can be done on subparts of it.
//!
//! The way that vector pattern matches are dealt with, then, is as
//! follows. First, we make the actual condition associated with a
//! vector pattern simply a vector length comparison. So the pattern
//! [1, .. x] gets the condition "vec len >= 1", and the pattern
//! [.. x] gets the condition "vec len >= 0". The problem here is that
//! having the condition "vec len >= 1" hold clearly does not mean that
//! only a pattern that has exactly that condition will match. This
//! means that it may well be the case that a condition holds, but none
//! of the patterns matching that condition match; to deal with this,
//! when doing vector length matches, we have match failures proceed to
//! the next condition to check.
//!
//! There are a couple more subtleties to deal with. While the "actual"
//! condition associated with vector length tests is simply a test on
//! the vector length, the actual vec_len Opt entry contains more
//! information used to restrict which matches are associated with it.
//! So that all matches in a submatch are matching against the same
//! values from inside the vector, they are split up by how many
//! elements they match at the front and at the back of the vector. In
//! order to make sure that arms are properly checked in order, even
//! with the overmatching conditions, each vec_len Opt entry is
//! associated with a range of matches.
//! Consider the following:
//!
//!   match &[1, 2, 3] {
//!       [1, 1, .. _] => 0,
//!       [1, 2, 2, .. _] => 1,
//!       [1, 2, 3, .. _] => 2,
//!       [1, 2, .. _] => 3,
//!       _ => 4
//!   }
//! The proper arm to match is arm 2, but arms 0 and 3 both have the
//! condition "len >= 2". If arm 3 was lumped in with arm 0, then the
//! wrong branch would be taken. Instead, vec_len Opts are associated
//! with a contiguous range of matches that have the same "shape".
//! This is sort of ugly and requires a bunch of special handling of
//! vec_len options.

pub use self::BranchKind::*;
pub use self::OptResult::*;
pub use self::TransBindingMode::*;
use self::Opt::*;
use self::FailureHandler::*;

use llvm::{ValueRef, BasicBlockRef};
use middle::check_match::StaticInliner;
use middle::check_match;
use middle::const_eval;
use middle::def::{self, DefMap};
use middle::def_id::DefId;
use middle::expr_use_visitor as euv;
use middle::infer;
use middle::lang_items::StrEqFnLangItem;
use middle::mem_categorization as mc;
use middle::mem_categorization::Categorization;
use middle::pat_util::*;
use trans::adt;
use trans::base::*;
use trans::build::{AddCase, And, Br, CondBr, GEPi, InBoundsGEP, Load, PointerCast};
use trans::build::{Not, Store, Sub, add_comment};
use trans::build;
use trans::callee;
use trans::cleanup::{self, CleanupMethods, DropHintMethods};
use trans::common::*;
use trans::consts;
use trans::datum::*;
use trans::debuginfo::{self, DebugLoc, ToDebugLoc};
use trans::expr::{self, Dest};
use trans::monomorphize;
use trans::tvec;
use trans::type_of;
use middle::ty::{self, Ty};
use session::config::NoDebugInfo;
use util::common::indenter;
use util::nodemap::FnvHashMap;
use util::ppaux;

use std;
use std::cmp::Ordering;
use std::fmt;
use std::rc::Rc;
use rustc_front::hir;
use syntax::ast::{self, DUMMY_NODE_ID, NodeId};
use syntax::codemap::Span;
use rustc_front::fold::Folder;
use syntax::ptr::P;

#[derive(Copy, Clone, Debug)]
struct ConstantExpr<'a>(&'a hir::Expr);

impl<'a> ConstantExpr<'a> {
    fn eq(self, other: ConstantExpr<'a>, tcx: &ty::ctxt) -> bool {
        match const_eval::compare_lit_exprs(tcx, self.0, other.0) {
            Some(result) => result == Ordering::Equal,
            None => panic!("compare_list_exprs: type mismatch"),
        }
    }
}

// An option identifying a branch (either a literal, an enum variant or a range)
#[derive(Debug)]
enum Opt<'a, 'tcx> {
    ConstantValue(ConstantExpr<'a>, DebugLoc),
    ConstantRange(ConstantExpr<'a>, ConstantExpr<'a>, DebugLoc),
    Variant(ty::Disr, Rc<adt::Repr<'tcx>>, DefId, DebugLoc),
    SliceLengthEqual(usize, DebugLoc),
    SliceLengthGreaterOrEqual(/* prefix length */ usize,
                              /* suffix length */ usize,
                              DebugLoc),
}

impl<'a, 'tcx> Opt<'a, 'tcx> {
    fn eq(&self, other: &Opt<'a, 'tcx>, tcx: &ty::ctxt<'tcx>) -> bool {
        match (self, other) {
            (&ConstantValue(a, _), &ConstantValue(b, _)) => a.eq(b, tcx),
            (&ConstantRange(a1, a2, _), &ConstantRange(b1, b2, _)) => {
                a1.eq(b1, tcx) && a2.eq(b2, tcx)
            }
            (&Variant(a_disr, ref a_repr, a_def, _),
             &Variant(b_disr, ref b_repr, b_def, _)) => {
                a_disr == b_disr && *a_repr == *b_repr && a_def == b_def
            }
            (&SliceLengthEqual(a, _), &SliceLengthEqual(b, _)) => a == b,
            (&SliceLengthGreaterOrEqual(a1, a2, _),
             &SliceLengthGreaterOrEqual(b1, b2, _)) => {
                a1 == b1 && a2 == b2
            }
            _ => false
        }
    }

    fn trans<'blk>(&self, mut bcx: Block<'blk, 'tcx>) -> OptResult<'blk, 'tcx> {
        use trans::consts::TrueConst::Yes;
        let _icx = push_ctxt("match::trans_opt");
        let ccx = bcx.ccx();
        match *self {
            ConstantValue(ConstantExpr(lit_expr), _) => {
                let lit_ty = bcx.tcx().node_id_to_type(lit_expr.id);
                let expr = consts::const_expr(ccx, &*lit_expr, bcx.fcx.param_substs, None, Yes);
                let llval = match expr {
                    Ok((llval, _)) => llval,
                    Err(err) => bcx.ccx().sess().span_fatal(lit_expr.span, &err.description()),
                };
                let lit_datum = immediate_rvalue(llval, lit_ty);
                let lit_datum = unpack_datum!(bcx, lit_datum.to_appropriate_datum(bcx));
                SingleResult(Result::new(bcx, lit_datum.val))
            }
            ConstantRange(ConstantExpr(ref l1), ConstantExpr(ref l2), _) => {
                let l1 = match consts::const_expr(ccx, &**l1, bcx.fcx.param_substs, None, Yes) {
                    Ok((l1, _)) => l1,
                    Err(err) => bcx.ccx().sess().span_fatal(l1.span, &err.description()),
                };
                let l2 = match consts::const_expr(ccx, &**l2, bcx.fcx.param_substs, None, Yes) {
                    Ok((l2, _)) => l2,
                    Err(err) => bcx.ccx().sess().span_fatal(l2.span, &err.description()),
                };
                RangeResult(Result::new(bcx, l1), Result::new(bcx, l2))
            }
            Variant(disr_val, ref repr, _, _) => {
                adt::trans_case(bcx, &**repr, disr_val)
            }
            SliceLengthEqual(length, _) => {
                SingleResult(Result::new(bcx, C_uint(ccx, length)))
            }
            SliceLengthGreaterOrEqual(prefix, suffix, _) => {
                LowerBound(Result::new(bcx, C_uint(ccx, prefix + suffix)))
            }
        }
    }

    fn debug_loc(&self) -> DebugLoc {
        match *self {
            ConstantValue(_,debug_loc)                 |
            ConstantRange(_, _, debug_loc)             |
            Variant(_, _, _, debug_loc)                |
            SliceLengthEqual(_, debug_loc)             |
            SliceLengthGreaterOrEqual(_, _, debug_loc) => debug_loc
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum BranchKind {
    NoBranch,
    Single,
    Switch,
    Compare,
    CompareSliceLength
}

pub enum OptResult<'blk, 'tcx: 'blk> {
    SingleResult(Result<'blk, 'tcx>),
    RangeResult(Result<'blk, 'tcx>, Result<'blk, 'tcx>),
    LowerBound(Result<'blk, 'tcx>)
}

#[derive(Clone, Copy, PartialEq)]
pub enum TransBindingMode {
    /// By-value binding for a copy type: copies from matched data
    /// into a fresh LLVM alloca.
    TrByCopy(/* llbinding */ ValueRef),

    /// By-value binding for a non-copy type where we copy into a
    /// fresh LLVM alloca; this most accurately reflects the language
    /// semantics (e.g. it properly handles overwrites of the matched
    /// input), but potentially injects an unwanted copy.
    TrByMoveIntoCopy(/* llbinding */ ValueRef),

    /// Binding a non-copy type by reference under the hood; this is
    /// a codegen optimization to avoid unnecessary memory traffic.
    TrByMoveRef,

    /// By-ref binding exposed in the original source input.
    TrByRef,
}

impl TransBindingMode {
    /// if binding by making a fresh copy; returns the alloca that it
    /// will copy into; otherwise None.
    fn alloca_if_copy(&self) -> Option<ValueRef> {
        match *self {
            TrByCopy(llbinding) | TrByMoveIntoCopy(llbinding) => Some(llbinding),
            TrByMoveRef | TrByRef => None,
        }
    }
}

/// Information about a pattern binding:
/// - `llmatch` is a pointer to a stack slot.  The stack slot contains a
///   pointer into the value being matched.  Hence, llmatch has type `T**`
///   where `T` is the value being matched.
/// - `trmode` is the trans binding mode
/// - `id` is the node id of the binding
/// - `ty` is the Rust type of the binding
#[derive(Clone, Copy)]
pub struct BindingInfo<'tcx> {
    pub llmatch: ValueRef,
    pub trmode: TransBindingMode,
    pub id: ast::NodeId,
    pub span: Span,
    pub ty: Ty<'tcx>,
}

type BindingsMap<'tcx> = FnvHashMap<ast::Name, BindingInfo<'tcx>>;

struct ArmData<'p, 'blk, 'tcx: 'blk> {
    bodycx: Block<'blk, 'tcx>,
    arm: &'p hir::Arm,
    bindings_map: BindingsMap<'tcx>
}

/// Info about Match.
/// If all `pats` are matched then arm `data` will be executed.
/// As we proceed `bound_ptrs` are filled with pointers to values to be bound,
/// these pointers are stored in llmatch variables just before executing `data` arm.
struct Match<'a, 'p: 'a, 'blk: 'a, 'tcx: 'blk> {
    pats: Vec<&'p hir::Pat>,
    data: &'a ArmData<'p, 'blk, 'tcx>,
    bound_ptrs: Vec<(ast::Name, ValueRef)>,
    // Thread along renamings done by the check_match::StaticInliner, so we can
    // map back to original NodeIds
    pat_renaming_map: Option<&'a FnvHashMap<(NodeId, Span), NodeId>>
}

impl<'a, 'p, 'blk, 'tcx> fmt::Debug for Match<'a, 'p, 'blk, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if ppaux::verbose() {
            // for many programs, this just take too long to serialize
            write!(f, "{:?}", self.pats)
        } else {
            write!(f, "{} pats", self.pats.len())
        }
    }
}

fn has_nested_bindings(m: &[Match], col: usize) -> bool {
    for br in m {
        match br.pats[col].node {
            hir::PatIdent(_, _, Some(_)) => return true,
            _ => ()
        }
    }
    return false;
}

// As noted in `fn match_datum`, we should eventually pass around a
// `Datum<Lvalue>` for the `val`; but until we get to that point, this
// `MatchInput` struct will serve -- it has everything `Datum<Lvalue>`
// does except for the type field.
#[derive(Copy, Clone)]
pub struct MatchInput { val: ValueRef, lval: Lvalue }

impl<'tcx> Datum<'tcx, Lvalue> {
    pub fn match_input(&self) -> MatchInput {
        MatchInput {
            val: self.val,
            lval: self.kind,
        }
    }
}

impl MatchInput {
    fn from_val(val: ValueRef) -> MatchInput {
        MatchInput {
            val: val,
            lval: Lvalue::new("MatchInput::from_val"),
        }
    }

    fn to_datum<'tcx>(self, ty: Ty<'tcx>) -> Datum<'tcx, Lvalue> {
        Datum::new(self.val, ty, self.lval)
    }
}

fn expand_nested_bindings<'a, 'p, 'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                              m: &[Match<'a, 'p, 'blk, 'tcx>],
                                              col: usize,
                                              val: MatchInput)
                                              -> Vec<Match<'a, 'p, 'blk, 'tcx>> {
    debug!("expand_nested_bindings(bcx={}, m={:?}, col={}, val={})",
           bcx.to_str(),
           m,
           col,
           bcx.val_to_string(val.val));
    let _indenter = indenter();

    m.iter().map(|br| {
        let mut bound_ptrs = br.bound_ptrs.clone();
        let mut pat = br.pats[col];
        loop {
            pat = match pat.node {
                hir::PatIdent(_, ref path, Some(ref inner)) => {
                    bound_ptrs.push((path.node.name, val.val));
                    &**inner
                },
                _ => break
            }
        }

        let mut pats = br.pats.clone();
        pats[col] = pat;
        Match {
            pats: pats,
            data: &*br.data,
            bound_ptrs: bound_ptrs,
            pat_renaming_map: br.pat_renaming_map,
        }
    }).collect()
}

fn enter_match<'a, 'b, 'p, 'blk, 'tcx, F>(bcx: Block<'blk, 'tcx>,
                                          dm: &DefMap,
                                          m: &[Match<'a, 'p, 'blk, 'tcx>],
                                          col: usize,
                                          val: MatchInput,
                                          mut e: F)
                                          -> Vec<Match<'a, 'p, 'blk, 'tcx>> where
    F: FnMut(&[&'p hir::Pat]) -> Option<Vec<&'p hir::Pat>>,
{
    debug!("enter_match(bcx={}, m={:?}, col={}, val={})",
           bcx.to_str(),
           m,
           col,
           bcx.val_to_string(val.val));
    let _indenter = indenter();

    m.iter().filter_map(|br| {
        e(&br.pats).map(|pats| {
            let this = br.pats[col];
            let mut bound_ptrs = br.bound_ptrs.clone();
            match this.node {
                hir::PatIdent(_, ref path, None) => {
                    if pat_is_binding(dm, &*this) {
                        bound_ptrs.push((path.node.name, val.val));
                    }
                }
                hir::PatVec(ref before, Some(ref slice), ref after) => {
                    if let hir::PatIdent(_, ref path, None) = slice.node {
                        let subslice_val = bind_subslice_pat(
                            bcx, this.id, val,
                            before.len(), after.len());
                        bound_ptrs.push((path.node.name, subslice_val));
                    }
                }
                _ => {}
            }
            Match {
                pats: pats,
                data: br.data,
                bound_ptrs: bound_ptrs,
                pat_renaming_map: br.pat_renaming_map,
            }
        })
    }).collect()
}

fn enter_default<'a, 'p, 'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                     dm: &DefMap,
                                     m: &[Match<'a, 'p, 'blk, 'tcx>],
                                     col: usize,
                                     val: MatchInput)
                                     -> Vec<Match<'a, 'p, 'blk, 'tcx>> {
    debug!("enter_default(bcx={}, m={:?}, col={}, val={})",
           bcx.to_str(),
           m,
           col,
           bcx.val_to_string(val.val));
    let _indenter = indenter();

    // Collect all of the matches that can match against anything.
    enter_match(bcx, dm, m, col, val, |pats| {
        if pat_is_binding_or_wild(dm, &*pats[col]) {
            let mut r = pats[..col].to_vec();
            r.push_all(&pats[col + 1..]);
            Some(r)
        } else {
            None
        }
    })
}

// <pcwalton> nmatsakis: what does enter_opt do?
// <pcwalton> in trans/match
// <pcwalton> trans/match.rs is like stumbling around in a dark cave
// <nmatsakis> pcwalton: the enter family of functions adjust the set of
//             patterns as needed
// <nmatsakis> yeah, at some point I kind of achieved some level of
//             understanding
// <nmatsakis> anyhow, they adjust the patterns given that something of that
//             kind has been found
// <nmatsakis> pcwalton: ok, right, so enter_XXX() adjusts the patterns, as I
//             said
// <nmatsakis> enter_match() kind of embodies the generic code
// <nmatsakis> it is provided with a function that tests each pattern to see
//             if it might possibly apply and so forth
// <nmatsakis> so, if you have a pattern like {a: _, b: _, _} and one like _
// <nmatsakis> then _ would be expanded to (_, _)
// <nmatsakis> one spot for each of the sub-patterns
// <nmatsakis> enter_opt() is one of the more complex; it covers the fallible
//             cases
// <nmatsakis> enter_rec_or_struct() or enter_tuple() are simpler, since they
//             are infallible patterns
// <nmatsakis> so all patterns must either be records (resp. tuples) or
//             wildcards

/// The above is now outdated in that enter_match() now takes a function that
/// takes the complete row of patterns rather than just the first one.
/// Also, most of the enter_() family functions have been unified with
/// the check_match specialization step.
fn enter_opt<'a, 'p, 'blk, 'tcx>(
             bcx: Block<'blk, 'tcx>,
             _: ast::NodeId,
             dm: &DefMap,
             m: &[Match<'a, 'p, 'blk, 'tcx>],
             opt: &Opt,
             col: usize,
             variant_size: usize,
             val: MatchInput)
             -> Vec<Match<'a, 'p, 'blk, 'tcx>> {
    debug!("enter_opt(bcx={}, m={:?}, opt={:?}, col={}, val={})",
           bcx.to_str(),
           m,
           *opt,
           col,
           bcx.val_to_string(val.val));
    let _indenter = indenter();

    let ctor = match opt {
        &ConstantValue(ConstantExpr(expr), _) => check_match::ConstantValue(
            const_eval::eval_const_expr(bcx.tcx(), &*expr)
        ),
        &ConstantRange(ConstantExpr(lo), ConstantExpr(hi), _) => check_match::ConstantRange(
            const_eval::eval_const_expr(bcx.tcx(), &*lo),
            const_eval::eval_const_expr(bcx.tcx(), &*hi)
        ),
        &SliceLengthEqual(n, _) =>
            check_match::Slice(n),
        &SliceLengthGreaterOrEqual(before, after, _) =>
            check_match::SliceWithSubslice(before, after),
        &Variant(_, _, def_id, _) =>
            check_match::Constructor::Variant(def_id)
    };

    let param_env = bcx.tcx().empty_parameter_environment();
    let mcx = check_match::MatchCheckCtxt {
        tcx: bcx.tcx(),
        param_env: param_env,
    };
    enter_match(bcx, dm, m, col, val, |pats|
        check_match::specialize(&mcx, &pats[..], &ctor, col, variant_size)
    )
}

// Returns the options in one column of matches. An option is something that
// needs to be conditionally matched at runtime; for example, the discriminant
// on a set of enum variants or a literal.
fn get_branches<'a, 'p, 'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                    m: &[Match<'a, 'p, 'blk, 'tcx>],
                                    col: usize)
                                    -> Vec<Opt<'p, 'tcx>> {
    let tcx = bcx.tcx();

    let mut found: Vec<Opt> = vec![];
    for br in m {
        let cur = br.pats[col];
        let debug_loc = match br.pat_renaming_map {
            Some(pat_renaming_map) => {
                match pat_renaming_map.get(&(cur.id, cur.span)) {
                    Some(&id) => DebugLoc::At(id, cur.span),
                    None => DebugLoc::At(cur.id, cur.span),
                }
            }
            None => DebugLoc::None
        };

        let opt = match cur.node {
            hir::PatLit(ref l) => {
                ConstantValue(ConstantExpr(&**l), debug_loc)
            }
            hir::PatIdent(..) | hir::PatEnum(..) | hir::PatStruct(..) => {
                // This is either an enum variant or a variable binding.
                let opt_def = tcx.def_map.borrow().get(&cur.id).map(|d| d.full_def());
                match opt_def {
                    Some(def::DefVariant(enum_id, var_id, _)) => {
                        let variant = tcx.lookup_adt_def(enum_id).variant_with_id(var_id);
                        Variant(variant.disr_val,
                                adt::represent_node(bcx, cur.id),
                                var_id,
                                debug_loc)
                    }
                    _ => continue
                }
            }
            hir::PatRange(ref l1, ref l2) => {
                ConstantRange(ConstantExpr(&**l1), ConstantExpr(&**l2), debug_loc)
            }
            hir::PatVec(ref before, None, ref after) => {
                SliceLengthEqual(before.len() + after.len(), debug_loc)
            }
            hir::PatVec(ref before, Some(_), ref after) => {
                SliceLengthGreaterOrEqual(before.len(), after.len(), debug_loc)
            }
            _ => continue
        };

        if !found.iter().any(|x| x.eq(&opt, tcx)) {
            found.push(opt);
        }
    }
    found
}

struct ExtractedBlock<'blk, 'tcx: 'blk> {
    vals: Vec<ValueRef>,
    bcx: Block<'blk, 'tcx>,
}

fn extract_variant_args<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                    repr: &adt::Repr<'tcx>,
                                    disr_val: ty::Disr,
                                    val: MatchInput)
                                    -> ExtractedBlock<'blk, 'tcx> {
    let _icx = push_ctxt("match::extract_variant_args");
    let args = (0..adt::num_args(repr, disr_val)).map(|i| {
        adt::trans_field_ptr(bcx, repr, val.val, disr_val, i)
    }).collect();

    ExtractedBlock { vals: args, bcx: bcx }
}

/// Helper for converting from the ValueRef that we pass around in the match code, which is always
/// an lvalue, into a Datum. Eventually we should just pass around a Datum and be done with it.
fn match_datum<'tcx>(val: MatchInput, left_ty: Ty<'tcx>) -> Datum<'tcx, Lvalue> {
    val.to_datum(left_ty)
}

fn bind_subslice_pat(bcx: Block,
                     pat_id: ast::NodeId,
                     val: MatchInput,
                     offset_left: usize,
                     offset_right: usize) -> ValueRef {
    let _icx = push_ctxt("match::bind_subslice_pat");
    let vec_ty = node_id_type(bcx, pat_id);
    let vec_ty_contents = match vec_ty.sty {
        ty::TyBox(ty) => ty,
        ty::TyRef(_, mt) | ty::TyRawPtr(mt) => mt.ty,
        _ => vec_ty
    };
    let unit_ty = vec_ty_contents.sequence_element_type(bcx.tcx());
    let vec_datum = match_datum(val, vec_ty);
    let (base, len) = vec_datum.get_vec_base_and_len(bcx);

    let slice_begin = InBoundsGEP(bcx, base, &[C_uint(bcx.ccx(), offset_left)]);
    let slice_len_offset = C_uint(bcx.ccx(), offset_left + offset_right);
    let slice_len = Sub(bcx, len, slice_len_offset, DebugLoc::None);
    let slice_ty = bcx.tcx().mk_imm_ref(bcx.tcx().mk_region(ty::ReStatic),
                                         bcx.tcx().mk_slice(unit_ty));
    let scratch = rvalue_scratch_datum(bcx, slice_ty, "");
    Store(bcx, slice_begin, expr::get_dataptr(bcx, scratch.val));
    Store(bcx, slice_len, expr::get_meta(bcx, scratch.val));
    scratch.val
}

fn extract_vec_elems<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                 left_ty: Ty<'tcx>,
                                 before: usize,
                                 after: usize,
                                 val: MatchInput)
                                 -> ExtractedBlock<'blk, 'tcx> {
    let _icx = push_ctxt("match::extract_vec_elems");
    let vec_datum = match_datum(val, left_ty);
    let (base, len) = vec_datum.get_vec_base_and_len(bcx);
    let mut elems = vec![];
    elems.extend((0..before).map(|i| GEPi(bcx, base, &[i])));
    elems.extend((0..after).rev().map(|i| {
        InBoundsGEP(bcx, base, &[
            Sub(bcx, len, C_uint(bcx.ccx(), i + 1), DebugLoc::None)
        ])
    }));
    ExtractedBlock { vals: elems, bcx: bcx }
}

// Macro for deciding whether any of the remaining matches fit a given kind of
// pattern.  Note that, because the macro is well-typed, either ALL of the
// matches should fit that sort of pattern or NONE (however, some of the
// matches may be wildcards like _ or identifiers).
macro_rules! any_pat {
    ($m:expr, $col:expr, $pattern:pat) => (
        ($m).iter().any(|br| {
            match br.pats[$col].node {
                $pattern => true,
                _ => false
            }
        })
    )
}

fn any_uniq_pat(m: &[Match], col: usize) -> bool {
    any_pat!(m, col, hir::PatBox(_))
}

fn any_region_pat(m: &[Match], col: usize) -> bool {
    any_pat!(m, col, hir::PatRegion(..))
}

fn any_irrefutable_adt_pat(tcx: &ty::ctxt, m: &[Match], col: usize) -> bool {
    m.iter().any(|br| {
        let pat = br.pats[col];
        match pat.node {
            hir::PatTup(_) => true,
            hir::PatStruct(..) => {
                match tcx.def_map.borrow().get(&pat.id).map(|d| d.full_def()) {
                    Some(def::DefVariant(..)) => false,
                    _ => true,
                }
            }
            hir::PatEnum(..) | hir::PatIdent(_, _, None) => {
                match tcx.def_map.borrow().get(&pat.id).map(|d| d.full_def()) {
                    Some(def::DefStruct(..)) => true,
                    _ => false
                }
            }
            _ => false
        }
    })
}

/// What to do when the pattern match fails.
enum FailureHandler {
    Infallible,
    JumpToBasicBlock(BasicBlockRef),
    Unreachable
}

impl FailureHandler {
    fn is_fallible(&self) -> bool {
        match *self {
            Infallible => false,
            _ => true
        }
    }

    fn is_infallible(&self) -> bool {
        !self.is_fallible()
    }

    fn handle_fail(&self, bcx: Block) {
        match *self {
            Infallible =>
                panic!("attempted to panic in a non-panicking panic handler!"),
            JumpToBasicBlock(basic_block) =>
                Br(bcx, basic_block, DebugLoc::None),
            Unreachable =>
                build::Unreachable(bcx)
        }
    }
}

fn pick_column_to_specialize(def_map: &DefMap, m: &[Match]) -> Option<usize> {
    fn pat_score(def_map: &DefMap, pat: &hir::Pat) -> usize {
        match pat.node {
            hir::PatIdent(_, _, Some(ref inner)) => pat_score(def_map, &**inner),
            _ if pat_is_refutable(def_map, pat) => 1,
            _ => 0
        }
    }

    let column_score = |m: &[Match], col: usize| -> usize {
        let total_score = m.iter()
            .map(|row| row.pats[col])
            .map(|pat| pat_score(def_map, pat))
            .sum();

        // Irrefutable columns always go first, they'd only be duplicated in the branches.
        if total_score == 0 {
            std::usize::MAX
        } else {
            total_score
        }
    };

    let column_contains_any_nonwild_patterns = |&col: &usize| -> bool {
        m.iter().any(|row| match row.pats[col].node {
            hir::PatWild(_) => false,
            _ => true
        })
    };

    (0..m[0].pats.len())
        .filter(column_contains_any_nonwild_patterns)
        .map(|col| (col, column_score(m, col)))
        .max_by(|&(_, score)| score)
        .map(|(col, _)| col)
}

// Compiles a comparison between two things.
fn compare_values<'blk, 'tcx>(cx: Block<'blk, 'tcx>,
                              lhs: ValueRef,
                              rhs: ValueRef,
                              rhs_t: Ty<'tcx>,
                              debug_loc: DebugLoc)
                              -> Result<'blk, 'tcx> {
    fn compare_str<'blk, 'tcx>(cx: Block<'blk, 'tcx>,
                               lhs_data: ValueRef,
                               lhs_len: ValueRef,
                               rhs_data: ValueRef,
                               rhs_len: ValueRef,
                               rhs_t: Ty<'tcx>,
                               debug_loc: DebugLoc)
                               -> Result<'blk, 'tcx> {
        let did = langcall(cx,
                           None,
                           &format!("comparison of `{}`", rhs_t),
                           StrEqFnLangItem);
        callee::trans_lang_call(cx, did, &[lhs_data, lhs_len, rhs_data, rhs_len], None, debug_loc)
    }

    let _icx = push_ctxt("compare_values");
    if rhs_t.is_scalar() {
        let cmp = compare_scalar_types(cx, lhs, rhs, rhs_t, hir::BiEq, debug_loc);
        return Result::new(cx, cmp);
    }

    match rhs_t.sty {
        ty::TyRef(_, mt) => match mt.ty.sty {
            ty::TyStr => {
                let lhs_data = Load(cx, expr::get_dataptr(cx, lhs));
                let lhs_len = Load(cx, expr::get_meta(cx, lhs));
                let rhs_data = Load(cx, expr::get_dataptr(cx, rhs));
                let rhs_len = Load(cx, expr::get_meta(cx, rhs));
                compare_str(cx, lhs_data, lhs_len, rhs_data, rhs_len, rhs_t, debug_loc)
            }
            ty::TyArray(ty, _) | ty::TySlice(ty) => match ty.sty {
                ty::TyUint(ast::TyU8) => {
                    // NOTE: cast &[u8] and &[u8; N] to &str and abuse the str_eq lang item,
                    // which calls memcmp().
                    let pat_len = val_ty(rhs).element_type().array_length();
                    let ty_str_slice = cx.tcx().mk_static_str();

                    let rhs_data = GEPi(cx, rhs, &[0, 0]);
                    let rhs_len = C_uint(cx.ccx(), pat_len);

                    let lhs_data;
                    let lhs_len;
                    if val_ty(lhs) == val_ty(rhs) {
                        // Both the discriminant and the pattern are thin pointers
                        lhs_data = GEPi(cx, lhs, &[0, 0]);
                        lhs_len = C_uint(cx.ccx(), pat_len);
                    } else {
                        // The discriminant is a fat pointer
                        let llty_str_slice = type_of::type_of(cx.ccx(), ty_str_slice).ptr_to();
                        let lhs_str = PointerCast(cx, lhs, llty_str_slice);
                        lhs_data = Load(cx, expr::get_dataptr(cx, lhs_str));
                        lhs_len = Load(cx, expr::get_meta(cx, lhs_str));
                    }

                    compare_str(cx, lhs_data, lhs_len, rhs_data, rhs_len, rhs_t, debug_loc)
                },
                _ => cx.sess().bug("only byte strings supported in compare_values"),
            },
            _ => cx.sess().bug("only string and byte strings supported in compare_values"),
        },
        _ => cx.sess().bug("only scalars, byte strings, and strings supported in compare_values"),
    }
}

/// For each binding in `data.bindings_map`, adds an appropriate entry into the `fcx.lllocals` map
fn insert_lllocals<'blk, 'tcx>(mut bcx: Block<'blk, 'tcx>,
                               bindings_map: &BindingsMap<'tcx>,
                               cs: Option<cleanup::ScopeId>)
                               -> Block<'blk, 'tcx> {
    for (&name, &binding_info) in bindings_map {
        let (llval, aliases_other_state) = match binding_info.trmode {
            // By value mut binding for a copy type: load from the ptr
            // into the matched value and copy to our alloca
            TrByCopy(llbinding) |
            TrByMoveIntoCopy(llbinding) => {
                let llval = Load(bcx, binding_info.llmatch);
                let lvalue = match binding_info.trmode {
                    TrByCopy(..) =>
                        Lvalue::new("_match::insert_lllocals"),
                    TrByMoveIntoCopy(..) => {
                        // match_input moves from the input into a
                        // separate stack slot.
                        //
                        // E.g. consider moving the value `D(A)` out
                        // of the tuple `(D(A), D(B))` and into the
                        // local variable `x` via the pattern `(x,_)`,
                        // leaving the remainder of the tuple `(_,
                        // D(B))` still to be dropped in the future.
                        //
                        // Thus, here we must zero the place that we
                        // are moving *from*, because we do not yet
                        // track drop flags for a fragmented parent
                        // match input expression.
                        //
                        // Longer term we will be able to map the move
                        // into `(x, _)` up to the parent path that
                        // owns the whole tuple, and mark the
                        // corresponding stack-local drop-flag
                        // tracking the first component of the tuple.
                        let hint_kind = HintKind::ZeroAndMaintain;
                        Lvalue::new_with_hint("_match::insert_lllocals (match_input)",
                                              bcx, binding_info.id, hint_kind)
                    }
                    _ => unreachable!(),
                };
                let datum = Datum::new(llval, binding_info.ty, lvalue);
                call_lifetime_start(bcx, llbinding);
                bcx = datum.store_to(bcx, llbinding);
                if let Some(cs) = cs {
                    bcx.fcx.schedule_lifetime_end(cs, llbinding);
                }

                (llbinding, false)
            },

            // By value move bindings: load from the ptr into the matched value
            TrByMoveRef => (Load(bcx, binding_info.llmatch), true),

            // By ref binding: use the ptr into the matched value
            TrByRef => (binding_info.llmatch, true),
        };


        // A local that aliases some other state must be zeroed, since
        // the other state (e.g. some parent data that we matched
        // into) will still have its subcomponents (such as this
        // local) destructed at the end of the parent's scope. Longer
        // term, we will properly map such parents to the set of
        // unique drop flags for its fragments.
        let hint_kind = if aliases_other_state {
            HintKind::ZeroAndMaintain
        } else {
            HintKind::DontZeroJustUse
        };
        let lvalue = Lvalue::new_with_hint("_match::insert_lllocals (local)",
                                           bcx,
                                           binding_info.id,
                                           hint_kind);
        let datum = Datum::new(llval, binding_info.ty, lvalue);
        if let Some(cs) = cs {
            let opt_datum = lvalue.dropflag_hint(bcx);
            bcx.fcx.schedule_lifetime_end(cs, binding_info.llmatch);
            bcx.fcx.schedule_drop_and_fill_mem(cs, llval, binding_info.ty, opt_datum);
        }

        debug!("binding {} to {}", binding_info.id, bcx.val_to_string(llval));
        bcx.fcx.lllocals.borrow_mut().insert(binding_info.id, datum);
        debuginfo::create_match_binding_metadata(bcx, name, binding_info);
    }
    bcx
}

fn compile_guard<'a, 'p, 'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                     guard_expr: &hir::Expr,
                                     data: &ArmData<'p, 'blk, 'tcx>,
                                     m: &[Match<'a, 'p, 'blk, 'tcx>],
                                     vals: &[MatchInput],
                                     chk: &FailureHandler,
                                     has_genuine_default: bool)
                                     -> Block<'blk, 'tcx> {
    debug!("compile_guard(bcx={}, guard_expr={:?}, m={:?}, vals=[{}])",
           bcx.to_str(),
           guard_expr,
           m,
           vals.iter().map(|v| bcx.val_to_string(v.val)).collect::<Vec<_>>().join(", "));
    let _indenter = indenter();

    let mut bcx = insert_lllocals(bcx, &data.bindings_map, None);

    let val = unpack_datum!(bcx, expr::trans(bcx, guard_expr));
    let val = val.to_llbool(bcx);

    for (_, &binding_info) in &data.bindings_map {
        if let Some(llbinding) = binding_info.trmode.alloca_if_copy() {
            call_lifetime_end(bcx, llbinding)
        }
    }

    for (_, &binding_info) in &data.bindings_map {
        bcx.fcx.lllocals.borrow_mut().remove(&binding_info.id);
    }

    with_cond(bcx, Not(bcx, val, guard_expr.debug_loc()), |bcx| {
        for (_, &binding_info) in &data.bindings_map {
            call_lifetime_end(bcx, binding_info.llmatch);
        }
        match chk {
            // If the default arm is the only one left, move on to the next
            // condition explicitly rather than (possibly) falling back to
            // the default arm.
            &JumpToBasicBlock(_) if m.len() == 1 && has_genuine_default => {
                chk.handle_fail(bcx);
            }
            _ => {
                compile_submatch(bcx, m, vals, chk, has_genuine_default);
            }
        };
        bcx
    })
}

fn compile_submatch<'a, 'p, 'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                        m: &[Match<'a, 'p, 'blk, 'tcx>],
                                        vals: &[MatchInput],
                                        chk: &FailureHandler,
                                        has_genuine_default: bool) {
    debug!("compile_submatch(bcx={}, m={:?}, vals=[{}])",
           bcx.to_str(),
           m,
           vals.iter().map(|v| bcx.val_to_string(v.val)).collect::<Vec<_>>().join(", "));
    let _indenter = indenter();
    let _icx = push_ctxt("match::compile_submatch");
    let mut bcx = bcx;
    if m.is_empty() {
        if chk.is_fallible() {
            chk.handle_fail(bcx);
        }
        return;
    }

    let tcx = bcx.tcx();
    let def_map = &tcx.def_map;
    match pick_column_to_specialize(def_map, m) {
        Some(col) => {
            let val = vals[col];
            if has_nested_bindings(m, col) {
                let expanded = expand_nested_bindings(bcx, m, col, val);
                compile_submatch_continue(bcx,
                                          &expanded[..],
                                          vals,
                                          chk,
                                          col,
                                          val,
                                          has_genuine_default)
            } else {
                compile_submatch_continue(bcx, m, vals, chk, col, val, has_genuine_default)
            }
        }
        None => {
            let data = &m[0].data;
            for &(ref ident, ref value_ptr) in &m[0].bound_ptrs {
                let binfo = *data.bindings_map.get(ident).unwrap();
                call_lifetime_start(bcx, binfo.llmatch);
                if binfo.trmode == TrByRef && type_is_fat_ptr(bcx.tcx(), binfo.ty) {
                    expr::copy_fat_ptr(bcx, *value_ptr, binfo.llmatch);
                }
                else {
                    Store(bcx, *value_ptr, binfo.llmatch);
                }
            }
            match data.arm.guard {
                Some(ref guard_expr) => {
                    bcx = compile_guard(bcx,
                                        &**guard_expr,
                                        m[0].data,
                                        &m[1..m.len()],
                                        vals,
                                        chk,
                                        has_genuine_default);
                }
                _ => ()
            }
            Br(bcx, data.bodycx.llbb, DebugLoc::None);
        }
    }
}

fn compile_submatch_continue<'a, 'p, 'blk, 'tcx>(mut bcx: Block<'blk, 'tcx>,
                                                 m: &[Match<'a, 'p, 'blk, 'tcx>],
                                                 vals: &[MatchInput],
                                                 chk: &FailureHandler,
                                                 col: usize,
                                                 val: MatchInput,
                                                 has_genuine_default: bool) {
    let fcx = bcx.fcx;
    let tcx = bcx.tcx();
    let dm = &tcx.def_map;

    let mut vals_left = vals[0..col].to_vec();
    vals_left.push_all(&vals[col + 1..]);
    let ccx = bcx.fcx.ccx;

    // Find a real id (we're adding placeholder wildcard patterns, but
    // each column is guaranteed to have at least one real pattern)
    let pat_id = m.iter().map(|br| br.pats[col].id)
                         .find(|&id| id != DUMMY_NODE_ID)
                         .unwrap_or(DUMMY_NODE_ID);

    let left_ty = if pat_id == DUMMY_NODE_ID {
        tcx.mk_nil()
    } else {
        node_id_type(bcx, pat_id)
    };

    let mcx = check_match::MatchCheckCtxt {
        tcx: bcx.tcx(),
        param_env: bcx.tcx().empty_parameter_environment(),
    };
    let adt_vals = if any_irrefutable_adt_pat(bcx.tcx(), m, col) {
        let repr = adt::represent_type(bcx.ccx(), left_ty);
        let arg_count = adt::num_args(&*repr, 0);
        let (arg_count, struct_val) = if type_is_sized(bcx.tcx(), left_ty) {
            (arg_count, val.val)
        } else {
            // For an unsized ADT (i.e. DST struct), we need to treat
            // the last field specially: instead of simply passing a
            // ValueRef pointing to that field, as with all the others,
            // we skip it and instead construct a 'fat ptr' below.
            (arg_count - 1, Load(bcx, expr::get_dataptr(bcx, val.val)))
        };
        let mut field_vals: Vec<ValueRef> = (0..arg_count).map(|ix|
            adt::trans_field_ptr(bcx, &*repr, struct_val, 0, ix)
        ).collect();

        match left_ty.sty {
            ty::TyStruct(def, substs) if !type_is_sized(bcx.tcx(), left_ty) => {
                // The last field is technically unsized but
                // since we can only ever match that field behind
                // a reference we construct a fat ptr here.
                let unsized_ty = def.struct_variant().fields.last().map(|field| {
                    monomorphize::field_ty(bcx.tcx(), substs, field)
                }).unwrap();
                let scratch = alloc_ty(bcx, unsized_ty, "__struct_field_fat_ptr");
                let data = adt::trans_field_ptr(bcx, &*repr, struct_val, 0, arg_count);
                let len = Load(bcx, expr::get_meta(bcx, val.val));
                Store(bcx, data, expr::get_dataptr(bcx, scratch));
                Store(bcx, len, expr::get_meta(bcx, scratch));
                field_vals.push(scratch);
            }
            _ => {}
        }
        Some(field_vals)
    } else if any_uniq_pat(m, col) || any_region_pat(m, col) {
        Some(vec!(Load(bcx, val.val)))
    } else {
        match left_ty.sty {
            ty::TyArray(_, n) => {
                let args = extract_vec_elems(bcx, left_ty, n, 0, val);
                Some(args.vals)
            }
            _ => None
        }
    };
    match adt_vals {
        Some(field_vals) => {
            let pats = enter_match(bcx, dm, m, col, val, |pats|
                check_match::specialize(&mcx, pats,
                                        &check_match::Single, col,
                                        field_vals.len())
            );
            let mut vals: Vec<_> = field_vals.into_iter()
                .map(|v|MatchInput::from_val(v))
                .collect();
            vals.push_all(&vals_left);
            compile_submatch(bcx, &pats, &vals, chk, has_genuine_default);
            return;
        }
        _ => ()
    }

    // Decide what kind of branch we need
    let opts = get_branches(bcx, m, col);
    debug!("options={:?}", opts);
    let mut kind = NoBranch;
    let mut test_val = val.val;
    debug!("test_val={}", bcx.val_to_string(test_val));
    if !opts.is_empty() {
        match opts[0] {
            ConstantValue(..) | ConstantRange(..) => {
                test_val = load_if_immediate(bcx, val.val, left_ty);
                kind = if left_ty.is_integral() {
                    Switch
                } else {
                    Compare
                };
            }
            Variant(_, ref repr, _, _) => {
                let (the_kind, val_opt) = adt::trans_switch(bcx, &**repr, val.val);
                kind = the_kind;
                if let Some(tval) = val_opt { test_val = tval; }
            }
            SliceLengthEqual(..) | SliceLengthGreaterOrEqual(..) => {
                let (_, len) = tvec::get_base_and_len(bcx, val.val, left_ty);
                test_val = len;
                kind = Switch;
            }
        }
    }
    for o in &opts {
        match *o {
            ConstantRange(..) => { kind = Compare; break },
            SliceLengthGreaterOrEqual(..) => { kind = CompareSliceLength; break },
            _ => ()
        }
    }
    let else_cx = match kind {
        NoBranch | Single => bcx,
        _ => bcx.fcx.new_temp_block("match_else")
    };
    let sw = if kind == Switch {
        build::Switch(bcx, test_val, else_cx.llbb, opts.len())
    } else {
        C_int(ccx, 0) // Placeholder for when not using a switch
    };

    let defaults = enter_default(else_cx, dm, m, col, val);
    let exhaustive = chk.is_infallible() && defaults.is_empty();
    let len = opts.len();

    if exhaustive && kind == Switch {
        build::Unreachable(else_cx);
    }

    // Compile subtrees for each option
    for (i, opt) in opts.iter().enumerate() {
        // In some cases of range and vector pattern matching, we need to
        // override the failure case so that instead of failing, it proceeds
        // to try more matching. branch_chk, then, is the proper failure case
        // for the current conditional branch.
        let mut branch_chk = None;
        let mut opt_cx = else_cx;
        let debug_loc = opt.debug_loc();

        if kind == Switch || !exhaustive || i + 1 < len {
            opt_cx = bcx.fcx.new_temp_block("match_case");
            match kind {
                Single => Br(bcx, opt_cx.llbb, debug_loc),
                Switch => {
                    match opt.trans(bcx) {
                        SingleResult(r) => {
                            AddCase(sw, r.val, opt_cx.llbb);
                            bcx = r.bcx;
                        }
                        _ => {
                            bcx.sess().bug(
                                "in compile_submatch, expected \
                                 opt.trans() to return a SingleResult")
                        }
                    }
                }
                Compare | CompareSliceLength => {
                    let t = if kind == Compare {
                        left_ty
                    } else {
                        tcx.types.usize // vector length
                    };
                    let Result { bcx: after_cx, val: matches } = {
                        match opt.trans(bcx) {
                            SingleResult(Result { bcx, val }) => {
                                compare_values(bcx, test_val, val, t, debug_loc)
                            }
                            RangeResult(Result { val: vbegin, .. },
                                        Result { bcx, val: vend }) => {
                                let llge = compare_scalar_types(bcx, test_val, vbegin,
                                                                t, hir::BiGe, debug_loc);
                                let llle = compare_scalar_types(bcx, test_val, vend,
                                                                t, hir::BiLe, debug_loc);
                                Result::new(bcx, And(bcx, llge, llle, DebugLoc::None))
                            }
                            LowerBound(Result { bcx, val }) => {
                                Result::new(bcx, compare_scalar_types(bcx, test_val,
                                                                      val, t, hir::BiGe,
                                                                      debug_loc))
                            }
                        }
                    };
                    bcx = fcx.new_temp_block("compare_next");

                    // If none of the sub-cases match, and the current condition
                    // is guarded or has multiple patterns, move on to the next
                    // condition, if there is any, rather than falling back to
                    // the default.
                    let guarded = m[i].data.arm.guard.is_some();
                    let multi_pats = m[i].pats.len() > 1;
                    if i + 1 < len && (guarded || multi_pats || kind == CompareSliceLength) {
                        branch_chk = Some(JumpToBasicBlock(bcx.llbb));
                    }
                    CondBr(after_cx, matches, opt_cx.llbb, bcx.llbb, debug_loc);
                }
                _ => ()
            }
        } else if kind == Compare || kind == CompareSliceLength {
            Br(bcx, else_cx.llbb, debug_loc);
        }

        let mut size = 0;
        let mut unpacked = Vec::new();
        match *opt {
            Variant(disr_val, ref repr, _, _) => {
                let ExtractedBlock {vals: argvals, bcx: new_bcx} =
                    extract_variant_args(opt_cx, &**repr, disr_val, val);
                size = argvals.len();
                unpacked = argvals;
                opt_cx = new_bcx;
            }
            SliceLengthEqual(len, _) => {
                let args = extract_vec_elems(opt_cx, left_ty, len, 0, val);
                size = args.vals.len();
                unpacked = args.vals.clone();
                opt_cx = args.bcx;
            }
            SliceLengthGreaterOrEqual(before, after, _) => {
                let args = extract_vec_elems(opt_cx, left_ty, before, after, val);
                size = args.vals.len();
                unpacked = args.vals.clone();
                opt_cx = args.bcx;
            }
            ConstantValue(..) | ConstantRange(..) => ()
        }
        let opt_ms = enter_opt(opt_cx, pat_id, dm, m, opt, col, size, val);
        let mut opt_vals: Vec<_> = unpacked.into_iter()
            .map(|v|MatchInput::from_val(v))
            .collect();
        opt_vals.push_all(&vals_left[..]);
        compile_submatch(opt_cx,
                         &opt_ms[..],
                         &opt_vals[..],
                         branch_chk.as_ref().unwrap_or(chk),
                         has_genuine_default);
    }

    // Compile the fall-through case, if any
    if !exhaustive && kind != Single {
        if kind == Compare || kind == CompareSliceLength {
            Br(bcx, else_cx.llbb, DebugLoc::None);
        }
        match chk {
            // If there is only one default arm left, move on to the next
            // condition explicitly rather than (eventually) falling back to
            // the last default arm.
            &JumpToBasicBlock(_) if defaults.len() == 1 && has_genuine_default => {
                chk.handle_fail(else_cx);
            }
            _ => {
                compile_submatch(else_cx,
                                 &defaults[..],
                                 &vals_left[..],
                                 chk,
                                 has_genuine_default);
            }
        }
    }
}

pub fn trans_match<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                               match_expr: &hir::Expr,
                               discr_expr: &hir::Expr,
                               arms: &[hir::Arm],
                               dest: Dest)
                               -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("match::trans_match");
    trans_match_inner(bcx, match_expr.id, discr_expr, arms, dest)
}

/// Checks whether the binding in `discr` is assigned to anywhere in the expression `body`
fn is_discr_reassigned(bcx: Block, discr: &hir::Expr, body: &hir::Expr) -> bool {
    let (vid, field) = match discr.node {
        hir::ExprPath(..) => match bcx.def(discr.id) {
            def::DefLocal(_, vid) | def::DefUpvar(_, vid, _, _) => (vid, None),
            _ => return false
        },
        hir::ExprField(ref base, field) => {
            let vid = match bcx.tcx().def_map.borrow().get(&base.id).map(|d| d.full_def()) {
                Some(def::DefLocal(_, vid)) | Some(def::DefUpvar(_, vid, _, _)) => vid,
                _ => return false
            };
            (vid, Some(mc::NamedField(field.node)))
        },
        hir::ExprTupField(ref base, field) => {
            let vid = match bcx.tcx().def_map.borrow().get(&base.id).map(|d| d.full_def()) {
                Some(def::DefLocal(_, vid)) | Some(def::DefUpvar(_, vid, _, _)) => vid,
                _ => return false
            };
            (vid, Some(mc::PositionalField(field.node)))
        },
        _ => return false
    };

    let mut rc = ReassignmentChecker {
        node: vid,
        field: field,
        reassigned: false
    };
    {
        let infcx = infer::normalizing_infer_ctxt(bcx.tcx(), &bcx.tcx().tables);
        let mut visitor = euv::ExprUseVisitor::new(&mut rc, &infcx);
        visitor.walk_expr(body);
    }
    rc.reassigned
}

struct ReassignmentChecker {
    node: ast::NodeId,
    field: Option<mc::FieldName>,
    reassigned: bool
}

// Determine if the expression we're matching on is reassigned to within
// the body of the match's arm.
// We only care for the `mutate` callback since this check only matters
// for cases where the matched value is moved.
impl<'tcx> euv::Delegate<'tcx> for ReassignmentChecker {
    fn consume(&mut self, _: ast::NodeId, _: Span, _: mc::cmt, _: euv::ConsumeMode) {}
    fn matched_pat(&mut self, _: &hir::Pat, _: mc::cmt, _: euv::MatchMode) {}
    fn consume_pat(&mut self, _: &hir::Pat, _: mc::cmt, _: euv::ConsumeMode) {}
    fn borrow(&mut self, _: ast::NodeId, _: Span, _: mc::cmt, _: ty::Region,
              _: ty::BorrowKind, _: euv::LoanCause) {}
    fn decl_without_init(&mut self, _: ast::NodeId, _: Span) {}

    fn mutate(&mut self, _: ast::NodeId, _: Span, cmt: mc::cmt, _: euv::MutateMode) {
        match cmt.cat {
            Categorization::Upvar(mc::Upvar { id: ty::UpvarId { var_id: vid, .. }, .. }) |
            Categorization::Local(vid) => self.reassigned |= self.node == vid,
            Categorization::Interior(ref base_cmt, mc::InteriorField(field)) => {
                match base_cmt.cat {
                    Categorization::Upvar(mc::Upvar { id: ty::UpvarId { var_id: vid, .. }, .. }) |
                    Categorization::Local(vid) => {
                        self.reassigned |= self.node == vid &&
                            (self.field.is_none() || Some(field) == self.field)
                    },
                    _ => {}
                }
            },
            _ => {}
        }
    }
}

fn create_bindings_map<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, pat: &hir::Pat,
                                   discr: &hir::Expr, body: &hir::Expr)
                                   -> BindingsMap<'tcx> {
    // Create the bindings map, which is a mapping from each binding name
    // to an alloca() that will be the value for that local variable.
    // Note that we use the names because each binding will have many ids
    // from the various alternatives.
    let ccx = bcx.ccx();
    let tcx = bcx.tcx();
    let reassigned = is_discr_reassigned(bcx, discr, body);
    let mut bindings_map = FnvHashMap();
    pat_bindings(&tcx.def_map, &*pat, |bm, p_id, span, path1| {
        let name = path1.node;
        let variable_ty = node_id_type(bcx, p_id);
        let llvariable_ty = type_of::type_of(ccx, variable_ty);
        let tcx = bcx.tcx();
        let param_env = tcx.empty_parameter_environment();

        let llmatch;
        let trmode;
        let moves_by_default = variable_ty.moves_by_default(&param_env, span);
        match bm {
            hir::BindByValue(_) if !moves_by_default || reassigned =>
            {
                llmatch = alloca(bcx, llvariable_ty.ptr_to(), "__llmatch");
                let llcopy = alloca(bcx, llvariable_ty, &bcx.name(name));
                trmode = if moves_by_default {
                    TrByMoveIntoCopy(llcopy)
                } else {
                    TrByCopy(llcopy)
                };
            }
            hir::BindByValue(_) => {
                // in this case, the final type of the variable will be T,
                // but during matching we need to store a *T as explained
                // above
                llmatch = alloca(bcx, llvariable_ty.ptr_to(), &bcx.name(name));
                trmode = TrByMoveRef;
            }
            hir::BindByRef(_) => {
                llmatch = alloca(bcx, llvariable_ty, &bcx.name(name));
                trmode = TrByRef;
            }
        };
        bindings_map.insert(name, BindingInfo {
            llmatch: llmatch,
            trmode: trmode,
            id: p_id,
            span: span,
            ty: variable_ty
        });
    });
    return bindings_map;
}

fn trans_match_inner<'blk, 'tcx>(scope_cx: Block<'blk, 'tcx>,
                                 match_id: ast::NodeId,
                                 discr_expr: &hir::Expr,
                                 arms: &[hir::Arm],
                                 dest: Dest) -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("match::trans_match_inner");
    let fcx = scope_cx.fcx;
    let mut bcx = scope_cx;
    let tcx = bcx.tcx();

    let discr_datum = unpack_datum!(bcx, expr::trans_to_lvalue(bcx, discr_expr,
                                                               "match"));
    if bcx.unreachable.get() {
        return bcx;
    }

    let t = node_id_type(bcx, discr_expr.id);
    let chk = if t.is_empty(tcx) {
        Unreachable
    } else {
        Infallible
    };

    let arm_datas: Vec<ArmData> = arms.iter().map(|arm| ArmData {
        bodycx: fcx.new_id_block("case_body", arm.body.id),
        arm: arm,
        bindings_map: create_bindings_map(bcx, &*arm.pats[0], discr_expr, &*arm.body)
    }).collect();

    let mut pat_renaming_map = if scope_cx.sess().opts.debuginfo != NoDebugInfo {
        Some(FnvHashMap())
    } else {
        None
    };

    let arm_pats: Vec<Vec<P<hir::Pat>>> = {
        let mut static_inliner = StaticInliner::new(scope_cx.tcx(),
                                                    pat_renaming_map.as_mut());
        arm_datas.iter().map(|arm_data| {
            arm_data.arm.pats.iter().map(|p| static_inliner.fold_pat((*p).clone())).collect()
        }).collect()
    };

    let mut matches = Vec::new();
    for (arm_data, pats) in arm_datas.iter().zip(&arm_pats) {
        matches.extend(pats.iter().map(|p| Match {
            pats: vec![&**p],
            data: arm_data,
            bound_ptrs: Vec::new(),
            pat_renaming_map: pat_renaming_map.as_ref()
        }));
    }

    // `compile_submatch` works one column of arm patterns a time and
    // then peels that column off. So as we progress, it may become
    // impossible to tell whether we have a genuine default arm, i.e.
    // `_ => foo` or not. Sometimes it is important to know that in order
    // to decide whether moving on to the next condition or falling back
    // to the default arm.
    let has_default = arms.last().map_or(false, |arm| {
        arm.pats.len() == 1
        && arm.pats.last().unwrap().node == hir::PatWild(hir::PatWildSingle)
    });

    compile_submatch(bcx, &matches[..], &[discr_datum.match_input()], &chk, has_default);

    let mut arm_cxs = Vec::new();
    for arm_data in &arm_datas {
        let mut bcx = arm_data.bodycx;

        // insert bindings into the lllocals map and add cleanups
        let cs = fcx.push_custom_cleanup_scope();
        bcx = insert_lllocals(bcx, &arm_data.bindings_map, Some(cleanup::CustomScope(cs)));
        bcx = expr::trans_into(bcx, &*arm_data.arm.body, dest);
        bcx = fcx.pop_and_trans_custom_cleanup_scope(bcx, cs);
        arm_cxs.push(bcx);
    }

    bcx = scope_cx.fcx.join_blocks(match_id, &arm_cxs[..]);
    return bcx;
}

/// Generates code for a local variable declaration like `let <pat>;` or `let <pat> =
/// <opt_init_expr>`.
pub fn store_local<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                               local: &hir::Local)
                               -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("match::store_local");
    let mut bcx = bcx;
    let tcx = bcx.tcx();
    let pat = &*local.pat;

    fn create_dummy_locals<'blk, 'tcx>(mut bcx: Block<'blk, 'tcx>,
                                       pat: &hir::Pat)
                                       -> Block<'blk, 'tcx> {
        let _icx = push_ctxt("create_dummy_locals");
        // create dummy memory for the variables if we have no
        // value to store into them immediately
        let tcx = bcx.tcx();
        pat_bindings(&tcx.def_map, pat, |_, p_id, _, path1| {
            let scope = cleanup::var_scope(tcx, p_id);
            bcx = mk_binding_alloca(
                bcx, p_id, path1.node, scope, (),
                "_match::store_local::create_dummy_locals",
                |(), bcx, Datum { val: llval, ty, kind }| {
                    // Dummy-locals start out uninitialized, so set their
                    // drop-flag hints (if any) to "moved."
                    if let Some(hint) = kind.dropflag_hint(bcx) {
                        let moved_hint = adt::DTOR_MOVED_HINT;
                        debug!("store moved_hint={} for hint={:?}, uninitialized dummy",
                               moved_hint, hint);
                        Store(bcx, C_u8(bcx.fcx.ccx, moved_hint), hint.to_value().value());
                    }

                    if kind.drop_flag_info.must_zero() {
                        // if no drop-flag hint, or the hint requires
                        // we maintain the embedded drop-flag, then
                        // mark embedded drop-flag(s) as moved
                        // (i.e. "already dropped").
                        drop_done_fill_mem(bcx, llval, ty);
                    }
                    bcx
                });
        });
        bcx
    }

    match local.init {
        Some(ref init_expr) => {
            // Optimize the "let x = expr" case. This just writes
            // the result of evaluating `expr` directly into the alloca
            // for `x`. Often the general path results in similar or the
            // same code post-optimization, but not always. In particular,
            // in unsafe code, you can have expressions like
            //
            //    let x = intrinsics::uninit();
            //
            // In such cases, the more general path is unsafe, because
            // it assumes it is matching against a valid value.
            match simple_name(pat) {
                Some(name) => {
                    let var_scope = cleanup::var_scope(tcx, local.id);
                    return mk_binding_alloca(
                        bcx, pat.id, name, var_scope, (),
                        "_match::store_local",
                        |(), bcx, Datum { val: v, .. }| expr::trans_into(bcx, &**init_expr,
                                                                         expr::SaveIn(v)));
                }

                None => {}
            }

            // General path.
            let init_datum =
                unpack_datum!(bcx, expr::trans_to_lvalue(bcx, &**init_expr, "let"));
            if bcx.sess().asm_comments() {
                add_comment(bcx, "creating zeroable ref llval");
            }
            let var_scope = cleanup::var_scope(tcx, local.id);
            bind_irrefutable_pat(bcx, pat, init_datum.match_input(), var_scope)
        }
        None => {
            create_dummy_locals(bcx, pat)
        }
    }
}

fn mk_binding_alloca<'blk, 'tcx, A, F>(bcx: Block<'blk, 'tcx>,
                                       p_id: ast::NodeId,
                                       name: ast::Name,
                                       cleanup_scope: cleanup::ScopeId,
                                       arg: A,
                                       caller_name: &'static str,
                                       populate: F)
                                       -> Block<'blk, 'tcx> where
    F: FnOnce(A, Block<'blk, 'tcx>, Datum<'tcx, Lvalue>) -> Block<'blk, 'tcx>,
{
    let var_ty = node_id_type(bcx, p_id);

    // Allocate memory on stack for the binding.
    let llval = alloc_ty(bcx, var_ty, &bcx.name(name));
    let lvalue = Lvalue::new_with_hint(caller_name, bcx, p_id, HintKind::DontZeroJustUse);
    let datum = Datum::new(llval, var_ty, lvalue);

    // Subtle: be sure that we *populate* the memory *before*
    // we schedule the cleanup.
    call_lifetime_start(bcx, llval);
    let bcx = populate(arg, bcx, datum);
    bcx.fcx.schedule_lifetime_end(cleanup_scope, llval);
    bcx.fcx.schedule_drop_mem(cleanup_scope, llval, var_ty, lvalue.dropflag_hint(bcx));

    // Now that memory is initialized and has cleanup scheduled,
    // insert datum into the local variable map.
    bcx.fcx.lllocals.borrow_mut().insert(p_id, datum);
    bcx
}

/// A simple version of the pattern matching code that only handles
/// irrefutable patterns. This is used in let/argument patterns,
/// not in match statements. Unifying this code with the code above
/// sounds nice, but in practice it produces very inefficient code,
/// since the match code is so much more general. In most cases,
/// LLVM is able to optimize the code, but it causes longer compile
/// times and makes the generated code nigh impossible to read.
///
/// # Arguments
/// - bcx: starting basic block context
/// - pat: the irrefutable pattern being matched.
/// - val: the value being matched -- must be an lvalue (by ref, with cleanup)
pub fn bind_irrefutable_pat<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                    pat: &hir::Pat,
                                    val: MatchInput,
                                    cleanup_scope: cleanup::ScopeId)
                                    -> Block<'blk, 'tcx> {
    debug!("bind_irrefutable_pat(bcx={}, pat={:?})",
           bcx.to_str(),
           pat);

    if bcx.sess().asm_comments() {
        add_comment(bcx, &format!("bind_irrefutable_pat(pat={:?})",
                                 pat));
    }

    let _indenter = indenter();

    let _icx = push_ctxt("match::bind_irrefutable_pat");
    let mut bcx = bcx;
    let tcx = bcx.tcx();
    let ccx = bcx.ccx();
    match pat.node {
        hir::PatIdent(pat_binding_mode, ref path1, ref inner) => {
            if pat_is_binding(&tcx.def_map, &*pat) {
                // Allocate the stack slot where the value of this
                // binding will live and place it into the appropriate
                // map.
                bcx = mk_binding_alloca(
                    bcx, pat.id, path1.node.name, cleanup_scope, (),
                    "_match::bind_irrefutable_pat",
                    |(), bcx, Datum { val: llval, ty, kind: _ }| {
                        match pat_binding_mode {
                            hir::BindByValue(_) => {
                                // By value binding: move the value that `val`
                                // points at into the binding's stack slot.
                                let d = val.to_datum(ty);
                                d.store_to(bcx, llval)
                            }

                            hir::BindByRef(_) => {
                                // By ref binding: the value of the variable
                                // is the pointer `val` itself or fat pointer referenced by `val`
                                if type_is_fat_ptr(bcx.tcx(), ty) {
                                    expr::copy_fat_ptr(bcx, val.val, llval);
                                }
                                else {
                                    Store(bcx, val.val, llval);
                                }

                                bcx
                            }
                        }
                    });
            }

            if let Some(ref inner_pat) = *inner {
                bcx = bind_irrefutable_pat(bcx, &**inner_pat, val, cleanup_scope);
            }
        }
        hir::PatEnum(_, ref sub_pats) => {
            let opt_def = bcx.tcx().def_map.borrow().get(&pat.id).map(|d| d.full_def());
            match opt_def {
                Some(def::DefVariant(enum_id, var_id, _)) => {
                    let repr = adt::represent_node(bcx, pat.id);
                    let vinfo = ccx.tcx().lookup_adt_def(enum_id).variant_with_id(var_id);
                    let args = extract_variant_args(bcx,
                                                    &*repr,
                                                    vinfo.disr_val,
                                                    val);
                    if let Some(ref sub_pat) = *sub_pats {
                        for (i, &argval) in args.vals.iter().enumerate() {
                            bcx = bind_irrefutable_pat(
                                bcx,
                                &*sub_pat[i],
                                MatchInput::from_val(argval),
                                cleanup_scope);
                        }
                    }
                }
                Some(def::DefStruct(..)) => {
                    match *sub_pats {
                        None => {
                            // This is a unit-like struct. Nothing to do here.
                        }
                        Some(ref elems) => {
                            // This is the tuple struct case.
                            let repr = adt::represent_node(bcx, pat.id);
                            for (i, elem) in elems.iter().enumerate() {
                                let fldptr = adt::trans_field_ptr(bcx, &*repr,
                                                                  val.val, 0, i);
                                bcx = bind_irrefutable_pat(
                                    bcx,
                                    &**elem,
                                    MatchInput::from_val(fldptr),
                                    cleanup_scope);
                            }
                        }
                    }
                }
                _ => {
                    // Nothing to do here.
                }
            }
        }
        hir::PatStruct(_, ref fields, _) => {
            let tcx = bcx.tcx();
            let pat_ty = node_id_type(bcx, pat.id);
            let pat_repr = adt::represent_type(bcx.ccx(), pat_ty);
            let pat_v = VariantInfo::of_node(tcx, pat_ty, pat.id);
            for f in fields {
                let name = f.node.name;
                let fldptr = adt::trans_field_ptr(
                    bcx,
                    &*pat_repr,
                    val.val,
                    pat_v.discr,
                    pat_v.field_index(name));
                bcx = bind_irrefutable_pat(bcx,
                                           &*f.node.pat,
                                           MatchInput::from_val(fldptr),
                                           cleanup_scope);
            }
        }
        hir::PatTup(ref elems) => {
            let repr = adt::represent_node(bcx, pat.id);
            for (i, elem) in elems.iter().enumerate() {
                let fldptr = adt::trans_field_ptr(bcx, &*repr, val.val, 0, i);
                bcx = bind_irrefutable_pat(
                    bcx,
                    &**elem,
                    MatchInput::from_val(fldptr),
                    cleanup_scope);
            }
        }
        hir::PatBox(ref inner) => {
            let llbox = Load(bcx, val.val);
            bcx = bind_irrefutable_pat(
                bcx, &**inner, MatchInput::from_val(llbox), cleanup_scope);
        }
        hir::PatRegion(ref inner, _) => {
            let loaded_val = Load(bcx, val.val);
            bcx = bind_irrefutable_pat(
                bcx,
                &**inner,
                MatchInput::from_val(loaded_val),
                cleanup_scope);
        }
        hir::PatVec(ref before, ref slice, ref after) => {
            let pat_ty = node_id_type(bcx, pat.id);
            let mut extracted = extract_vec_elems(bcx, pat_ty, before.len(), after.len(), val);
            match slice {
                &Some(_) => {
                    extracted.vals.insert(
                        before.len(),
                        bind_subslice_pat(bcx, pat.id, val, before.len(), after.len())
                    );
                }
                &None => ()
            }
            bcx = before
                .iter()
                .chain(slice.iter())
                .chain(after.iter())
                .zip(extracted.vals)
                .fold(bcx, |bcx, (inner, elem)| {
                    bind_irrefutable_pat(
                        bcx,
                        &**inner,
                        MatchInput::from_val(elem),
                        cleanup_scope)
                });
        }
        hir::PatQPath(..) | hir::PatWild(_) | hir::PatLit(_) |
        hir::PatRange(_, _) => ()
    }
    return bcx;
}
