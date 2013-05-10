// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 *
 * # Compilation of match statements
 *
 * I will endeavor to explain the code as best I can.  I have only a loose
 * understanding of some parts of it.
 *
 * ## Matching
 *
 * The basic state of the code is maintained in an array `m` of `@Match`
 * objects.  Each `@Match` describes some list of patterns, all of which must
 * match against the current list of values.  If those patterns match, then
 * the arm listed in the match is the correct arm.  A given arm may have
 * multiple corresponding match entries, one for each alternative that
 * remains.  As we proceed these sets of matches are adjusted by the various
 * `enter_XXX()` functions, each of which adjusts the set of options given
 * some information about the value which has been matched.
 *
 * So, initially, there is one value and N matches, each of which have one
 * constituent pattern.  N here is usually the number of arms but may be
 * greater, if some arms have multiple alternatives.  For example, here:
 *
 *     enum Foo { A, B(int), C(uint, uint) }
 *     match foo {
 *         A => ...,
 *         B(x) => ...,
 *         C(1u, 2) => ...,
 *         C(_) => ...
 *     }
 *
 * The value would be `foo`.  There would be four matches, each of which
 * contains one pattern (and, in one case, a guard).  We could collect the
 * various options and then compile the code for the case where `foo` is an
 * `A`, a `B`, and a `C`.  When we generate the code for `C`, we would (1)
 * drop the two matches that do not match a `C` and (2) expand the other two
 * into two patterns each.  In the first case, the two patterns would be `1u`
 * and `2`, and the in the second case the _ pattern would be expanded into
 * `_` and `_`.  The two values are of course the arguments to `C`.
 *
 * Here is a quick guide to the various functions:
 *
 * - `compile_submatch()`: The main workhouse.  It takes a list of values and
 *   a list of matches and finds the various possibilities that could occur.
 *
 * - `enter_XXX()`: modifies the list of matches based on some information
 *   about the value that has been matched.  For example,
 *   `enter_rec_or_struct()` adjusts the values given that a record or struct
 *   has been matched.  This is an infallible pattern, so *all* of the matches
 *   must be either wildcards or record/struct patterns.  `enter_opt()`
 *   handles the fallible cases, and it is correspondingly more complex.
 *
 * ## Bindings
 *
 * We store information about the bound variables for each arm as part of the
 * per-arm `ArmData` struct.  There is a mapping from identifiers to
 * `BindingInfo` structs.  These structs contain the mode/id/type of the
 * binding, but they also contain up to two LLVM values, called `llmatch` and
 * `llbinding` respectively (the `llbinding`, as will be described shortly, is
 * optional and only present for by-value bindings---therefore it is bundled
 * up as part of the `TransBindingMode` type).  Both point at allocas.
 *
 * The `llmatch` binding always stores a pointer into the value being matched
 * which points at the data for the binding.  If the value being matched has
 * type `T`, then, `llmatch` will point at an alloca of type `T*` (and hence
 * `llmatch` has type `T**`).  So, if you have a pattern like:
 *
 *    let a: A = ...;
 *    let b: B = ...;
 *    match (a, b) { (ref c, copy d) => { ... } }
 *
 * For `c` and `d`, we would generate allocas of type `C*` and `D*`
 * respectively.  These are called the `llmatch`.  As we match, when we come
 * up against an identifier, we store the current pointer into the
 * corresponding alloca.
 *
 * In addition, for each by-value binding (copy or move), we will create a
 * second alloca (`llbinding`) that will hold the final value.  In this
 * example, that means that `d` would have this second alloca of type `D` (and
 * hence `llbinding` has type `D*`).
 *
 * Once a pattern is completely matched, and assuming that there is no guard
 * pattern, we will branch to a block that leads to the body itself.  For any
 * by-value bindings, this block will first load the ptr from `llmatch` (the
 * one of type `D*`) and copy/move the value into `llbinding` (the one of type
 * `D`).  The second alloca then becomes the value of the local variable.  For
 * by ref bindings, the value of the local variable is simply the first
 * alloca.
 *
 * So, for the example above, we would generate a setup kind of like this:
 *
 *        +-------+
 *        | Entry |
 *        +-------+
 *            |
 *        +-------------------------------------------+
 *        | llmatch_c = (addr of first half of tuple) |
 *        | llmatch_d = (addr of first half of tuple) |
 *        +-------------------------------------------+
 *            |
 *        +--------------------------------------+
 *        | *llbinding_d = **llmatch_dlbinding_d |
 *        +--------------------------------------+
 *
 * If there is a guard, the situation is slightly different, because we must
 * execute the guard code.  Moreover, we need to do so once for each of the
 * alternatives that lead to the arm, because if the guard fails, they may
 * have different points from which to continue the search. Therefore, in that
 * case, we generate code that looks more like:
 *
 *        +-------+
 *        | Entry |
 *        +-------+
 *            |
 *        +-------------------------------------------+
 *        | llmatch_c = (addr of first half of tuple) |
 *        | llmatch_d = (addr of first half of tuple) |
 *        +-------------------------------------------+
 *            |
 *        +-------------------------------------------------+
 *        | *llbinding_d = **llmatch_dlbinding_d            |
 *        | check condition                                 |
 *        | if false { free *llbinding_d, goto next case }  |
 *        | if true { goto body }                           |
 *        +-------------------------------------------------+
 *
 * The handling for the cleanups is a bit... sensitive.  Basically, the body
 * is the one that invokes `add_clean()` for each binding.  During the guard
 * evaluation, we add temporary cleanups and revoke them after the guard is
 * evaluated (it could fail, after all).  Presuming the guard fails, we drop
 * the various values we copied explicitly.  Note that guards and moves are
 * just plain incompatible.
 *
 */

use back::abi;
use lib::llvm::{llvm, ValueRef, BasicBlockRef};
use middle::const_eval;
use middle::borrowck::root_map_key;
use middle::pat_util::*;
use middle::resolve::DefMap;
use middle::trans::adt;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::callee;
use middle::trans::common::*;
use middle::trans::consts;
use middle::trans::controlflow;
use middle::trans::datum;
use middle::trans::datum::*;
use middle::trans::expr::Dest;
use middle::trans::expr;
use middle::trans::glue;
use middle::trans::tvec;
use middle::trans::type_of;
use middle::ty;
use util::common::indenter;

use core::hashmap::HashMap;
use syntax::ast;
use syntax::ast::ident;
use syntax::ast_util::path_to_ident;
use syntax::ast_util;
use syntax::codemap::{span, dummy_sp};
use syntax::print::pprust::pat_to_str;

// An option identifying a literal: either a unit-like struct or an
// expression.
pub enum Lit {
    UnitLikeStructLit(ast::node_id),    // the node ID of the pattern
    ExprLit(@ast::expr),
    ConstLit(ast::def_id),              // the def ID of the constant
}

// An option identifying a branch (either a literal, a enum variant or a
// range)
pub enum Opt {
    lit(Lit),
    var(/* disr val */int, @adt::Repr),
    range(@ast::expr, @ast::expr),
    vec_len_eq(uint),
    vec_len_ge(uint, /* slice */uint)
}

pub fn opt_eq(tcx: ty::ctxt, a: &Opt, b: &Opt) -> bool {
    match (a, b) {
      (&lit(a), &lit(b)) => {
        match (a, b) {
            (UnitLikeStructLit(a), UnitLikeStructLit(b)) => a == b,
            _ => {
                let a_expr;
                match a {
                    ExprLit(existing_a_expr) => a_expr = existing_a_expr,
                    ConstLit(a_const) => {
                        let e = const_eval::lookup_const_by_id(tcx, a_const);
                        a_expr = e.get();
                    }
                    UnitLikeStructLit(_) => {
                        fail!(~"UnitLikeStructLit should have been handled \
                               above")
                    }
                }

                let b_expr;
                match b {
                    ExprLit(existing_b_expr) => b_expr = existing_b_expr,
                    ConstLit(b_const) => {
                        let e = const_eval::lookup_const_by_id(tcx, b_const);
                        b_expr = e.get();
                    }
                    UnitLikeStructLit(_) => {
                        fail!(~"UnitLikeStructLit should have been handled \
                               above")
                    }
                }

                const_eval::compare_lit_exprs(tcx, a_expr, b_expr) == 0
            }
        }
      }
      (&range(a1, a2), &range(b1, b2)) => {
        const_eval::compare_lit_exprs(tcx, a1, b1) == 0 &&
        const_eval::compare_lit_exprs(tcx, a2, b2) == 0
      }
      (&var(a, _), &var(b, _)) => a == b,
      (&vec_len_eq(a), &vec_len_eq(b)) => a == b,
      (&vec_len_ge(a, _), &vec_len_ge(b, _)) => a == b,
      _ => false
    }
}

pub enum opt_result {
    single_result(Result),
    lower_bound(Result),
    range_result(Result, Result),
}
pub fn trans_opt(bcx: block, o: &Opt) -> opt_result {
    let _icx = bcx.insn_ctxt("match::trans_opt");
    let ccx = bcx.ccx();
    let bcx = bcx;
    match *o {
        lit(ExprLit(lit_expr)) => {
            let datumblock = expr::trans_to_datum(bcx, lit_expr);
            return single_result(datumblock.to_result());
        }
        lit(UnitLikeStructLit(pat_id)) => {
            let struct_ty = ty::node_id_to_type(bcx.tcx(), pat_id);
            let datumblock = datum::scratch_datum(bcx, struct_ty, true);
            return single_result(datumblock.to_result(bcx));
        }
        lit(ConstLit(lit_id)) => {
            let llval = consts::get_const_val(bcx.ccx(), lit_id);
            return single_result(rslt(bcx, llval));
        }
        var(disr_val, repr) => {
            return adt::trans_case(bcx, repr, disr_val);
        }
        range(l1, l2) => {
            return range_result(rslt(bcx, consts::const_expr(ccx, l1)),
                                rslt(bcx, consts::const_expr(ccx, l2)));
        }
        vec_len_eq(n) => {
            return single_result(rslt(bcx, C_int(ccx, n as int)));
        }
        vec_len_ge(n, _) => {
            return lower_bound(rslt(bcx, C_int(ccx, n as int)));
        }
    }
}

pub fn variant_opt(bcx: block, pat_id: ast::node_id)
    -> Opt {
    let ccx = bcx.ccx();
    match ccx.tcx.def_map.get_copy(&pat_id) {
        ast::def_variant(enum_id, var_id) => {
            let variants = ty::enum_variants(ccx.tcx, enum_id);
            for (*variants).each |v| {
                if var_id == v.id {
                    return var(v.disr_val,
                               adt::represent_node(bcx, pat_id))
                }
            }
            ::core::util::unreachable();
        }
        ast::def_fn(*) |
        ast::def_struct(_) => {
            return lit(UnitLikeStructLit(pat_id));
        }
        _ => {
            ccx.sess.bug(~"non-variant or struct in variant_opt()");
        }
    }
}

pub enum TransBindingMode {
    TrByValue(/*ismove:*/ bool, /*llbinding:*/ ValueRef),
    TrByRef,
    TrByImplicitRef
}

/**
 * Information about a pattern binding:
 * - `llmatch` is a pointer to a stack slot.  The stack slot contains a
 *   pointer into the value being matched.  Hence, llmatch has type `T**`
 *   where `T` is the value being matched.
 * - `trmode` is the trans binding mode
 * - `id` is the node id of the binding
 * - `ty` is the Rust type of the binding */
pub struct BindingInfo {
    llmatch: ValueRef,
    trmode: TransBindingMode,
    id: ast::node_id,
    ty: ty::t,
}

pub type BindingsMap = HashMap<ident, BindingInfo>;

pub struct ArmData<'self> {
    bodycx: block,
    arm: &'self ast::arm,
    bindings_map: BindingsMap
}

pub struct Match<'self> {
    pats: ~[@ast::pat],
    data: @ArmData<'self>
}

pub fn match_to_str(bcx: block, m: &Match) -> ~str {
    if bcx.sess().verbose() {
        // for many programs, this just take too long to serialize
        fmt!("%?", m.pats.map(|p| pat_to_str(*p, bcx.sess().intr())))
    } else {
        fmt!("%u pats", m.pats.len())
    }
}

pub fn matches_to_str(bcx: block, m: &[@Match]) -> ~str {
    fmt!("%?", m.map(|n| match_to_str(bcx, *n)))
}

pub fn has_nested_bindings(m: &[@Match], col: uint) -> bool {
    for m.each |br| {
        match br.pats[col].node {
          ast::pat_ident(_, _, Some(_)) => return true,
          _ => ()
        }
    }
    return false;
}

pub fn expand_nested_bindings<'r>(bcx: block,
                                  m: &[@Match<'r>],
                                  col: uint,
                                  val: ValueRef)
                              -> ~[@Match<'r>] {
    debug!("expand_nested_bindings(bcx=%s, m=%s, col=%u, val=%?)",
           bcx.to_str(),
           matches_to_str(bcx, m),
           col,
           bcx.val_str(val));
    let _indenter = indenter();

    do m.map |br| {
        match br.pats[col].node {
            ast::pat_ident(_, path, Some(inner)) => {
                let pats = vec::append(
                    vec::slice(br.pats, 0u, col).to_vec(),
                    vec::append(~[inner],
                                vec::slice(br.pats, col + 1u,
                                           br.pats.len())));

                let binding_info =
                    br.data.bindings_map.get(&path_to_ident(path));

                Store(bcx, val, binding_info.llmatch);
                @Match {pats: pats, data: br.data}
            }
            _ => {
                *br
            }
        }
    }
}

pub type enter_pat<'self> = &'self fn(@ast::pat) -> Option<~[@ast::pat]>;

pub fn assert_is_binding_or_wild(bcx: block, p: @ast::pat) {
    if !pat_is_binding_or_wild(bcx.tcx().def_map, p) {
        bcx.sess().span_bug(
            p.span,
            fmt!("Expected an identifier pattern but found p: %s",
                 pat_to_str(p, bcx.sess().intr())));
    }
}

pub fn enter_match<'r>(bcx: block,
                       dm: DefMap,
                       m: &[@Match<'r>],
                       col: uint,
                       val: ValueRef,
                       e: enter_pat)
                    -> ~[@Match<'r>] {
    debug!("enter_match(bcx=%s, m=%s, col=%u, val=%?)",
           bcx.to_str(),
           matches_to_str(bcx, m),
           col,
           bcx.val_str(val));
    let _indenter = indenter();

    let mut result = ~[];
    for m.each |br| {
        match e(br.pats[col]) {
            Some(sub) => {
                let pats =
                    vec::append(
                        vec::append(sub, vec::slice(br.pats, 0u, col)),
                        vec::slice(br.pats, col + 1u, br.pats.len()));

                let self = br.pats[col];
                match self.node {
                    ast::pat_ident(_, path, None) => {
                        if pat_is_binding(dm, self) {
                            let binding_info =
                                br.data.bindings_map.get(
                                    &path_to_ident(path));
                            Store(bcx, val, binding_info.llmatch);
                        }
                    }
                    _ => {}
                }

                result.push(@Match {pats: pats, data: br.data});
            }
            None => ()
        }
    }

    debug!("result=%s", matches_to_str(bcx, result));

    return result;
}

pub fn enter_default<'r>(bcx: block,
                         dm: DefMap,
                         m: &[@Match<'r>],
                         col: uint,
                         val: ValueRef)
                      -> ~[@Match<'r>] {
    debug!("enter_default(bcx=%s, m=%s, col=%u, val=%?)",
           bcx.to_str(),
           matches_to_str(bcx, m),
           col,
           bcx.val_str(val));
    let _indenter = indenter();

    do enter_match(bcx, dm, m, col, val) |p| {
        match p.node {
          ast::pat_wild | ast::pat_tup(_) | ast::pat_struct(*) => Some(~[]),
          ast::pat_ident(_, _, None) if pat_is_binding(dm, p) => Some(~[]),
          _ => None
        }
    }
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

pub fn enter_opt<'r>(bcx: block,
                     m: &[@Match<'r>],
                     opt: &Opt,
                     col: uint,
                     variant_size: uint,
                     val: ValueRef)
                  -> ~[@Match<'r>] {
    debug!("enter_opt(bcx=%s, m=%s, col=%u, val=%?)",
           bcx.to_str(),
           matches_to_str(bcx, m),
           col,
           bcx.val_str(val));
    let _indenter = indenter();

    let tcx = bcx.tcx();
    let dummy = @ast::pat {id: 0, node: ast::pat_wild, span: dummy_sp()};
    do enter_match(bcx, tcx.def_map, m, col, val) |p| {
        match p.node {
            ast::pat_enum(*) |
            ast::pat_ident(_, _, None) if pat_is_const(tcx.def_map, p) => {
                let const_def = tcx.def_map.get_copy(&p.id);
                let const_def_id = ast_util::def_id_of_def(const_def);
                if opt_eq(tcx, &lit(ConstLit(const_def_id)), opt) {
                    Some(~[])
                } else {
                    None
                }
            }
            ast::pat_enum(_, ref subpats) => {
                if opt_eq(tcx, &variant_opt(bcx, p.id), opt) {
                    match *subpats {
                        None => Some(vec::from_elem(variant_size, dummy)),
                        _ => copy *subpats
                    }
                } else {
                    None
                }
            }
            ast::pat_ident(_, _, None)
                    if pat_is_variant_or_struct(tcx.def_map, p) => {
                if opt_eq(tcx, &variant_opt(bcx, p.id), opt) {
                    Some(~[])
                } else {
                    None
                }
            }
            ast::pat_lit(l) => {
                if opt_eq(tcx, &lit(ExprLit(l)), opt) {Some(~[])} else {None}
            }
            ast::pat_range(l1, l2) => {
                if opt_eq(tcx, &range(l1, l2), opt) {Some(~[])} else {None}
            }
            ast::pat_struct(_, ref field_pats, _) => {
                if opt_eq(tcx, &variant_opt(bcx, p.id), opt) {
                    // Look up the struct variant ID.
                    let struct_id;
                    match tcx.def_map.get_copy(&p.id) {
                        ast::def_variant(_, found_struct_id) => {
                            struct_id = found_struct_id;
                        }
                        _ => {
                            tcx.sess.span_bug(p.span, "expected enum variant def");
                        }
                    }

                    // Reorder the patterns into the same order they were
                    // specified in the struct definition. Also fill in
                    // unspecified fields with dummy.
                    let mut reordered_patterns = ~[];
                    for ty::lookup_struct_fields(tcx, struct_id).each
                        |field| {
                            match field_pats.find(|p|
                                                  p.ident == field.ident) {
                                None => reordered_patterns.push(dummy),
                                Some(fp) => reordered_patterns.push(fp.pat)
                            }
                    }
                    Some(reordered_patterns)
                } else {
                    None
                }
            }
            ast::pat_vec(ref before, slice, ref after) => {
                match slice {
                    Some(slice) => {
                        let n = before.len() + after.len();
                        let i = before.len();
                        if opt_eq(tcx, &vec_len_ge(n, i), opt) {
                            Some(vec::append_one(copy *before, slice) +
                                    *after)
                        } else {
                            None
                        }
                    }
                    None => {
                        let n = before.len();
                        if opt_eq(tcx, &vec_len_eq(n), opt) {
                            Some(copy *before)
                        } else {
                            None
                        }
                    }
                }
            }
            _ => {
                assert_is_binding_or_wild(bcx, p);
                Some(vec::from_elem(variant_size, dummy))
            }
        }
    }
}

pub fn enter_rec_or_struct<'r>(bcx: block,
                               dm: DefMap,
                               m: &[@Match<'r>],
                               col: uint,
                               fields: &[ast::ident],
                               val: ValueRef)
                            -> ~[@Match<'r>] {
    debug!("enter_rec_or_struct(bcx=%s, m=%s, col=%u, val=%?)",
           bcx.to_str(),
           matches_to_str(bcx, m),
           col,
           bcx.val_str(val));
    let _indenter = indenter();

    let dummy = @ast::pat {id: 0, node: ast::pat_wild, span: dummy_sp()};
    do enter_match(bcx, dm, m, col, val) |p| {
        match p.node {
            ast::pat_struct(_, ref fpats, _) => {
                let mut pats = ~[];
                for fields.each |fname| {
                    match fpats.find(|p| p.ident == *fname) {
                        None => pats.push(dummy),
                        Some(pat) => pats.push(pat.pat)
                    }
                }
                Some(pats)
            }
            _ => {
                assert_is_binding_or_wild(bcx, p);
                Some(vec::from_elem(fields.len(), dummy))
            }
        }
    }
}

pub fn enter_tup<'r>(bcx: block,
                     dm: DefMap,
                     m: &[@Match<'r>],
                     col: uint,
                     val: ValueRef,
                     n_elts: uint)
                  -> ~[@Match<'r>] {
    debug!("enter_tup(bcx=%s, m=%s, col=%u, val=%?)",
           bcx.to_str(),
           matches_to_str(bcx, m),
           col,
           bcx.val_str(val));
    let _indenter = indenter();

    let dummy = @ast::pat {id: 0, node: ast::pat_wild, span: dummy_sp()};
    do enter_match(bcx, dm, m, col, val) |p| {
        match p.node {
            ast::pat_tup(/*bad*/copy elts) => {
                Some(elts)
            }
            _ => {
                assert_is_binding_or_wild(bcx, p);
                Some(vec::from_elem(n_elts, dummy))
            }
        }
    }
}

pub fn enter_tuple_struct<'r>(bcx: block,
                              dm: DefMap,
                              m: &[@Match<'r>],
                              col: uint,
                              val: ValueRef,
                              n_elts: uint)
                          -> ~[@Match<'r>] {
    debug!("enter_tuple_struct(bcx=%s, m=%s, col=%u, val=%?)",
           bcx.to_str(),
           matches_to_str(bcx, m),
           col,
           bcx.val_str(val));
    let _indenter = indenter();

    let dummy = @ast::pat {id: 0, node: ast::pat_wild, span: dummy_sp()};
    do enter_match(bcx, dm, m, col, val) |p| {
        match p.node {
            ast::pat_enum(_, Some(/*bad*/copy elts)) => Some(elts),
            _ => {
                assert_is_binding_or_wild(bcx, p);
                Some(vec::from_elem(n_elts, dummy))
            }
        }
    }
}

pub fn enter_box<'r>(bcx: block,
                     dm: DefMap,
                     m: &[@Match<'r>],
                     col: uint,
                     val: ValueRef)
                 -> ~[@Match<'r>] {
    debug!("enter_box(bcx=%s, m=%s, col=%u, val=%?)",
           bcx.to_str(),
           matches_to_str(bcx, m),
           col,
           bcx.val_str(val));
    let _indenter = indenter();

    let dummy = @ast::pat {id: 0, node: ast::pat_wild, span: dummy_sp()};
    do enter_match(bcx, dm, m, col, val) |p| {
        match p.node {
            ast::pat_box(sub) => {
                Some(~[sub])
            }
            _ => {
                assert_is_binding_or_wild(bcx, p);
                Some(~[dummy])
            }
        }
    }
}

pub fn enter_uniq<'r>(bcx: block,
                      dm: DefMap,
                      m: &[@Match<'r>],
                      col: uint,
                      val: ValueRef)
                  -> ~[@Match<'r>] {
    debug!("enter_uniq(bcx=%s, m=%s, col=%u, val=%?)",
           bcx.to_str(),
           matches_to_str(bcx, m),
           col,
           bcx.val_str(val));
    let _indenter = indenter();

    let dummy = @ast::pat {id: 0, node: ast::pat_wild, span: dummy_sp()};
    do enter_match(bcx, dm, m, col, val) |p| {
        match p.node {
            ast::pat_uniq(sub) => {
                Some(~[sub])
            }
            _ => {
                assert_is_binding_or_wild(bcx, p);
                Some(~[dummy])
            }
        }
    }
}

pub fn enter_region<'r>(bcx: block,
                        dm: DefMap,
                        m: &[@Match<'r>],
                        col: uint,
                        val: ValueRef)
                    -> ~[@Match<'r>] {
    debug!("enter_region(bcx=%s, m=%s, col=%u, val=%?)",
           bcx.to_str(),
           matches_to_str(bcx, m),
           col,
           bcx.val_str(val));
    let _indenter = indenter();

    let dummy = @ast::pat { id: 0, node: ast::pat_wild, span: dummy_sp() };
    do enter_match(bcx, dm, m, col, val) |p| {
        match p.node {
            ast::pat_region(sub) => {
                Some(~[sub])
            }
            _ => {
                assert_is_binding_or_wild(bcx, p);
                Some(~[dummy])
            }
        }
    }
}

// Returns the options in one column of matches. An option is something that
// needs to be conditionally matched at runtime; for example, the discriminant
// on a set of enum variants or a literal.
pub fn get_options(bcx: block, m: &[@Match], col: uint) -> ~[Opt] {
    let ccx = bcx.ccx();
    fn add_to_set(tcx: ty::ctxt, set: &mut ~[Opt], val: Opt) {
        if set.any(|l| opt_eq(tcx, l, &val)) {return;}
        set.push(val);
    }

    let mut found = ~[];
    for m.each |br| {
        let cur = br.pats[col];
        match cur.node {
            ast::pat_lit(l) => {
                add_to_set(ccx.tcx, &mut found, lit(ExprLit(l)));
            }
            ast::pat_ident(*) => {
                // This is one of: an enum variant, a unit-like struct, or a
                // variable binding.
                match ccx.tcx.def_map.find(&cur.id) {
                    Some(&ast::def_variant(*)) => {
                        add_to_set(ccx.tcx, &mut found,
                                   variant_opt(bcx, cur.id));
                    }
                    Some(&ast::def_struct(*)) => {
                        add_to_set(ccx.tcx, &mut found,
                                   lit(UnitLikeStructLit(cur.id)));
                    }
                    Some(&ast::def_const(const_did)) => {
                        add_to_set(ccx.tcx, &mut found,
                                   lit(ConstLit(const_did)));
                    }
                    _ => {}
                }
            }
            ast::pat_enum(*) | ast::pat_struct(*) => {
                // This could be one of: a tuple-like enum variant, a
                // struct-like enum variant, or a struct.
                match ccx.tcx.def_map.find(&cur.id) {
                    Some(&ast::def_fn(*)) |
                    Some(&ast::def_variant(*)) => {
                        add_to_set(ccx.tcx, &mut found,
                                   variant_opt(bcx, cur.id));
                    }
                    Some(&ast::def_const(const_did)) => {
                        add_to_set(ccx.tcx, &mut found,
                                   lit(ConstLit(const_did)));
                    }
                    _ => {}
                }
            }
            ast::pat_range(l1, l2) => {
                add_to_set(ccx.tcx, &mut found, range(l1, l2));
            }
            ast::pat_vec(ref before, slice, ref after) => {
                let opt = match slice {
                    None => vec_len_eq(before.len()),
                    Some(_) => vec_len_ge(before.len() + after.len(),
                                          before.len())
                };
                add_to_set(ccx.tcx, &mut found, opt);
            }
            _ => {}
        }
    }
    return found;
}

pub struct ExtractedBlock {
    vals: ~[ValueRef],
    bcx: block
}

pub fn extract_variant_args(bcx: block,
                            repr: &adt::Repr,
                            disr_val: int,
                            val: ValueRef)
    -> ExtractedBlock {
    let _icx = bcx.insn_ctxt("match::extract_variant_args");
    let args = do vec::from_fn(adt::num_args(repr, disr_val)) |i| {
        adt::trans_field_ptr(bcx, repr, val, disr_val, i)
    };

    ExtractedBlock { vals: args, bcx: bcx }
}

fn match_datum(bcx: block, val: ValueRef, pat_id: ast::node_id) -> Datum {
    //! Helper for converting from the ValueRef that we pass around in
    //! the match code, which is always by ref, into a Datum. Eventually
    //! we should just pass around a Datum and be done with it.

    let ty = node_id_type(bcx, pat_id);
    Datum {val: val, ty: ty, mode: datum::ByRef, source: RevokeClean}
}


pub fn extract_vec_elems(bcx: block,
                         pat_span: span,
                         pat_id: ast::node_id,
                         elem_count: uint,
                         slice: Option<uint>,
                         val: ValueRef,
                         count: ValueRef)
                      -> ExtractedBlock {
    let _icx = bcx.insn_ctxt("match::extract_vec_elems");
    let vec_datum = match_datum(bcx, val, pat_id);
    let (bcx, base, len) = vec_datum.get_vec_base_and_len(bcx, pat_span,
                                                          pat_id, 0);
    let vt = tvec::vec_types(bcx, node_id_type(bcx, pat_id));

    let mut elems = do vec::from_fn(elem_count) |i| {
        match slice {
            None => GEPi(bcx, base, ~[i]),
            Some(n) if i < n => GEPi(bcx, base, ~[i]),
            Some(n) if i > n => {
                InBoundsGEP(bcx, base, ~[
                    Sub(bcx, count,
                        C_int(bcx.ccx(), (elem_count - i) as int))])
            }
            _ => unsafe { llvm::LLVMGetUndef(vt.llunit_ty) }
        }
    };
    if slice.is_some() {
        let n = slice.get();
        let slice_offset = Mul(bcx, vt.llunit_size,
            C_int(bcx.ccx(), n as int)
        );
        let slice_begin = tvec::pointer_add(bcx, base, slice_offset);
        let slice_len_offset = Mul(bcx, vt.llunit_size,
            C_int(bcx.ccx(), (elem_count - 1u) as int)
        );
        let slice_len = Sub(bcx, len, slice_len_offset);
        let slice_ty = ty::mk_evec(bcx.tcx(),
            ty::mt {ty: vt.unit_ty, mutbl: ast::m_imm},
            ty::vstore_slice(ty::re_static)
        );
        let scratch = scratch_datum(bcx, slice_ty, false);
        Store(bcx, slice_begin,
            GEPi(bcx, scratch.val, [0u, abi::slice_elt_base])
        );
        Store(bcx, slice_len,
            GEPi(bcx, scratch.val, [0u, abi::slice_elt_len])
        );
        elems[n] = scratch.val;
        scratch.add_clean(bcx);
    }

    ExtractedBlock { vals: elems, bcx: bcx }
}

// NB: This function does not collect fields from struct-like enum variants.
pub fn collect_record_or_struct_fields(bcx: block,
                                       m: &[@Match],
                                       col: uint)
                                    -> ~[ast::ident] {
    let mut fields: ~[ast::ident] = ~[];
    for m.each |br| {
        match br.pats[col].node {
          ast::pat_struct(_, ref fs, _) => {
            match ty::get(node_id_type(bcx, br.pats[col].id)).sty {
              ty::ty_struct(*) => extend(&mut fields, *fs),
              _ => ()
            }
          }
          _ => ()
        }
    }
    return fields;

    fn extend(idents: &mut ~[ast::ident], field_pats: &[ast::field_pat]) {
        for field_pats.each |field_pat| {
            let field_ident = field_pat.ident;
            if !vec::any(*idents, |x| *x == field_ident) {
                idents.push(field_ident);
            }
        }
    }
}

pub fn pats_require_rooting(bcx: block,
                            m: &[@Match],
                            col: uint)
                         -> bool {
    vec::any(m, |br| {
        let pat_id = br.pats[col].id;
        let key = root_map_key {id: pat_id, derefs: 0u };
        bcx.ccx().maps.root_map.contains_key(&key)
    })
}

pub fn root_pats_as_necessary(mut bcx: block,
                              m: &[@Match],
                              col: uint,
                              val: ValueRef)
                           -> block {
    for m.each |br| {
        let pat_id = br.pats[col].id;
        if pat_id != 0 {
            let datum = Datum {val: val, ty: node_id_type(bcx, pat_id),
                               mode: ByRef, source: ZeroMem};
            bcx = datum.root_and_write_guard(bcx, br.pats[col].span, pat_id, 0);
        }
    }
    return bcx;
}

// Macro for deciding whether any of the remaining matches fit a given kind of
// pattern.  Note that, because the macro is well-typed, either ALL of the
// matches should fit that sort of pattern or NONE (however, some of the
// matches may be wildcards like _ or identifiers).
macro_rules! any_pat (
    ($m:expr, $pattern:pat) => (
        vec::any($m, |br| {
            match br.pats[col].node {
                $pattern => true,
                _ => false
            }
        })
    )
)

pub fn any_box_pat(m: &[@Match], col: uint) -> bool {
    any_pat!(m, ast::pat_box(_))
}

pub fn any_uniq_pat(m: &[@Match], col: uint) -> bool {
    any_pat!(m, ast::pat_uniq(_))
}

pub fn any_region_pat(m: &[@Match], col: uint) -> bool {
    any_pat!(m, ast::pat_region(_))
}

pub fn any_tup_pat(m: &[@Match], col: uint) -> bool {
    any_pat!(m, ast::pat_tup(_))
}

pub fn any_tuple_struct_pat(bcx: block, m: &[@Match], col: uint) -> bool {
    vec::any(m, |br| {
        let pat = br.pats[col];
        match pat.node {
            ast::pat_enum(_, Some(_)) => {
                match bcx.tcx().def_map.find(&pat.id) {
                    Some(&ast::def_fn(*)) |
                    Some(&ast::def_struct(*)) => true,
                    _ => false
                }
            }
            _ => false
        }
    })
}

pub type mk_fail = @fn() -> BasicBlockRef;

pub fn pick_col(m: &[@Match]) -> uint {
    fn score(p: @ast::pat) -> uint {
        match p.node {
          ast::pat_lit(_) | ast::pat_enum(_, _) | ast::pat_range(_, _) => 1u,
          ast::pat_ident(_, _, Some(p)) => score(p),
          _ => 0u
        }
    }
    let mut scores = vec::from_elem(m[0].pats.len(), 0u);
    for m.each |br| {
        let mut i = 0u;
        for br.pats.each |p| { scores[i] += score(*p); i += 1u; }
    }
    let mut max_score = 0u;
    let mut best_col = 0u;
    let mut i = 0u;
    for scores.each |score| {
        let score = *score;

        // Irrefutable columns always go first, they'd only be duplicated in
        // the branches.
        if score == 0u { return i; }
        // If no irrefutable ones are found, we pick the one with the biggest
        // branching factor.
        if score > max_score { max_score = score; best_col = i; }
        i += 1u;
    }
    return best_col;
}

#[deriving(Eq)]
pub enum branch_kind { no_branch, single, switch, compare, compare_vec_len, }

// Compiles a comparison between two things.
//
// NB: This must produce an i1, not a Rust bool (i8).
pub fn compare_values(cx: block,
                      lhs: ValueRef,
                      rhs: ValueRef,
                      rhs_t: ty::t)
                   -> Result {
    let _icx = cx.insn_ctxt("compare_values");
    if ty::type_is_scalar(rhs_t) {
      let rs = compare_scalar_types(cx, lhs, rhs, rhs_t, ast::eq);
      return rslt(rs.bcx, rs.val);
    }

    match ty::get(rhs_t).sty {
        ty::ty_estr(ty::vstore_uniq) => {
            let scratch_result = scratch_datum(cx, ty::mk_bool(), false);
            let scratch_lhs = alloca(cx, val_ty(lhs));
            Store(cx, lhs, scratch_lhs);
            let scratch_rhs = alloca(cx, val_ty(rhs));
            Store(cx, rhs, scratch_rhs);
            let did = cx.tcx().lang_items.uniq_str_eq_fn();
            let bcx = callee::trans_lang_call(cx, did,
                                                        ~[scratch_lhs,
                                                          scratch_rhs],
                                                        expr::SaveIn(
                                                         scratch_result.val));
            let result = scratch_result.to_result(bcx);
            Result {
                bcx: result.bcx,
                val: bool_to_i1(result.bcx, result.val)
            }
        }
        ty::ty_estr(_) => {
            let scratch_result = scratch_datum(cx, ty::mk_bool(), false);
            let did = cx.tcx().lang_items.str_eq_fn();
            let bcx = callee::trans_lang_call(cx, did,
                                                        ~[lhs, rhs],
                                                        expr::SaveIn(
                                                         scratch_result.val));
            let result = scratch_result.to_result(bcx);
            Result {
                bcx: result.bcx,
                val: bool_to_i1(result.bcx, result.val)
            }
        }
        _ => {
            cx.tcx().sess.bug(~"only scalars and strings supported in \
                                compare_values");
        }
    }
}

pub fn store_non_ref_bindings(bcx: block,
                              data: &ArmData,
                              opt_temp_cleanups: Option<&mut ~[ValueRef]>)
    -> block
{
    /*!
     *
     * For each copy/move binding, copy the value from the value
     * being matched into its final home.  This code executes once
     * one of the patterns for a given arm has completely matched.
     * It adds temporary cleanups to the `temp_cleanups` array,
     * if one is provided.
     */

    let mut bcx = bcx;
    let mut opt_temp_cleanups = opt_temp_cleanups;
    for data.bindings_map.each_value |&binding_info| {
        match binding_info.trmode {
            TrByValue(is_move, lldest) => {
                let llval = Load(bcx, binding_info.llmatch); // get a T*
                let datum = Datum {val: llval, ty: binding_info.ty,
                                   mode: ByRef, source: ZeroMem};
                bcx = {
                    if is_move {
                        datum.move_to(bcx, INIT, lldest)
                    } else {
                        datum.copy_to(bcx, INIT, lldest)
                    }
                };

                do opt_temp_cleanups.mutate |temp_cleanups| {
                    add_clean_temp_mem(bcx, lldest, binding_info.ty);
                    temp_cleanups.push(lldest);
                    temp_cleanups
                }
            }
            TrByRef | TrByImplicitRef => {}
        }
    }
    return bcx;
}

pub fn insert_lllocals(bcx: block,
                       data: &ArmData,
                       add_cleans: bool) -> block {
    /*!
     *
     * For each binding in `data.bindings_map`, adds an appropriate entry into
     * the `fcx.lllocals` map.  If add_cleans is true, then adds cleanups for
     * the bindings. */

    for data.bindings_map.each_value |&binding_info| {
        let llval = match binding_info.trmode {
            // By value bindings: use the stack slot that we
            // copied/moved the value into
            TrByValue(_, lldest) => {
                if add_cleans {
                    add_clean(bcx, lldest, binding_info.ty);
                }

                lldest
            }

            // By ref binding: use the ptr into the matched value
            TrByRef => {
                binding_info.llmatch
            }

            // Ugly: for implicit ref, we actually want a T*, but
            // we have a T**, so we had to load.  This will go away
            // once implicit refs go away.
            TrByImplicitRef => {
                Load(bcx, binding_info.llmatch)
            }
        };

        bcx.fcx.lllocals.insert(binding_info.id,
                                local_mem(llval));
    }
    return bcx;
}

pub fn compile_guard(bcx: block,
                     guard_expr: @ast::expr,
                     data: &ArmData,
                     m: &[@Match],
                     vals: &[ValueRef],
                     chk: Option<mk_fail>)
                  -> block {
    debug!("compile_guard(bcx=%s, guard_expr=%s, m=%s, vals=%?)",
           bcx.to_str(),
           bcx.expr_to_str(guard_expr),
           matches_to_str(bcx, m),
           vals.map(|v| bcx.val_str(*v)));
    let _indenter = indenter();

    let mut bcx = bcx;
    let mut temp_cleanups = ~[];
    bcx = store_non_ref_bindings(bcx, data, Some(&mut temp_cleanups));
    bcx = insert_lllocals(bcx, data, false);

    let val = unpack_result!(bcx, {
        do with_scope_result(bcx, guard_expr.info(),
                             "guard") |bcx| {
            expr::trans_to_datum(bcx, guard_expr).to_result()
        }
    });
    let val = bool_to_i1(bcx, val);

    // Revoke the temp cleanups now that the guard successfully executed.
    for temp_cleanups.each |llval| {
        revoke_clean(bcx, *llval);
    }

    return do with_cond(bcx, Not(bcx, val)) |bcx| {
        // Guard does not match: free the values we copied,
        // and remove all bindings from the lllocals table
        let bcx = drop_bindings(bcx, data);
        compile_submatch(bcx, m, vals, chk);
        bcx
    };

    fn drop_bindings(bcx: block, data: &ArmData) -> block {
        let mut bcx = bcx;
        for data.bindings_map.each_value |&binding_info| {
            match binding_info.trmode {
                TrByValue(_, llval) => {
                    bcx = glue::drop_ty(bcx, llval, binding_info.ty);
                }
                TrByRef | TrByImplicitRef => {}
            }
            bcx.fcx.lllocals.remove(&binding_info.id);
        }
        return bcx;
    }
}

pub fn compile_submatch(bcx: block,
                        m: &[@Match],
                        vals: &[ValueRef],
                        chk: Option<mk_fail>) {
    debug!("compile_submatch(bcx=%s, m=%s, vals=%?)",
           bcx.to_str(),
           matches_to_str(bcx, m),
           vals.map(|v| bcx.val_str(*v)));
    let _indenter = indenter();

    /*
      For an empty match, a fall-through case must exist
     */
    assert!((m.len() > 0u || chk.is_some()));
    let _icx = bcx.insn_ctxt("match::compile_submatch");
    let mut bcx = bcx;
    let tcx = bcx.tcx(), dm = tcx.def_map;
    if m.len() == 0u {
        Br(bcx, chk.get()());
        return;
    }
    if m[0].pats.len() == 0u {
        let data = m[0].data;
        match data.arm.guard {
            Some(guard_expr) => {
                bcx = compile_guard(bcx, guard_expr, m[0].data,
                                    vec::slice(m, 1, m.len()),
                                    vals, chk);
            }
            _ => ()
        }
        Br(bcx, data.bodycx.llbb);
        return;
    }

    let col = pick_col(m);
    let val = vals[col];
    let m = {
        if has_nested_bindings(m, col) {
            expand_nested_bindings(bcx, m, col, val)
        } else {
            m.to_vec()
        }
    };

    let vals_left = vec::append(vec::slice(vals, 0u, col).to_vec(),
                                vec::slice(vals, col + 1u, vals.len()));
    let ccx = *bcx.fcx.ccx;
    let mut pat_id = 0;
    let mut pat_span = dummy_sp();
    for m.each |br| {
        // Find a real id (we're adding placeholder wildcard patterns, but
        // each column is guaranteed to have at least one real pattern)
        if pat_id == 0 {
            pat_id = br.pats[col].id;
            pat_span = br.pats[col].span;
        }
    }

    // If we are not matching against an `@T`, we should not be
    // required to root any values.
    assert!(any_box_pat(m, col) || !pats_require_rooting(bcx, m, col));

    let rec_fields = collect_record_or_struct_fields(bcx, m, col);
    if rec_fields.len() > 0 {
        let pat_ty = node_id_type(bcx, pat_id);
        let pat_repr = adt::represent_type(bcx.ccx(), pat_ty);
        do expr::with_field_tys(tcx, pat_ty, None) |discr, field_tys| {
            let rec_vals = rec_fields.map(|field_name| {
                let ix = ty::field_idx_strict(tcx, *field_name, field_tys);
                adt::trans_field_ptr(bcx, pat_repr, val, discr, ix)
            });
            compile_submatch(
                bcx,
                enter_rec_or_struct(bcx, dm, m, col, rec_fields, val),
                vec::append(rec_vals, vals_left),
                chk);
        }
        return;
    }

    if any_tup_pat(m, col) {
        let tup_ty = node_id_type(bcx, pat_id);
        let tup_repr = adt::represent_type(bcx.ccx(), tup_ty);
        let n_tup_elts = match ty::get(tup_ty).sty {
          ty::ty_tup(ref elts) => elts.len(),
          _ => ccx.sess.bug(~"non-tuple type in tuple pattern")
        };
        let tup_vals = do vec::from_fn(n_tup_elts) |i| {
            adt::trans_field_ptr(bcx, tup_repr, val, 0, i)
        };
        compile_submatch(bcx, enter_tup(bcx, dm, m, col, val, n_tup_elts),
                         vec::append(tup_vals, vals_left), chk);
        return;
    }

    if any_tuple_struct_pat(bcx, m, col) {
        let struct_ty = node_id_type(bcx, pat_id);
        let struct_element_count;
        match ty::get(struct_ty).sty {
            ty::ty_struct(struct_id, _) => {
                struct_element_count =
                    ty::lookup_struct_fields(tcx, struct_id).len();
            }
            _ => {
                ccx.sess.bug(~"non-struct type in tuple struct pattern");
            }
        }

        let struct_repr = adt::represent_type(bcx.ccx(), struct_ty);
        let llstructvals = do vec::from_fn(struct_element_count) |i| {
            adt::trans_field_ptr(bcx, struct_repr, val, 0, i)
        };
        compile_submatch(bcx,
                         enter_tuple_struct(bcx, dm, m, col, val,
                                            struct_element_count),
                         vec::append(llstructvals, vals_left),
                         chk);
        return;
    }

    // Unbox in case of a box field
    if any_box_pat(m, col) {
        bcx = root_pats_as_necessary(bcx, m, col, val);
        let llbox = Load(bcx, val);
        let box_no_addrspace = non_gc_box_cast(bcx, llbox);
        let unboxed =
            GEPi(bcx, box_no_addrspace, [0u, abi::box_field_body]);
        compile_submatch(bcx, enter_box(bcx, dm, m, col, val),
                         vec::append(~[unboxed], vals_left), chk);
        return;
    }

    if any_uniq_pat(m, col) {
        let llbox = Load(bcx, val);
        let box_no_addrspace = non_gc_box_cast(bcx, llbox);
        let unboxed =
            GEPi(bcx, box_no_addrspace, [0u, abi::box_field_body]);
        compile_submatch(bcx, enter_uniq(bcx, dm, m, col, val),
                         vec::append(~[unboxed], vals_left), chk);
        return;
    }

    if any_region_pat(m, col) {
        let loaded_val = Load(bcx, val);
        compile_submatch(bcx, enter_region(bcx, dm, m, col, val),
                         vec::append(~[loaded_val], vals_left), chk);
        return;
    }

    // Decide what kind of branch we need
    let opts = get_options(bcx, m, col);
    let mut kind = no_branch;
    let mut test_val = val;
    if opts.len() > 0u {
        match opts[0] {
            var(_, repr) => {
                let (the_kind, val_opt) = adt::trans_switch(bcx, repr, val);
                kind = the_kind;
                for val_opt.each |&tval| { test_val = tval; }
            }
            lit(_) => {
                let pty = node_id_type(bcx, pat_id);
                test_val = load_if_immediate(bcx, val, pty);
                kind = if ty::type_is_integral(pty) { switch }
                else { compare };
            }
            range(_, _) => {
                test_val = Load(bcx, val);
                kind = compare;
            },
            vec_len_eq(*) | vec_len_ge(*) => {
                let vt = tvec::vec_types(bcx, node_id_type(bcx, pat_id));
                let unboxed = load_if_immediate(bcx, val, vt.vec_ty);
                let (_, len) = tvec::get_base_and_len(
                    bcx, unboxed, vt.vec_ty
                );
                test_val = SDiv(bcx, len, vt.llunit_size);
                kind = compare_vec_len;
            }
        }
    }
    for opts.each |o| {
        match *o {
            range(_, _) => { kind = compare; break }
            _ => ()
        }
    }
    let else_cx = match kind {
        no_branch | single => bcx,
        _ => sub_block(bcx, "match_else")
    };
    let sw = if kind == switch {
        Switch(bcx, test_val, else_cx.llbb, opts.len())
    } else {
        C_int(ccx, 0) // Placeholder for when not using a switch
    };

    let defaults = enter_default(else_cx, dm, m, col, val);
    let exhaustive = chk.is_none() && defaults.len() == 0u;
    let len = opts.len();
    let mut i = 0u;

    // Compile subtrees for each option
    for opts.each |opt| {
        i += 1u;
        let mut opt_cx = else_cx;
        if !exhaustive || i < len {
            opt_cx = sub_block(bcx, "match_case");
            match kind {
              single => Br(bcx, opt_cx.llbb),
              switch => {
                  match trans_opt(bcx, opt) {
                      single_result(r) => {
                        unsafe {
                          llvm::LLVMAddCase(sw, r.val, opt_cx.llbb);
                          bcx = r.bcx;
                        }
                      }
                      _ => {
                          bcx.sess().bug(
                              ~"in compile_submatch, expected \
                                trans_opt to return a single_result")
                      }
                  }
              }
              compare => {
                  let t = node_id_type(bcx, pat_id);
                  let Result {bcx: after_cx, val: matches} = {
                      do with_scope_result(bcx, None,
                                           "compare_scope") |bcx| {
                          match trans_opt(bcx, opt) {
                              single_result(
                                  Result {bcx, val}) => {
                                  compare_values(bcx, test_val, val, t)
                              }
                              lower_bound(
                                  Result {bcx, val}) => {
                                  compare_scalar_types(
                                          bcx, test_val, val,
                                          t, ast::ge)
                              }
                              range_result(
                                  Result {val: vbegin, _},
                                  Result {bcx, val: vend}) => {
                                  let Result {bcx, val: llge} =
                                      compare_scalar_types(
                                          bcx, test_val,
                                          vbegin, t, ast::ge);
                                  let Result {bcx, val: llle} =
                                      compare_scalar_types(
                                          bcx, test_val, vend,
                                          t, ast::le);
                                  rslt(bcx, And(bcx, llge, llle))
                              }
                          }
                      }
                  };
                  bcx = sub_block(after_cx, "compare_next");
                  CondBr(after_cx, matches, opt_cx.llbb, bcx.llbb);
              }
              compare_vec_len => {
                  let Result {bcx: after_cx, val: matches} = {
                      do with_scope_result(bcx, None,
                                           "compare_vec_len_scope") |bcx| {
                          match trans_opt(bcx, opt) {
                              single_result(
                                  Result {bcx, val}) => {
                                  let value = compare_scalar_values(
                                      bcx, test_val, val,
                                      signed_int, ast::eq);
                                  rslt(bcx, value)
                              }
                              lower_bound(
                                  Result {bcx, val: val}) => {
                                  let value = compare_scalar_values(
                                      bcx, test_val, val,
                                      signed_int, ast::ge);
                                  rslt(bcx, value)
                              }
                              range_result(
                                  Result {val: vbegin, _},
                                  Result {bcx, val: vend}) => {
                                  let llge =
                                      compare_scalar_values(
                                          bcx, test_val,
                                          vbegin, signed_int, ast::ge);
                                  let llle =
                                      compare_scalar_values(
                                          bcx, test_val, vend,
                                          signed_int, ast::le);
                                  rslt(bcx, And(bcx, llge, llle))
                              }
                          }
                      }
                  };
                  bcx = sub_block(after_cx, "compare_vec_len_next");
                  CondBr(after_cx, matches, opt_cx.llbb, bcx.llbb);
              }
              _ => ()
            }
        } else if kind == compare || kind == compare_vec_len {
            Br(bcx, else_cx.llbb);
        }

        let mut size = 0u;
        let mut unpacked = ~[];
        match *opt {
            var(disr_val, repr) => {
                let ExtractedBlock {vals: argvals, bcx: new_bcx} =
                    extract_variant_args(opt_cx, repr, disr_val, val);
                size = argvals.len();
                unpacked = argvals;
                opt_cx = new_bcx;
            }
            vec_len_eq(n) | vec_len_ge(n, _) => {
                let n = match *opt {
                    vec_len_ge(*) => n + 1u,
                    _ => n
                };
                let slice = match *opt {
                    vec_len_ge(_, i) => Some(i),
                    _ => None
                };
                let args = extract_vec_elems(opt_cx, pat_span, pat_id, n, slice,
                                             val, test_val);
                size = args.vals.len();
                unpacked = /*bad*/copy args.vals;
                opt_cx = args.bcx;
            }
            lit(_) | range(_, _) => ()
        }
        let opt_ms = enter_opt(opt_cx, m, opt, col, size, val);
        let opt_vals = vec::append(unpacked, vals_left);
        compile_submatch(opt_cx, opt_ms, opt_vals, chk);
    }

    // Compile the fall-through case, if any
    if !exhaustive {
        if kind == compare || kind == compare_vec_len {
            Br(bcx, else_cx.llbb);
        }
        if kind != single {
            compile_submatch(else_cx, defaults, vals_left, chk);
        }
    }
}

pub fn trans_match(bcx: block,
                   match_expr: @ast::expr,
                   discr_expr: @ast::expr,
                   arms: ~[ast::arm],
                   dest: Dest) -> block {
    let _icx = bcx.insn_ctxt("match::trans_match");
    do with_scope(bcx, match_expr.info(), "match") |bcx| {
        trans_match_inner(bcx, discr_expr, arms, dest)
    }
}

pub fn trans_match_inner(scope_cx: block,
                         discr_expr: @ast::expr,
                         arms: &[ast::arm],
                         dest: Dest) -> block {
    let _icx = scope_cx.insn_ctxt("match::trans_match_inner");
    let mut bcx = scope_cx;
    let tcx = bcx.tcx();

    let discr_datum = unpack_datum!(bcx, {
        expr::trans_to_datum(bcx, discr_expr)
    });
    if bcx.unreachable {
        return bcx;
    }

    let mut arm_datas = ~[], matches = ~[];
    for arms.each |arm| {
        let body = scope_block(bcx, arm.body.info(), "case_body");

        // Create the bindings map, which is a mapping from each binding name
        // to an alloca() that will be the value for that local variable.
        // Note that we use the names because each binding will have many ids
        // from the various alternatives.
        let mut bindings_map = HashMap::new();
        do pat_bindings(tcx.def_map, arm.pats[0]) |bm, p_id, _s, path| {
            let ident = path_to_ident(path);
            let variable_ty = node_id_type(bcx, p_id);
            let llvariable_ty = type_of::type_of(bcx.ccx(), variable_ty);

            let llmatch, trmode;
            match bm {
                ast::bind_by_copy | ast::bind_infer => {
                    // in this case, the final type of the variable will be T,
                    // but during matching we need to store a *T as explained
                    // above
                    let is_move =
                        scope_cx.ccx().maps.moves_map.contains(&p_id);
                    llmatch = alloca(bcx, T_ptr(llvariable_ty));
                    trmode = TrByValue(is_move, alloca(bcx, llvariable_ty));
                }
                ast::bind_by_ref(_) => {
                    llmatch = alloca(bcx, llvariable_ty);
                    trmode = TrByRef;
                }
            };
            bindings_map.insert(ident, BindingInfo {
                llmatch: llmatch, trmode: trmode,
                id: p_id, ty: variable_ty
            });
        }

        let arm_data = @ArmData {bodycx: body,
                                 arm: arm,
                                 bindings_map: bindings_map};
        arm_datas.push(arm_data);
        for arm.pats.each |p| {
            matches.push(@Match {pats: ~[*p], data: arm_data});
        }
    }

    let t = node_id_type(bcx, discr_expr.id);
    let chk = {
        if ty::type_is_empty(tcx, t) {
            // Special case for empty types
            let fail_cx = @mut None;
            let f: mk_fail = || mk_fail(scope_cx, discr_expr.span,
                            @~"scrutinizing value that can't exist", fail_cx);
            Some(f)
        } else {
            None
        }
    };
    let lldiscr = discr_datum.to_ref_llval(bcx);
    compile_submatch(bcx, matches, ~[lldiscr], chk);

    let mut arm_cxs = ~[];
    for arm_datas.each |arm_data| {
        let mut bcx = arm_data.bodycx;

        // If this arm has a guard, then the various by-value bindings have
        // already been copied into their homes.  If not, we do it here.  This
        // is just to reduce code space.  See extensive comment at the start
        // of the file for more details.
        if arm_data.arm.guard.is_none() {
            bcx = store_non_ref_bindings(bcx, *arm_data, None);
        }

        // insert bindings into the lllocals map and add cleanups
        bcx = insert_lllocals(bcx, *arm_data, true);

        bcx = controlflow::trans_block(bcx, &arm_data.arm.body, dest);
        bcx = trans_block_cleanups(bcx, block_cleanups(arm_data.bodycx));
        arm_cxs.push(bcx);
    }

    bcx = controlflow::join_blocks(scope_cx, arm_cxs);
    return bcx;

    fn mk_fail(bcx: block, sp: span, msg: @~str,
               finished: @mut Option<BasicBlockRef>) -> BasicBlockRef {
        match *finished { Some(bb) => return bb, _ => () }
        let fail_cx = sub_block(bcx, "case_fallthrough");
        controlflow::trans_fail(fail_cx, Some(sp), msg);
        *finished = Some(fail_cx.llbb);
        return fail_cx.llbb;
    }
}

pub enum IrrefutablePatternBindingMode {
    // Stores the association between node ID and LLVM value in `lllocals`.
    BindLocal,
    // Stores the association between node ID and LLVM value in `llargs`.
    BindArgument
}

// Not match-related, but similar to the pattern-munging code above
pub fn bind_irrefutable_pat(bcx: block,
                            pat: @ast::pat,
                            val: ValueRef,
                            make_copy: bool,
                            binding_mode: IrrefutablePatternBindingMode)
                         -> block {
    let _icx = bcx.insn_ctxt("match::bind_irrefutable_pat");
    let ccx = *bcx.fcx.ccx;
    let mut bcx = bcx;

    // Necessary since bind_irrefutable_pat is called outside trans_match
    match pat.node {
        ast::pat_ident(_, _, ref inner) => {
            if pat_is_variant_or_struct(bcx.tcx().def_map, pat) {
                return bcx;
            }

            if make_copy {
                let binding_ty = node_id_type(bcx, pat.id);
                let datum = Datum {val: val, ty: binding_ty,
                                   mode: ByRef, source: RevokeClean};
                let scratch = scratch_datum(bcx, binding_ty, false);
                datum.copy_to_datum(bcx, INIT, scratch);
                match binding_mode {
                    BindLocal => {
                        bcx.fcx.lllocals.insert(pat.id,
                                                local_mem(scratch.val));
                    }
                    BindArgument => {
                        bcx.fcx.llargs.insert(pat.id,
                                              local_mem(scratch.val));
                    }
                }
                add_clean(bcx, scratch.val, binding_ty);
            } else {
                match binding_mode {
                    BindLocal => {
                        bcx.fcx.lllocals.insert(pat.id, local_mem(val));
                    }
                    BindArgument => {
                        bcx.fcx.llargs.insert(pat.id, local_mem(val));
                    }
                }
            }

            for inner.each |inner_pat| {
                bcx = bind_irrefutable_pat(
                    bcx, *inner_pat, val, true, binding_mode);
            }
        }
        ast::pat_enum(_, ref sub_pats) => {
            match bcx.tcx().def_map.find(&pat.id) {
                Some(&ast::def_variant(enum_id, var_id)) => {
                    let repr = adt::represent_node(bcx, pat.id);
                    let vinfo = ty::enum_variant_with_id(ccx.tcx,
                                                         enum_id,
                                                         var_id);
                    let args = extract_variant_args(bcx,
                                                    repr,
                                                    vinfo.disr_val,
                                                    val);
                    for sub_pats.each |sub_pat| {
                        for args.vals.eachi |i, argval| {
                            bcx = bind_irrefutable_pat(bcx,
                                                       sub_pat[i],
                                                       *argval,
                                                       make_copy,
                                                       binding_mode);
                        }
                    }
                }
                Some(&ast::def_fn(*)) |
                Some(&ast::def_struct(*)) => {
                    match *sub_pats {
                        None => {
                            // This is a unit-like struct. Nothing to do here.
                        }
                        Some(ref elems) => {
                            // This is the tuple struct case.
                            let repr = adt::represent_node(bcx, pat.id);
                            for elems.eachi |i, elem| {
                                let fldptr = adt::trans_field_ptr(bcx, repr,
                                                            val, 0, i);
                                bcx = bind_irrefutable_pat(bcx,
                                                           *elem,
                                                           fldptr,
                                                           make_copy,
                                                           binding_mode);
                            }
                        }
                    }
                }
                Some(&ast::def_const(*)) => {
                    bcx = bind_irrefutable_pat(bcx, pat, val, make_copy, binding_mode);
                }
                _ => {
                    // Nothing to do here.
                }
            }
        }
        ast::pat_struct(_, ref fields, _) => {
            let tcx = bcx.tcx();
            let pat_ty = node_id_type(bcx, pat.id);
            let pat_repr = adt::represent_type(bcx.ccx(), pat_ty);
            do expr::with_field_tys(tcx, pat_ty, None) |discr, field_tys| {
                for fields.each |f| {
                    let ix = ty::field_idx_strict(tcx, f.ident, field_tys);
                    let fldptr = adt::trans_field_ptr(bcx, pat_repr, val,
                                                discr, ix);
                    bcx = bind_irrefutable_pat(bcx,
                                               f.pat,
                                               fldptr,
                                               make_copy,
                                               binding_mode);
                }
            }
        }
        ast::pat_tup(ref elems) => {
            let repr = adt::represent_node(bcx, pat.id);
            for elems.eachi |i, elem| {
                let fldptr = adt::trans_field_ptr(bcx, repr, val, 0, i);
                bcx = bind_irrefutable_pat(bcx,
                                           *elem,
                                           fldptr,
                                           make_copy,
                                           binding_mode);
            }
        }
        ast::pat_box(inner) | ast::pat_uniq(inner) => {
            let llbox = Load(bcx, val);
            let unboxed = GEPi(bcx, llbox, [0u, abi::box_field_body]);
            bcx = bind_irrefutable_pat(bcx,
                                       inner,
                                       unboxed,
                                       true,
                                       binding_mode);
        }
        ast::pat_region(inner) => {
            let loaded_val = Load(bcx, val);
            bcx = bind_irrefutable_pat(bcx,
                                       inner,
                                       loaded_val,
                                       true,
                                       binding_mode);
        }
        ast::pat_wild | ast::pat_lit(_) | ast::pat_range(_, _) |
        ast::pat_vec(*) => ()
    }
    return bcx;
}
