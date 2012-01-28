import core::{str, vec, option};
import option::{some, none};

import driver::session::session;
import lib::llvm::llvm;
import lib::llvm::llvm::{ValueRef, BasicBlockRef};
import pat_util::*;
import build::*;
import base::{new_sub_block_ctxt, new_scope_block_ctxt,
              new_real_block_ctxt, load_if_immediate};
import syntax::ast;
import syntax::ast_util;
import syntax::ast_util::{dummy_sp};
import syntax::ast::def_id;
import syntax::codemap::span;
import syntax::print::pprust::pat_to_str;

import common::*;

// An option identifying a branch (either a literal, a enum variant or a
// range)
enum opt {
    lit(@ast::expr),
    var(/* disr val */int, /* variant dids */{tg: def_id, var: def_id}),
    range(@ast::expr, @ast::expr)
}
fn opt_eq(a: opt, b: opt) -> bool {
    alt (a, b) {
      (lit(a), lit(b)) { ast_util::compare_lit_exprs(a, b) == 0 }
      (range(a1, a2), range(b1, b2)) {
        ast_util::compare_lit_exprs(a1, b1) == 0 &&
        ast_util::compare_lit_exprs(a2, b2) == 0
      }
      (var(a, _), var(b, _)) { a == b }
      _ { false }
    }
}

enum opt_result {
    single_result(result),
    range_result(result, result),
}
fn trans_opt(bcx: @block_ctxt, o: opt) -> opt_result {
    let ccx = bcx_ccx(bcx), bcx = bcx;
    alt o {
      lit(l) {
        alt l.node {
          ast::expr_lit(@{node: ast::lit_str(s), _}) {
            let strty = ty::mk_str(bcx_tcx(bcx));
            let cell = base::empty_dest_cell();
            bcx = tvec::trans_str(bcx, s, base::by_val(cell));
            add_clean_temp(bcx, *cell, strty);
            ret single_result(rslt(bcx, *cell));
          }
          _ {
            ret single_result(
                rslt(bcx, base::trans_const_expr(ccx, l)));
          }
        }
      }
      var(disr_val, _) { ret single_result(rslt(bcx, C_int(ccx, disr_val))); }
      range(l1, l2) {
        ret range_result(rslt(bcx, base::trans_const_expr(ccx, l1)),
                         rslt(bcx, base::trans_const_expr(ccx, l2)));
      }
    }
}

// FIXME: invariant -- pat_id is bound in the def_map?
fn variant_opt(ccx: @crate_ctxt, pat_id: ast::node_id) -> opt {
    let vdef = ast_util::variant_def_ids(ccx.tcx.def_map.get(pat_id));
    let variants = ty::enum_variants(ccx.tcx, vdef.tg);
    for v: ty::variant_info in *variants {
        if vdef.var == v.id { ret var(v.disr_val, vdef); }
    }
    fail;
}

type bind_map = [{ident: ast::ident, val: ValueRef}];
fn assoc(key: str, list: bind_map) -> option::t<ValueRef> {
    for elt: {ident: ast::ident, val: ValueRef} in list {
        if str::eq(elt.ident, key) { ret some(elt.val); }
    }
    ret none;
}

type match_branch =
    @{pats: [@ast::pat],
      bound: bind_map,
      data: @{body: BasicBlockRef,
              guard: option::t<@ast::expr>,
              id_map: pat_id_map}};
type match = [match_branch];

fn has_nested_bindings(m: match, col: uint) -> bool {
    for br in m {
        alt br.pats[col].node {
          ast::pat_ident(_, some(_)) { ret true; }
          _ {}
        }
    }
    ret false;
}

fn expand_nested_bindings(m: match, col: uint, val: ValueRef) -> match {
    let result = [];
    for br in m {
      alt br.pats[col].node {
          ast::pat_ident(name, some(inner)) {
            let pats = vec::slice(br.pats, 0u, col) + [inner] +
                vec::slice(br.pats, col + 1u, vec::len(br.pats));
            result += [@{pats: pats,
                        bound: br.bound + [{ident: path_to_ident(name),
                                val: val}]
                         with *br}];
          }
          _ { result += [br]; }
        }
    }
    result
}

type enter_pat = fn@(@ast::pat) -> option::t<[@ast::pat]>;

fn enter_match(m: match, col: uint, val: ValueRef, e: enter_pat) -> match {
    let result = [];
    for br: match_branch in m {
        alt e(br.pats[col]) {
          some(sub) {
            let pats = sub + vec::slice(br.pats, 0u, col) +
                vec::slice(br.pats, col + 1u, vec::len(br.pats));
            let new_br = @{pats: pats,
                           bound: alt br.pats[col].node {
                             ast::pat_ident(name, none) {
                                 br.bound + [{ident: path_to_ident(name),
                                              val: val}]
                             }
                             _ { br.bound }
                           } with *br};
            result += [new_br];
          }
          none { }
        }
    }
    ret result;
}

fn enter_default(m: match, col: uint, val: ValueRef) -> match {
    fn matches_always(p: @ast::pat) -> bool {
        alt p.node {
                ast::pat_wild | ast::pat_rec(_, _) |
                ast::pat_ident(_, none) | ast::pat_tup(_) { true }
                _ { false }
        }
    }
    fn e(p: @ast::pat) -> option::t<[@ast::pat]> {
        ret if matches_always(p) { some([]) } else { none };
    }
    ret enter_match(m, col, val, e);
}

fn enter_opt(ccx: @crate_ctxt, m: match, opt: opt, col: uint, enum_size: uint,
             val: ValueRef) -> match {
    let dummy = @{id: 0, node: ast::pat_wild, span: dummy_sp()};
    fn e(ccx: @crate_ctxt, dummy: @ast::pat, opt: opt, size: uint,
         p: @ast::pat) -> option::t<[@ast::pat]> {
        alt p.node {
          ast::pat_enum(ctor, subpats) {
            ret if opt_eq(variant_opt(ccx, p.id), opt) {
                    some(subpats)
                } else { none };
          }
          ast::pat_lit(l) {
            ret if opt_eq(lit(l), opt) { some([]) } else { none };
          }
          ast::pat_range(l1, l2) {
            ret if opt_eq(range(l1, l2), opt) { some([]) } else { none };
          }
          _ { ret some(vec::init_elt(size, dummy)); }
        }
    }
    ret enter_match(m, col, val, bind e(ccx, dummy, opt, enum_size, _));
}

fn enter_rec(m: match, col: uint, fields: [ast::ident], val: ValueRef) ->
   match {
    let dummy = @{id: 0, node: ast::pat_wild, span: dummy_sp()};
    fn e(dummy: @ast::pat, fields: [ast::ident], p: @ast::pat) ->
       option::t<[@ast::pat]> {
        alt p.node {
          ast::pat_rec(fpats, _) {
            let pats = [];
            for fname: ast::ident in fields {
                let pat = dummy;
                for fpat: ast::field_pat in fpats {
                    if str::eq(fpat.ident, fname) { pat = fpat.pat; break; }
                }
                pats += [pat];
            }
            ret some(pats);
          }
          _ { ret some(vec::init_elt(vec::len(fields), dummy)); }
        }
    }
    ret enter_match(m, col, val, bind e(dummy, fields, _));
}

fn enter_tup(m: match, col: uint, val: ValueRef, n_elts: uint) -> match {
    let dummy = @{id: 0, node: ast::pat_wild, span: dummy_sp()};
    fn e(dummy: @ast::pat, n_elts: uint, p: @ast::pat) ->
       option::t<[@ast::pat]> {
        alt p.node {
          ast::pat_tup(elts) { ret some(elts); }
          _ { ret some(vec::init_elt(n_elts, dummy)); }
        }
    }
    ret enter_match(m, col, val, bind e(dummy, n_elts, _));
}

fn enter_box(m: match, col: uint, val: ValueRef) -> match {
    let dummy = @{id: 0, node: ast::pat_wild, span: dummy_sp()};
    fn e(dummy: @ast::pat, p: @ast::pat) -> option::t<[@ast::pat]> {
        alt p.node {
          ast::pat_box(sub) { ret some([sub]); }
          _ { ret some([dummy]); }
        }
    }
    ret enter_match(m, col, val, bind e(dummy, _));
}

fn enter_uniq(m: match, col: uint, val: ValueRef) -> match {
    let dummy = @{id: 0, node: ast::pat_wild, span: dummy_sp()};
    fn e(dummy: @ast::pat, p: @ast::pat) -> option::t<[@ast::pat]> {
        alt p.node {
          ast::pat_uniq(sub) { ret some([sub]); }
          _ { ret some([dummy]); }
        }
    }
    ret enter_match(m, col, val, bind e(dummy, _));
}

fn get_options(ccx: @crate_ctxt, m: match, col: uint) -> [opt] {
    fn add_to_set(&set: [opt], val: opt) {
        for l: opt in set { if opt_eq(l, val) { ret; } }
        set += [val];
    }

    let found = [];
    for br: match_branch in m {
        alt br.pats[col].node {
          ast::pat_lit(l) { add_to_set(found, lit(l)); }
          ast::pat_range(l1, l2) {
            add_to_set(found, range(l1, l2));
          }
          ast::pat_enum(_, _) {
            add_to_set(found, variant_opt(ccx, br.pats[col].id));
          }
          _ { }
        }
    }
    ret found;
}

fn extract_variant_args(bcx: @block_ctxt, pat_id: ast::node_id,
                        vdefs: {tg: def_id, var: def_id}, val: ValueRef) ->
   {vals: [ValueRef], bcx: @block_ctxt} {
    let ccx = bcx.fcx.lcx.ccx, bcx = bcx;
    // invariant:
    // pat_id must have the same length ty_param_substs as vdefs?
    let ty_param_substs = ty::node_id_to_type_params(ccx.tcx, pat_id);
    let blobptr = val;
    let variants = ty::enum_variants(ccx.tcx, vdefs.tg);
    let args = [];
    let size =
        vec::len(ty::enum_variant_with_id(ccx.tcx, vdefs.tg, vdefs.var).args);
    if size > 0u && vec::len(*variants) != 1u {
        let enumptr =
            PointerCast(bcx, val, T_opaque_enum_ptr(ccx));
        blobptr = GEPi(bcx, enumptr, [0, 1]);
    }
    let i = 0u;
    let vdefs_tg = vdefs.tg;
    let vdefs_var = vdefs.var;
    while i < size {
        check (valid_variant_index(i, bcx, vdefs_tg, vdefs_var));
        let r =
            // invariant needed:
            // how do we know it even makes sense to pass in ty_param_substs
            // here? What if it's [] and the enum type has variables in it?
            base::GEP_enum(bcx, blobptr, vdefs_tg, vdefs_var,
                            ty_param_substs, i);
        bcx = r.bcx;
        args += [r.val];
        i += 1u;
    }
    ret {vals: args, bcx: bcx};
}

fn collect_record_fields(m: match, col: uint) -> [ast::ident] {
    let fields = [];
    for br: match_branch in m {
        alt br.pats[col].node {
          ast::pat_rec(fs, _) {
            for f: ast::field_pat in fs {
                if !vec::any(fields, bind str::eq(f.ident, _)) {
                    fields += [f.ident];
                }
            }
          }
          _ { }
        }
    }
    ret fields;
}

fn any_box_pat(m: match, col: uint) -> bool {
    for br: match_branch in m {
        alt br.pats[col].node { ast::pat_box(_) { ret true; } _ { } }
    }
    ret false;
}

fn any_uniq_pat(m: match, col: uint) -> bool {
    for br: match_branch in m {
        alt br.pats[col].node { ast::pat_uniq(_) { ret true; } _ { } }
    }
    ret false;
}

fn any_tup_pat(m: match, col: uint) -> bool {
    for br: match_branch in m {
        alt br.pats[col].node { ast::pat_tup(_) { ret true; } _ { } }
    }
    ret false;
}

type exit_node = {bound: bind_map, from: BasicBlockRef, to: BasicBlockRef};
type mk_fail = fn@() -> BasicBlockRef;

fn pick_col(m: match) -> uint {
    fn score(p: @ast::pat) -> uint {
        alt p.node {
          ast::pat_lit(_) | ast::pat_enum(_, _) | ast::pat_range(_, _) { 1u }
          ast::pat_ident(_, some(p)) { score(p) }
          _ { 0u }
        }
    }
    let scores = vec::init_elt_mut(vec::len(m[0].pats), 0u);
    for br: match_branch in m {
        let i = 0u;
        for p: @ast::pat in br.pats { scores[i] += score(p); i += 1u; }
    }
    let max_score = 0u;
    let best_col = 0u;
    let i = 0u;
    for score: uint in scores {
        // Irrefutable columns always go first, they'd only be duplicated in
        // the branches.
        if score == 0u { ret i; }
        // If no irrefutable ones are found, we pick the one with the biggest
        // branching factor.
        if score > max_score { max_score = score; best_col = i; }
        i += 1u;
    }
    ret best_col;
}

fn compile_submatch(bcx: @block_ctxt, m: match, vals: [ValueRef], f: mk_fail,
                    &exits: [exit_node]) {
    let bcx = bcx;
    if vec::len(m) == 0u { Br(bcx, f()); ret; }
    if vec::len(m[0].pats) == 0u {
        let data = m[0].data;
        alt data.guard {
          some(e) {
            let guard_cx = new_scope_block_ctxt(bcx, "submatch_guard");
            Br(bcx, guard_cx.llbb);
            // Temporarily set bindings. They'll be rewritten to PHI nodes for
            // the actual arm block.
            data.id_map.items {|key, val|
                let local = local_mem(option::get(assoc(key, m[0].bound)));
                bcx.fcx.lllocals.insert(val, local);
            };
            let {bcx: guard_bcx, val: guard_val} =
                base::trans_temp_expr(guard_cx, e);
            guard_bcx = base::trans_block_cleanups(guard_bcx, guard_cx);
            let next_cx = new_sub_block_ctxt(guard_cx, "submatch_next");
            let else_cx = new_sub_block_ctxt(guard_cx, "submatch_else");
            CondBr(guard_bcx, guard_val, next_cx.llbb, else_cx.llbb);
            compile_submatch(else_cx, vec::slice(m, 1u, vec::len(m)), vals, f,
                             exits);
            bcx = next_cx;
          }
          _ { }
        }
        if !bcx.unreachable {
            exits += [{bound: m[0].bound, from: bcx.llbb, to: data.body}];
        }
        Br(bcx, data.body);
        ret;
    }

    let col = pick_col(m);
    let val = vals[col];
    let m = has_nested_bindings(m, col) ?
        expand_nested_bindings(m, col, val) : m;

    let vals_left =
        vec::slice(vals, 0u, col) +
            vec::slice(vals, col + 1u, vec::len(vals));
    let ccx = bcx.fcx.lcx.ccx;
    let pat_id = 0;
    for br: match_branch in m {
        // Find a real id (we're adding placeholder wildcard patterns, but
        // each column is guaranteed to have at least one real pattern)
        if pat_id == 0 { pat_id = br.pats[col].id; }
    }

    let rec_fields = collect_record_fields(m, col);
    // Separate path for extracting and binding record fields
    if vec::len(rec_fields) > 0u {
        let rec_ty = ty::node_id_to_monotype(ccx.tcx, pat_id);
        let fields =
            alt ty::struct(ccx.tcx, rec_ty) { ty::ty_rec(fields) { fields } };
        let rec_vals = [];
        for field_name: ast::ident in rec_fields {
            let ix = option::get(ty::field_idx(field_name, fields));
            // not sure how to get rid of this check
            check type_is_tup_like(bcx, rec_ty);
            let r = base::GEP_tup_like(bcx, rec_ty, val, [0, ix as int]);
            rec_vals += [r.val];
            bcx = r.bcx;
        }
        compile_submatch(bcx, enter_rec(m, col, rec_fields, val),
                         rec_vals + vals_left, f, exits);
        ret;
    }

    if any_tup_pat(m, col) {
        let tup_ty = ty::node_id_to_monotype(ccx.tcx, pat_id);
        let n_tup_elts =
            alt ty::struct(ccx.tcx, tup_ty) {
              ty::ty_tup(elts) { vec::len(elts) }
            };
        let tup_vals = [], i = 0u;
        while i < n_tup_elts {
            // how to get rid of this check?
            check type_is_tup_like(bcx, tup_ty);
            let r = base::GEP_tup_like(bcx, tup_ty, val, [0, i as int]);
            tup_vals += [r.val];
            bcx = r.bcx;
            i += 1u;
        }
        compile_submatch(bcx, enter_tup(m, col, val, n_tup_elts),
                         tup_vals + vals_left, f, exits);
        ret;
    }

    // Unbox in case of a box field
    if any_box_pat(m, col) {
        let box = Load(bcx, val);
        let unboxed = GEPi(bcx, box, [0, back::abi::box_rc_field_body]);
        compile_submatch(bcx, enter_box(m, col, val), [unboxed] + vals_left,
                         f, exits);
        ret;
    }

    if any_uniq_pat(m, col) {
        let unboxed = Load(bcx, val);
        compile_submatch(bcx, enter_uniq(m, col, val),
                         [unboxed] + vals_left, f, exits);
        ret;
    }

    // Decide what kind of branch we need
    let opts = get_options(ccx, m, col);
    enum branch_kind { no_branch, single, switch, compare, }
    let kind = no_branch;
    let test_val = val;
    if vec::len(opts) > 0u {
        alt opts[0] {
          var(_, vdef) {
            if vec::len(*ty::enum_variants(ccx.tcx, vdef.tg)) == 1u {
                kind = single;
            } else {
                let enumptr =
                    PointerCast(bcx, val, T_opaque_enum_ptr(ccx));
                let discrimptr = GEPi(bcx, enumptr, [0, 0]);
                test_val = Load(bcx, discrimptr);
                kind = switch;
            }
          }
          lit(l) {
            test_val = Load(bcx, val);
            let pty = ty::node_id_to_monotype(ccx.tcx, pat_id);
            kind = ty::type_is_integral(ccx.tcx, pty) ? switch : compare;
          }
          range(_, _) {
            test_val = Load(bcx, val);
            kind = compare;
          }
        }
    }
    for o: opt in opts {
        alt o {
          range(_, _) { kind = compare; break; }
          _ { }
        }
    }
    let else_cx =
        alt kind {
          no_branch | single { bcx }
          _ { new_sub_block_ctxt(bcx, "match_else") }
        };
    let sw;
    if kind == switch {
        sw = Switch(bcx, test_val, else_cx.llbb, vec::len(opts));
        // FIXME This statement is purely here as a work-around for a bug that
        // I expect to be the same as issue #951. If I remove it, sw ends up
        // holding a corrupted value (when the compiler is optimized).
        // This can be removed after our next LLVM upgrade.
        val_ty(sw);
    } else { sw = C_int(ccx, 0); } // Placeholder for when not using a switch

     // Compile subtrees for each option
    for opt: opt in opts {
        let opt_cx = new_sub_block_ctxt(bcx, "match_case");
        alt kind {
          single { Br(bcx, opt_cx.llbb); }
          switch {
            let res = trans_opt(bcx, opt);
            alt res {
              single_result(r) {
                llvm::LLVMAddCase(sw, r.val, opt_cx.llbb);
                bcx = r.bcx;
              }
            }
          }
          compare {
            let compare_cx = new_scope_block_ctxt(bcx, "compare_scope");
            Br(bcx, compare_cx.llbb);
            bcx = compare_cx;
            let t = ty::node_id_to_type(ccx.tcx, pat_id);
            let res = trans_opt(bcx, opt);
            alt res {
              single_result(r) {
                bcx = r.bcx;
                let eq =
                    base::trans_compare(bcx, ast::eq, test_val, t, r.val, t);
                let cleanup_cx = base::trans_block_cleanups(
                    eq.bcx, compare_cx);
                bcx = new_sub_block_ctxt(bcx, "compare_next");
                CondBr(cleanup_cx, eq.val, opt_cx.llbb, bcx.llbb);
              }
              range_result(rbegin, rend) {
                bcx = rend.bcx;
                let ge = base::trans_compare(bcx, ast::ge, test_val, t,
                                              rbegin.val, t);
                let le = base::trans_compare(ge.bcx, ast::le, test_val, t,
                                              rend.val, t);
                let in_range = rslt(le.bcx, And(le.bcx, ge.val, le.val));
                bcx = in_range.bcx;
                let cleanup_cx =
                    base::trans_block_cleanups(bcx, compare_cx);
                bcx = new_sub_block_ctxt(bcx, "compare_next");
                CondBr(cleanup_cx, in_range.val, opt_cx.llbb, bcx.llbb);
              }
            }
          }
          _ { }
        }
        let size = 0u;
        let unpacked = [];
        alt opt {
          var(_, vdef) {
            let args = extract_variant_args(opt_cx, pat_id, vdef, val);
            size = vec::len(args.vals);
            unpacked = args.vals;
            opt_cx = args.bcx;
          }
          lit(_) | range(_, _) { }
        }
        compile_submatch(opt_cx, enter_opt(ccx, m, opt, col, size, val),
                         unpacked + vals_left, f, exits);
    }

    // Compile the fall-through case
    if kind == compare { Br(bcx, else_cx.llbb); }
    if kind != single {
        compile_submatch(else_cx, enter_default(m, col, val), vals_left, f,
                         exits);
    }
}

// Returns false for unreachable blocks
fn make_phi_bindings(bcx: @block_ctxt, map: [exit_node],
                     ids: pat_util::pat_id_map) -> bool {
    let our_block = bcx.llbb as uint;
    let success = true, bcx = bcx;
    ids.items {|name, node_id|
        let llbbs = [];
        let vals = [];
        for ex: exit_node in map {
            if ex.to as uint == our_block {
                alt assoc(name, ex.bound) {
                  some(val) { llbbs += [ex.from]; vals += [val]; }
                  none { }
                }
            }
        }
        if vec::len(vals) > 0u {
            let local = Phi(bcx, val_ty(vals[0]), vals, llbbs);
            bcx.fcx.lllocals.insert(node_id, local_mem(local));
        } else { success = false; }
    };
    if success {
        // Copy references that the alias analysis considered unsafe
        ids.values {|node_id|
            if bcx_ccx(bcx).copy_map.contains_key(node_id) {
                let local = alt bcx.fcx.lllocals.get(node_id) {
                  local_mem(x) { x }
                };
                let e_ty = ty::node_id_to_type(bcx_tcx(bcx), node_id);
                let {bcx: abcx, val: alloc} = base::alloc_ty(bcx, e_ty);
                bcx = base::copy_val(abcx, base::INIT, alloc,
                                      load_if_immediate(abcx, local, e_ty),
                                      e_ty);
                add_clean(bcx, alloc, e_ty);
                bcx.fcx.lllocals.insert(node_id, local_mem(alloc));
            }
        };
    } else {
        Unreachable(bcx);
    }
    ret success;
}

fn trans_alt(cx: @block_ctxt, expr: @ast::expr, arms_: [ast::arm],
             dest: base::dest) -> @block_ctxt {
    let bodies = [];
    let match: match = [];
    let alt_cx = new_scope_block_ctxt(cx, "alt");
    Br(cx, alt_cx.llbb);

    let er = base::trans_temp_expr(alt_cx, expr);
    if er.bcx.unreachable { ret er.bcx; }

    /*
      n.b. nothing else in this module should need to normalize,
      b/c of this call
     */
    let arms = normalize_arms(bcx_tcx(cx), arms_);

    for a: ast::arm in arms {
        let body = new_real_block_ctxt(er.bcx, "case_body",
                                       a.body.span);
        let id_map = pat_util::pat_id_map(bcx_tcx(cx), a.pats[0]);
        bodies += [body];
        for p: @ast::pat in a.pats {
            match +=
                [@{pats: [p],
                   bound: [],
                   data: @{body: body.llbb, guard: a.guard, id_map: id_map}}];
        }
    }

    // Cached fail-on-fallthrough block
    let fail_cx = @mutable none;
    fn mk_fail(cx: @block_ctxt, sp: span,
               done: @mutable option::t<BasicBlockRef>) -> BasicBlockRef {
        alt *done { some(bb) { ret bb; } _ { } }
        let fail_cx = new_sub_block_ctxt(cx, "case_fallthrough");
        base::trans_fail(fail_cx, some(sp), "non-exhaustive match failure");;
        *done = some(fail_cx.llbb);
        ret fail_cx.llbb;
    }

    let exit_map = [];
    let t = base::node_id_type(cx.fcx.lcx.ccx, expr.id);
    let vr = base::spill_if_immediate(er.bcx, er.val, t);
    compile_submatch(vr.bcx, match, [vr.val],
                     bind mk_fail(alt_cx, expr.span, fail_cx), exit_map);

    let arm_cxs = [], arm_dests = [], i = 0u;
    for a: ast::arm in arms {
        let body_cx = bodies[i];
        if make_phi_bindings(body_cx, exit_map,
                             pat_util::pat_id_map(bcx_tcx(cx),
                                                  a.pats[0])) {
            let arm_dest = base::dup_for_join(dest);
            arm_dests += [arm_dest];
            arm_cxs += [base::trans_block_dps(body_cx, a.body, arm_dest)];
        }
        i += 1u;
    }
    let after_cx = base::join_returns(cx, arm_cxs, arm_dests, dest);
    after_cx = base::trans_block_cleanups(after_cx, alt_cx);
    let next_cx = new_sub_block_ctxt(after_cx, "next");
    Br(after_cx, next_cx.llbb);
    ret next_cx;
}

// Not alt-related, but similar to the pattern-munging code above
fn bind_irrefutable_pat(bcx: @block_ctxt, pat: @ast::pat, val: ValueRef,
                        make_copy: bool) -> @block_ctxt {
    let ccx = bcx.fcx.lcx.ccx, bcx = bcx;

    // Necessary since bind_irrefutable_pat is called outside trans_alt
    alt normalize_pat(bcx_tcx(bcx), pat).node {
      ast::pat_ident(_,inner) {
        if make_copy || ccx.copy_map.contains_key(pat.id) {
            let ty = ty::node_id_to_monotype(ccx.tcx, pat.id);
            // FIXME: Could constrain pat_bind to make this
            // check unnecessary.
            check (type_has_static_size(ccx, ty));
            check non_ty_var(ccx, ty);
            let llty = base::type_of(ccx, ty);
            let alloc = base::alloca(bcx, llty);
            bcx = base::copy_val(bcx, base::INIT, alloc,
                                  base::load_if_immediate(bcx, val, ty), ty);
            bcx.fcx.lllocals.insert(pat.id, local_mem(alloc));
            add_clean(bcx, alloc, ty);
        } else { bcx.fcx.lllocals.insert(pat.id, local_mem(val)); }
        alt inner {
          some(pat) { bcx = bind_irrefutable_pat(bcx, pat, val, true); }
          _ {}
        }
      }
      ast::pat_enum(_, sub) {
        if vec::len(sub) == 0u { ret bcx; }
        let vdefs = ast_util::variant_def_ids(ccx.tcx.def_map.get(pat.id));
        let args = extract_variant_args(bcx, pat.id, vdefs, val);
        let i = 0;
        for argval: ValueRef in args.vals {
            bcx = bind_irrefutable_pat(bcx, sub[i], argval, make_copy);
            i += 1;
        }
      }
      ast::pat_rec(fields, _) {
        let rec_ty = ty::node_id_to_monotype(ccx.tcx, pat.id);
        let rec_fields =
            alt ty::struct(ccx.tcx, rec_ty) { ty::ty_rec(fields) { fields } };
        for f: ast::field_pat in fields {
            let ix = option::get(ty::field_idx(f.ident, rec_fields));
            // how to get rid of this check?
            check type_is_tup_like(bcx, rec_ty);
            let r = base::GEP_tup_like(bcx, rec_ty, val, [0, ix as int]);
            bcx = bind_irrefutable_pat(r.bcx, f.pat, r.val, make_copy);
        }
      }
      ast::pat_tup(elems) {
        let tup_ty = ty::node_id_to_monotype(ccx.tcx, pat.id);
        let i = 0u;
        for elem in elems {
            // how to get rid of this check?
            check type_is_tup_like(bcx, tup_ty);
            let r = base::GEP_tup_like(bcx, tup_ty, val, [0, i as int]);
            bcx = bind_irrefutable_pat(r.bcx, elem, r.val, make_copy);
            i += 1u;
        }
      }
      ast::pat_box(inner) {
        let box = Load(bcx, val);
        let unboxed =
            GEPi(bcx, box, [0, back::abi::box_rc_field_body]);
        bcx = bind_irrefutable_pat(bcx, inner, unboxed, true);
      }
      ast::pat_uniq(inner) {
        let val = Load(bcx, val);
        bcx = bind_irrefutable_pat(bcx, inner, val, true);
      }
      ast::pat_wild | ast::pat_lit(_) | ast::pat_range(_, _) { }
    }
    ret bcx;
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
