import std::str;
import std::vec;
import std::option;
import option::some;
import option::none;
import std::map::hashmap;

import lib::llvm::llvm;
import lib::llvm::llvm::ValueRef;
import lib::llvm::llvm::TypeRef;
import lib::llvm::llvm::BasicBlockRef;
import trans_build::*;
import trans::new_sub_block_ctxt;
import trans::new_scope_block_ctxt;
import trans::load_if_immediate;
import ty::pat_ty;
import syntax::ast;
import syntax::ast_util;
import syntax::ast_util::dummy_sp;
import syntax::ast::def_id;
import syntax::codemap::span;
import util::common::lit_eq;

import trans_common::*;

// An option identifying a branch (either a literal or a tag variant)
tag opt {
    lit(@ast::lit);
    var(/* variant id */uint, /* variant dids */{tg: def_id, var: def_id});
}
fn opt_eq(a: &opt, b: &opt) -> bool {
    alt a {
      lit(la) {
        ret alt b { lit(lb) { lit_eq(la, lb) } var(_, _) { false } };
      }
      var(ida, _) {
        ret alt b { lit(_) { false } var(idb, _) { ida == idb } };
      }
    }
}
fn trans_opt(bcx: &@block_ctxt, o: &opt) -> result {
    alt o {
      lit(l) { ret trans::trans_lit(bcx, *l); }
      var(id, _) { ret rslt(bcx, C_int(id as int)); }
    }
}

fn variant_opt(ccx: &@crate_ctxt, pat_id: ast::node_id) -> opt {
    let vdef = ast_util::variant_def_ids(ccx.tcx.def_map.get(pat_id));
    let variants = ty::tag_variants(ccx.tcx, vdef.tg);
    let i = 0u;
    for v: ty::variant_info in variants {
        if vdef.var == v.id { ret var(i, vdef); }
        i += 1u;
    }
    fail;
}

type bind_map = [{ident: ast::ident, val: ValueRef}];
fn assoc(key: &istr, list: &bind_map) -> option::t<ValueRef> {
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
              id_map: ast_util::pat_id_map}};
type match = [match_branch];

fn matches_always(p: &@ast::pat) -> bool {
    ret alt p.node {
          ast::pat_wild. { true }
          ast::pat_bind(_) { true }
          ast::pat_rec(_, _) { true }
          ast::pat_tup(_) { true }
          _ { false }
        };
}

type enter_pat = fn(&@ast::pat) -> option::t<[@ast::pat]>;

fn enter_match(m: &match, col: uint, val: ValueRef, e: &enter_pat) -> match {
    let result = [];
    for br: match_branch in m {
        alt e(br.pats[col]) {
          some(sub) {
            let pats = vec::slice(br.pats, 0u, col) + sub +
                    vec::slice(br.pats, col + 1u, vec::len(br.pats));
            let new_br = @{pats: pats,
                           bound: alt br.pats[col].node {
                             ast::pat_bind(name) {
                               br.bound + [{ident: name, val: val}]
                             }
                             _ { br.bound }
                           }
                           with *br};
            result += [new_br];
          }
          none. { }
        }
    }
    ret result;
}

fn enter_default(m: &match, col: uint, val: ValueRef) -> match {
    fn e(p: &@ast::pat) -> option::t<[@ast::pat]> {
        ret if matches_always(p) { some([]) } else { none };
    }
    ret enter_match(m, col, val, e);
}

fn enter_opt(ccx: &@crate_ctxt, m: &match, opt: &opt, col: uint,
             tag_size: uint, val: ValueRef) -> match {
    let dummy = @{id: 0, node: ast::pat_wild, span: dummy_sp()};
    fn e(ccx: &@crate_ctxt, dummy: &@ast::pat, opt: &opt, size: uint,
         p: &@ast::pat) -> option::t<[@ast::pat]> {
        alt p.node {
          ast::pat_tag(ctor, subpats) {
            ret if opt_eq(variant_opt(ccx, p.id), opt) {
                    some(subpats)
                } else { none };
          }
          ast::pat_lit(l) {
            ret if opt_eq(lit(l), opt) { some([]) } else { none };
          }
          _ { ret some(vec::init_elt(dummy, size)); }
        }
    }
    ret enter_match(m, col, val, bind e(ccx, dummy, opt, tag_size, _));
}

fn enter_rec(m: &match, col: uint, fields: &[ast::ident], val: ValueRef) ->
   match {
    let dummy = @{id: 0, node: ast::pat_wild, span: dummy_sp()};
    fn e(dummy: &@ast::pat, fields: &[ast::ident], p: &@ast::pat) ->
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
          _ { ret some(vec::init_elt(dummy, vec::len(fields))); }
        }
    }
    ret enter_match(m, col, val, bind e(dummy, fields, _));
}

fn enter_tup(m: &match, col: uint, val: ValueRef, n_elts: uint) -> match {
    let dummy = @{id: 0, node: ast::pat_wild, span: dummy_sp()};
    fn e(dummy: &@ast::pat, n_elts: uint, p: &@ast::pat) ->
       option::t<[@ast::pat]> {
        alt p.node {
          ast::pat_tup(elts) { ret some(elts); }
          _ { ret some(vec::init_elt(dummy, n_elts)); }
        }
    }
    ret enter_match(m, col, val, bind e(dummy, n_elts, _));
}

fn enter_box(m: &match, col: uint, val: ValueRef) -> match {
    let dummy = @{id: 0, node: ast::pat_wild, span: dummy_sp()};
    fn e(dummy: &@ast::pat, p: &@ast::pat) -> option::t<[@ast::pat]> {
        alt p.node {
          ast::pat_box(sub) { ret some([sub]); }
          _ { ret some([dummy]); }
        }
    }
    ret enter_match(m, col, val, bind e(dummy, _));
}

fn get_options(ccx: &@crate_ctxt, m: &match, col: uint) -> [opt] {
    fn add_to_set(set: &mutable [opt], val: &opt) {
        for l: opt in set { if opt_eq(l, val) { ret; } }
        set += [val];
    }

    let found = [];
    for br: match_branch in m {
        alt br.pats[col].node {
          ast::pat_lit(l) { add_to_set(found, lit(l)); }
          ast::pat_tag(_, _) {
            add_to_set(found, variant_opt(ccx, br.pats[col].id));
          }
          _ { }
        }
    }
    ret found;
}

fn extract_variant_args(bcx: @block_ctxt, pat_id: ast::node_id,
                        vdefs: &{tg: def_id, var: def_id}, val: ValueRef) ->
   {vals: [ValueRef], bcx: @block_ctxt} {
    let ccx = bcx.fcx.lcx.ccx;
    let ty_param_substs = ty::node_id_to_type_params(ccx.tcx, pat_id);
    let blobptr = val;
    let variants = ty::tag_variants(ccx.tcx, vdefs.tg);
    let args = [];
    let size =
        vec::len(ty::tag_variant_with_id(ccx.tcx, vdefs.tg, vdefs.var).args);
    if size > 0u && vec::len(variants) != 1u {
        let tagptr =
            PointerCast(bcx, val,
                                  trans_common::T_opaque_tag_ptr(ccx.tn));
        blobptr = GEP(bcx, tagptr, [C_int(0), C_int(1)]);
    }
    let i = 0u;
    while i < size {
        let r =
            trans::GEP_tag(bcx, blobptr, vdefs.tg, vdefs.var, ty_param_substs,
                           i);
        bcx = r.bcx;
        args += [r.val];
        i += 1u;
    }
    ret {vals: args, bcx: bcx};
}

fn collect_record_fields(m: &match, col: uint) -> [ast::ident] {
    let fields = [];
    for br: match_branch in m {
        alt br.pats[col].node {
          ast::pat_rec(fs, _) {
            for f: ast::field_pat in fs {
                if !vec::any(bind str::eq(f.ident, _), fields) {
                    fields += [f.ident];
                }
            }
          }
          _ { }
        }
    }
    ret fields;
}

fn any_box_pat(m: &match, col: uint) -> bool {
    for br: match_branch in m {
        alt br.pats[col].node { ast::pat_box(_) { ret true; } _ { } }
    }
    ret false;
}

fn any_tup_pat(m: &match, col: uint) -> bool {
    for br: match_branch in m {
        alt br.pats[col].node { ast::pat_tup(_) { ret true; } _ { } }
    }
    ret false;
}

type exit_node = {bound: bind_map, from: BasicBlockRef, to: BasicBlockRef};
type mk_fail = fn() -> BasicBlockRef;

fn pick_col(m: &match) -> uint {
    let scores = vec::init_elt_mut(0u, vec::len(m[0].pats));
    for br: match_branch in m {
        let i = 0u;
        for p: @ast::pat in br.pats {
            alt p.node {
              ast::pat_lit(_) | ast::pat_tag(_, _) { scores[i] += 1u; }
              _ { }
            }
            i += 1u;
        }
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

fn compile_submatch(bcx: @block_ctxt, m: &match, vals: [ValueRef],
                    f: &mk_fail, exits: &mutable [exit_node]) {
    if vec::len(m) == 0u { Br(bcx, f()); ret; }
    if vec::len(m[0].pats) == 0u {
        let data = m[0].data;
        alt data.guard {
          some(e) {
            let guard_cx = new_scope_block_ctxt(bcx, ~"submatch_guard");
            let next_cx = new_sub_block_ctxt(bcx, ~"submatch_next");
            let else_cx = new_sub_block_ctxt(bcx, ~"submatch_else");
            Br(bcx, guard_cx.llbb);
            // Temporarily set bindings. They'll be rewritten to PHI nodes for
            // the actual arm block.
            for each @{key, val} in data.id_map.items() {
                bcx.fcx.lllocals.insert
                    (val, option::get(assoc(key,
                                            m[0].bound)));
            }
            let {bcx: guard_bcx, val: guard_val} =
                trans::trans_expr(guard_cx, e);
            guard_bcx = trans::trans_block_cleanups(guard_bcx, guard_cx);
            CondBr(guard_bcx, guard_val, next_cx.llbb, else_cx.llbb);
            compile_submatch(else_cx, vec::slice(m, 1u, vec::len(m)),
                             vals, f, exits);
            bcx = next_cx;
          }
          _ {}
        }
        exits += [{bound: m[0].bound, from: bcx.llbb, to: data.body}];
        Br(bcx, data.body);
        ret;
    }

    let col = pick_col(m);
    let val = vals[col];
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
            let ix: uint =
                ty::field_idx(ccx.sess, dummy_sp(), field_name, fields);
            let r = trans::GEP_tup_like(bcx, rec_ty, val, [0, ix as int]);
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
            let r = trans::GEP_tup_like(bcx, tup_ty, val, [0, i as int]);
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
        let unboxed =
            InBoundsGEP(bcx, box,
                                  [C_int(0),
                                   C_int(back::abi::box_rc_field_body)]);
        compile_submatch(bcx, enter_box(m, col, val), [unboxed] + vals_left,
                         f, exits);
        ret;
    }

    // Decide what kind of branch we need
    let opts = get_options(ccx, m, col);
    tag branch_kind { no_branch; single; switch; compare; }
    let kind = no_branch;
    let test_val = val;
    if vec::len(opts) > 0u {
        alt opts[0] {
          var(_, vdef) {
            if vec::len(ty::tag_variants(ccx.tcx, vdef.tg)) == 1u {
                kind = single;
            } else {
                let tagptr =
                    PointerCast(bcx, val,
                        trans_common::T_opaque_tag_ptr(ccx.tn));
                let discrimptr = GEP(bcx, tagptr, [C_int(0), C_int(0)]);
                test_val = Load(bcx, discrimptr);
                kind = switch;
            }
          }
          lit(l) {
            test_val = Load(bcx, val);
            kind = alt l.node { ast::lit_str(_, _) { compare } _ { switch } };
          }
        }
    }
    let else_cx =
        alt kind {
          no_branch. | single. { bcx }
          _ { new_sub_block_ctxt(bcx, ~"match_else") }
        };
    let sw =
        if kind == switch {
            Switch(bcx, test_val, else_cx.llbb, vec::len(opts))
        } else { C_int(0) }; // Placeholder for when not using a switch

     // Compile subtrees for each option
    for opt: opt in opts {
        let opt_cx = new_sub_block_ctxt(bcx, ~"match_case");
        alt kind {
          single. { Br(bcx, opt_cx.llbb); }
          switch. {
            let r = trans_opt(bcx, opt);
            bcx = r.bcx;
            llvm::LLVMAddCase(sw, r.val, opt_cx.llbb);
          }
          compare. {
            let compare_cx = new_scope_block_ctxt(bcx, ~"compare_scope");
            Br(bcx, compare_cx.llbb);
            bcx = compare_cx;
            let r = trans_opt(bcx, opt);
            bcx = r.bcx;
            let t = ty::node_id_to_type(ccx.tcx, pat_id);
            let eq =
                trans::trans_compare(bcx, ast::eq, test_val, t, r.val, t);
            let cleanup_cx = trans::trans_block_cleanups(bcx, compare_cx);
            bcx = new_sub_block_ctxt(bcx, ~"compare_next");
            CondBr(cleanup_cx, eq.val, opt_cx.llbb, bcx.llbb);
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
          lit(_) { }
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
fn make_phi_bindings(bcx: &@block_ctxt, map: &[exit_node],
                     ids: &ast_util::pat_id_map) -> bool {
    let our_block = bcx.llbb as uint;
    let success = true;
    for each item: @{key: ast::ident, val: ast::node_id} in ids.items() {
        let llbbs = [];
        let vals = [];
        for ex: exit_node in map {
            if ex.to as uint == our_block {
                alt assoc(item.key, ex.bound) {
                  some(val) { llbbs += [ex.from]; vals += [val]; }
                  none. { }
                }
            }
        }
        if vec::len(vals) > 0u {
            let phi = Phi(bcx, val_ty(vals[0]), vals, llbbs);
            bcx.fcx.lllocals.insert(item.val, phi);
        } else { success = false; }
    }
    ret success;
}

fn trans_alt(cx: &@block_ctxt, expr: &@ast::expr, arms: &[ast::arm],
             output: &trans::out_method) -> result {
    let bodies = [];
    let match: match = [];
    let er = trans::trans_expr(cx, expr);
    if ty::type_is_bot(bcx_tcx(cx), ty::expr_ty(bcx_tcx(cx), expr)) {

        // No need to generate code for alt,
        // since the disc diverges.
        if !is_terminated(cx) {
            ret rslt(cx, Unreachable(cx));
        } else { ret er; }
    }

    for a: ast::arm in arms {
        let body = new_scope_block_ctxt(cx, ~"case_body");
        let id_map = ast_util::pat_id_map(a.pats[0]);
        bodies += [body];
        for p: @ast::pat in a.pats {
            match += [@{pats: [p],
                        bound: [],
                        data: @{body: body.llbb,
                                guard: a.guard,
                                id_map: id_map}}];
        }
    }

    // Cached fail-on-fallthrough block
    let fail_cx = @mutable none;
    fn mk_fail(cx: &@block_ctxt, sp: &span,
               done: @mutable option::t<BasicBlockRef>) -> BasicBlockRef {
        alt *done { some(bb) { ret bb; } _ { } }
        let fail_cx = new_sub_block_ctxt(cx, ~"case_fallthrough");
        trans::trans_fail(fail_cx, some(sp), ~"non-exhaustive match failure");
        *done = some(fail_cx.llbb);
        ret fail_cx.llbb;
    }

    let exit_map = [];
    let t = trans::node_id_type(cx.fcx.lcx.ccx, expr.id);
    let v = trans::spill_if_immediate(er.bcx, er.val, t);
    compile_submatch(er.bcx, match, [v], bind mk_fail(cx, expr.span, fail_cx),
                     exit_map);

    let i = 0u;
    let arm_results = [];
    for a: ast::arm in arms {
        let body_cx = bodies[i];
        if make_phi_bindings(body_cx, exit_map,
                             ast_util::pat_id_map(a.pats[0])) {
            let block_res = trans::trans_block(body_cx, a.body, output);
            arm_results += [block_res];
        } else { // Unreachable
            arm_results += [rslt(body_cx, C_nil())];
        }
        i += 1u;
    }
    ret rslt(trans::join_branches(cx, arm_results), C_nil());
}

// Not alt-related, but similar to the pattern-munging code above
fn bind_irrefutable_pat(bcx: @block_ctxt, pat: &@ast::pat, val: ValueRef,
                        table: hashmap<ast::node_id, ValueRef>,
                        make_copy: bool) -> @block_ctxt {
    let ccx = bcx.fcx.lcx.ccx;
    alt pat.node {
      ast::pat_bind(_) {
        if make_copy {
            let ty = ty::node_id_to_monotype(ccx.tcx, pat.id);
            let llty = trans::type_of(ccx, pat.span, ty);
            let alloc = trans::alloca(bcx, llty);
            bcx = trans::copy_val(bcx, trans::INIT, alloc,
                                  trans::load_if_immediate(bcx, val, ty), ty);
            table.insert(pat.id, alloc);
            trans_common::add_clean(bcx, alloc, ty);
        } else { table.insert(pat.id, val); }
      }
      ast::pat_tag(_, sub) {
        if vec::len(sub) == 0u { ret bcx; }
        let vdefs = ast_util::variant_def_ids(ccx.tcx.def_map.get(pat.id));
        let args = extract_variant_args(bcx, pat.id, vdefs, val);
        let i = 0;
        for argval: ValueRef in args.vals {
            bcx = bind_irrefutable_pat(bcx, sub[i], argval, table, make_copy);
            i += 1;
        }
      }
      ast::pat_rec(fields, _) {
        let rec_ty = ty::node_id_to_monotype(ccx.tcx, pat.id);
        let rec_fields =
            alt ty::struct(ccx.tcx, rec_ty) { ty::ty_rec(fields) { fields } };
        for f: ast::field_pat in fields {
            let ix: uint =
                ty::field_idx(ccx.sess, pat.span, f.ident, rec_fields);
            let r = trans::GEP_tup_like(bcx, rec_ty, val, [0, ix as int]);
            bcx = bind_irrefutable_pat(r.bcx, f.pat, r.val, table, make_copy);
        }
      }
      ast::pat_tup(elems) {
        let tup_ty = ty::node_id_to_monotype(ccx.tcx, pat.id);
        let i = 0u;
        for elem in elems {
            let r = trans::GEP_tup_like(bcx, tup_ty, val, [0, i as int]);
            bcx = bind_irrefutable_pat(r.bcx, elem, r.val, table, make_copy);
            i += 1u;
        }
      }
      ast::pat_box(inner) {
        let box = Load(bcx, val);
        let unboxed = InBoundsGEP(bcx, box,
                                  [C_int(0),
                                   C_int(back::abi::box_rc_field_body)]);
        bcx = bind_irrefutable_pat(bcx, inner, unboxed, table, true);
      }
      ast::pat_wild. | ast::pat_lit(_) { }
    }
    ret bcx;
}

// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
