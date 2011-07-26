import std::str;
import std::ivec;
import std::option;
import option::some;
import option::none;
import std::map::hashmap;

import lib::llvm::llvm;
import lib::llvm::llvm::ValueRef;
import lib::llvm::llvm::TypeRef;
import lib::llvm::llvm::BasicBlockRef;
import trans::new_sub_block_ctxt;
import trans::new_scope_block_ctxt;
import trans::load_if_immediate;
import ty::pat_ty;
import syntax::ast;
import syntax::ast::def_id;
import syntax::codemap::span;
import util::common::lit_eq;

import trans_common::*;

// An option identifying a branch (either a literal or a tag variant)
tag opt {
    lit(@ast::lit);
    var(uint /* variant id */, rec(def_id tg, def_id var) /* variant dids */);
}
fn opt_eq(&opt a, &opt b) -> bool {
    alt (a) {
        lit(?la) {
            ret alt (b) { lit(?lb) { lit_eq(la, lb) } var(_, _) { false } };
        }
        var(?ida, _) {
            ret alt (b) { lit(_) { false } var(?idb, _) { ida == idb } };
        }
    }
}
fn trans_opt(&@block_ctxt bcx, &opt o) -> result {
    alt (o) {
        lit(?l) { ret trans::trans_lit(bcx, *l); }
        var(?id, _) { ret rslt(bcx, C_int(id as int)); }
    }
}

fn variant_opt(&@crate_ctxt ccx, ast::node_id pat_id) -> opt {
    auto vdef = ast::variant_def_ids(ccx.tcx.def_map.get(pat_id));
    auto variants = ty::tag_variants(ccx.tcx, vdef.tg);
    auto i = 0u;
    for (ty::variant_info v in variants) {
        if (vdef.var == v.id) { ret var(i, vdef); }
        i += 1u;
    }
    fail;
}

type bind_map = rec(ast::ident ident, ValueRef val)[];
type match_branch = @rec((@ast::pat)[] pats,
                         BasicBlockRef body,
                         mutable bind_map bound);
type match = match_branch[];

fn matches_always(&@ast::pat p) -> bool {
    ret alt p.node {
        ast::pat_wild { true }
        ast::pat_bind(_) { true }
        ast::pat_rec(_, _) { true }
        _ { false }
    };
}


fn bind_for_pat(&@ast::pat p, &match_branch br, ValueRef val) {
    alt p.node {
        ast::pat_bind(?name) {
            br.bound += ~[rec(ident=name, val=val)];
        }
        _ {}
    }
}

type enter_pat = fn(&@ast::pat) -> option::t[(@ast::pat)[]];

fn enter_match(&match m, uint col, ValueRef val, &enter_pat e) -> match {
    auto result = ~[];
    for (match_branch br in m) {
        alt (e(br.pats.(col))) {
            some(?sub) {
                auto pats = ivec::slice(br.pats, 0u, col) +
                    sub + ivec::slice(br.pats, col + 1u, ivec::len(br.pats));
                auto new_br = @rec(pats=pats with *br);
                result += ~[new_br];
                bind_for_pat(br.pats.(col), new_br, val);
            }
            none {}
        }
    }
    ret result;
}

fn enter_default(&match m, uint col, ValueRef val) -> match {
    fn e(&@ast::pat p) -> option::t[(@ast::pat)[]] {
        ret if (matches_always(p)) { some(~[]) }
            else { none };
    }
    ret enter_match(m, col, val, e);
}

fn enter_opt(&@crate_ctxt ccx, &match m, &opt opt,
             uint col, uint tag_size, ValueRef val) -> match {
    auto dummy = @rec(id=0, node=ast::pat_wild, span=rec(lo=0u, hi=0u));
    fn e(&@crate_ctxt ccx, &@ast::pat dummy, &opt opt, uint size,
         &@ast::pat p) -> option::t[(@ast::pat)[]] {
        alt (p.node) {
            ast::pat_tag(?ctor, ?subpats) {
                ret if (opt_eq(variant_opt(ccx, p.id), opt)) { some(subpats) }
                    else { none };
            }
            ast::pat_lit(?l) {
                ret if (opt_eq(lit(l), opt)) { some(~[]) }
                    else { none };
            }
            _ { ret some(ivec::init_elt(dummy, size)); }
        }
    }
    ret enter_match(m, col, val, bind e(ccx, dummy, opt, tag_size, _));
}

fn enter_rec(&match m, uint col, &ast::ident[] fields,
             ValueRef val) -> match {
    auto dummy = @rec(id=0, node=ast::pat_wild, span=rec(lo=0u, hi=0u));
    fn e(&@ast::pat dummy, &ast::ident[] fields, &@ast::pat p)
        -> option::t[(@ast::pat)[]] {
        alt (p.node) {
            ast::pat_rec(?fpats, _) {
                auto pats = ~[];
                for (ast::ident fname in fields) {
                    auto pat = dummy;
                    for (ast::field_pat fpat in fpats) {
                        if (str::eq(fpat.ident, fname)) {
                            pat = fpat.pat;
                            break;
                        }
                    }
                    pats += ~[pat];
                }
                ret some(pats);
            }
            _ { ret some(ivec::init_elt(dummy, ivec::len(fields))); }
        }
    }
    ret enter_match(m, col, val, bind e(dummy, fields, _));
}

fn enter_box(&match m, uint col, ValueRef val) -> match {
    auto dummy = @rec(id=0, node=ast::pat_wild, span=rec(lo=0u, hi=0u));
    fn e(&@ast::pat dummy, &@ast::pat p) -> option::t[(@ast::pat)[]] {
        alt (p.node) {
            ast::pat_box(?sub) { ret some(~[sub]); }
            _ { ret some(~[dummy]); }
        }
    }
    ret enter_match(m, col, val, bind e(dummy, _));
}

fn get_options(&@crate_ctxt ccx, &match m, uint col) -> opt[] {
    fn add_to_set(&mutable opt[] set, &opt val) {
        for (opt l in set) {
            if (opt_eq(l, val)) { ret; }
        }
        set += ~[val];
    }

    auto found = ~[];
    for (match_branch br in m) {
        alt (br.pats.(col).node) {
            ast::pat_lit(?l) { add_to_set(found, lit(l)); }
            ast::pat_tag(_, _) {
                add_to_set(found, variant_opt(ccx, br.pats.(col).id));
            }
            _ {}
        }
    }
    ret found;
}

fn extract_variant_args(@block_ctxt bcx, ast::node_id pat_id,
                        &rec(def_id tg, def_id var) vdefs, ValueRef val)
    -> rec(ValueRef[] vals, @block_ctxt bcx) {
    auto ccx = bcx.fcx.lcx.ccx;
    auto ty_param_substs = ty::node_id_to_type_params(ccx.tcx, pat_id);
    auto blobptr = val;
    auto variants = ty::tag_variants(ccx.tcx, vdefs.tg);
    auto args = ~[];
    auto size = ivec::len(ty::tag_variant_with_id
                          (ccx.tcx, vdefs.tg, vdefs.var).args);
    if (size > 0u && ivec::len(variants) != 1u) {
        auto tagptr = bcx.build.PointerCast
            (val, trans_common::T_opaque_tag_ptr(ccx.tn));
        blobptr = bcx.build.GEP(tagptr, ~[C_int(0), C_int(1)]);
    }
    auto i = 0u;
    while (i < size) {
        auto r = trans::GEP_tag(bcx, blobptr, vdefs.tg, vdefs.var,
                                ty_param_substs, i as int);
        bcx = r.bcx;
        args += ~[r.val];
        i += 1u;
    }
    ret rec(vals=args, bcx=bcx);
}

fn collect_record_fields(&match m, uint col) -> ast::ident[] {
    auto fields = ~[];
    for (match_branch br in m) {
        alt (br.pats.(col).node) {
            ast::pat_rec(?fs, _) {
                for (ast::field_pat f in fs) {
                    if (!ivec::any(bind str::eq(f.ident, _), fields)) {
                        fields += ~[f.ident];
                    }
                }
            }
            _ {}
        }
    }
    ret fields;
}

fn any_box_pat(&match m, uint col) -> bool {
    for (match_branch br in m) {
        alt (br.pats.(col).node) {
            ast::pat_box(_) { ret true; }
            _ {}
        }
    }
    ret false;
}

type exit_node = rec(bind_map bound,
                     BasicBlockRef from,
                     BasicBlockRef to);
type mk_fail = fn() -> BasicBlockRef;

fn compile_submatch(@block_ctxt bcx, &match m, ValueRef[] vals, &mk_fail f,
                    &mutable exit_node[] exits) {
    if (ivec::len(m) == 0u) {
        bcx.build.Br(f());
        ret;
    }
    if (ivec::len(m.(0).pats) == 0u) {
        exits += ~[rec(bound=m.(0).bound,
                       from=bcx.llbb,
                       to=m.(0).body)];
        bcx.build.Br(m.(0).body);
        ret;
    }

    // FIXME maybe be clever about picking a column.
    auto col = 0u;
    auto val = vals.(col);
    auto vals_left = ivec::slice(vals, 1u, ivec::len(vals));
    auto ccx = bcx.fcx.lcx.ccx;
    auto pat_id = 0;
    for (match_branch br in m) {
        // Find a real id (we're adding placeholder wildcard patterns, but
        // each column is guaranteed to have at least one real pattern)
        if (pat_id == 0) { pat_id = br.pats.(col).id; }
    }

    auto rec_fields = collect_record_fields(m, col);
    // Separate path for extracting and binding record fields
    if (ivec::len(rec_fields) > 0u) {
        auto rec_ty = ty::node_id_to_monotype(ccx.tcx, pat_id);
        auto fields = alt (ty::struct(ccx.tcx, rec_ty)) {
            ty::ty_rec(?fields) { fields }
        };
        auto rec_vals = ~[];
        for (ast::ident field_name in rec_fields) {
            let uint ix = ty::field_idx(ccx.sess, rec(lo=0u, hi=0u),
                                        field_name, fields);
            auto r = trans::GEP_tup_like(bcx, rec_ty, val, ~[0, ix as int]);
            rec_vals += ~[r.val];
            bcx = r.bcx;
        }
        compile_submatch(bcx, enter_rec(m, col, rec_fields, val),
                         rec_vals + vals_left, f, exits);
        ret;
    }

    // Unbox in case of a box field
    if (any_box_pat(m, col)) {
        auto box = bcx.build.Load(val);
        auto unboxed = bcx.build.InBoundsGEP
            (box, ~[C_int(0), C_int(back::abi::box_rc_field_body)]);
        compile_submatch(bcx, enter_box(m, col, val),
                         ~[unboxed] + vals_left, f, exits);
        ret;
    }

    // Decide what kind of branch we need
    auto opts = get_options(ccx, m, col);
    tag branch_kind { no_branch; single; switch; compare; }
    auto kind = no_branch;
    auto test_val = val;
    if (ivec::len(opts) > 0u) {
        alt (opts.(0)) {
            var(_, ?vdef) {
                if (ivec::len(ty::tag_variants(ccx.tcx, vdef.tg)) == 1u) {
                    kind = single;
                } else {
                    auto tagptr = bcx.build.PointerCast
                        (val, trans_common::T_opaque_tag_ptr(ccx.tn));
                    auto discrimptr = bcx.build.GEP
                        (tagptr, ~[C_int(0), C_int(0)]);
                    test_val = bcx.build.Load(discrimptr);
                    kind = switch;
                }
            }
            lit(?l) {
                test_val = bcx.build.Load(val);
                kind = alt (l.node) {
                    ast::lit_str(_, _) { compare }
                    _ { switch }
                };
            }
        }
    }
    auto else_cx = alt (kind) {
        no_branch | single { bcx }
        _ { new_sub_block_ctxt(bcx, "match_else") }
    };
    auto sw = if (kind == switch) {
        bcx.build.Switch(test_val, else_cx.llbb, ivec::len(opts))
    } else { C_int(0) }; // Placeholder for when not using a switch

    // Compile subtrees for each option
    for (opt opt in opts) {
        auto opt_cx = new_sub_block_ctxt(bcx, "match_case");
        alt (kind) {
            single { bcx.build.Br(opt_cx.llbb); }
            switch {
                auto r = trans_opt(bcx, opt);
                bcx = r.bcx;
                llvm::LLVMAddCase(sw, r.val, opt_cx.llbb);
            }
            compare {
                auto r = trans_opt(bcx, opt);
                bcx = r.bcx;
                auto t = ty::node_id_to_type(ccx.tcx, pat_id);
                auto eq = trans::trans_compare(bcx, ast::eq, t, test_val,
                                               r.val);
                bcx = new_sub_block_ctxt(bcx, "next");
                eq.bcx.build.CondBr(eq.val, opt_cx.llbb, bcx.llbb);
            }
            _ {}
        }
        auto size = 0u;
        auto unpacked = ~[];
        alt opt {
             var(_, ?vdef) {
                 auto args = extract_variant_args(opt_cx, pat_id, vdef, val);
                 size = ivec::len(args.vals);
                 unpacked = args.vals;
                 opt_cx = args.bcx;
             }
             lit(_) { }
        }
        compile_submatch(opt_cx, enter_opt(ccx, m, opt, col, size, val),
                         unpacked + vals_left, f, exits);
    }

    // Compile the fall-through case
    if (kind == compare) { bcx.build.Br(else_cx.llbb); }
    if (kind != single) {
        compile_submatch(else_cx, enter_default(m, col, val), vals_left,
                         f, exits);
    }
}

// Returns false for unreachable blocks
fn make_phi_bindings(&@block_ctxt bcx, &exit_node[] map,
                     &ast::pat_id_map ids) -> bool {
    fn assoc(str key, &bind_map list)
        -> option::t[ValueRef] {
        for (rec(ast::ident ident, ValueRef val) elt in list) {
            if (str::eq(elt.ident, key)) { ret some(elt.val); }
        }
        ret none;
    }

    auto our_block = bcx.llbb as uint;
    auto success = true;
    for each (@rec(ast::ident key, ast::node_id val) item
              in ids.items()) {
        auto llbbs = ~[];
        auto vals = ~[];
        for (exit_node ex in map) {
            if (ex.to as uint == our_block) {
                alt (assoc(item.key, ex.bound)) {
                    some(?val) {
                        llbbs += ~[ex.from];
                        vals += ~[val];
                    }
                    none {}
                }
            }
        }
        if (ivec::len(vals) > 0u) {
            auto phi = bcx.build.Phi(val_ty(vals.(0)), vals, llbbs);
            bcx.fcx.lllocals.insert(item.val, phi);
        } else { success = false; }
    }
    ret success;
}

fn trans_alt(&@block_ctxt cx, &@ast::expr expr, &ast::arm[] arms,
             ast::node_id id, &trans::out_method output) -> result {
    auto bodies = ~[];
    let match match = ~[];
    for (ast::arm a in arms) {
        auto body = new_scope_block_ctxt(cx, "case_body");
        bodies += ~[body];
        for (@ast::pat p in a.pats) {
            match += ~[@rec(pats=~[p], body=body.llbb, mutable bound=~[])];
        }
    }

    // Cached fail-on-fallthrough block
    auto fail_cx = @mutable none;
    fn mk_fail(&@block_ctxt cx, &span sp,
               @mutable option::t[BasicBlockRef] done) -> BasicBlockRef {
        alt (*done) {
            some(?bb) { ret bb; }
            _ {}
        }
        auto fail_cx = new_sub_block_ctxt(cx, "case_fallthrough");
        trans::trans_fail(fail_cx, some(sp), "non-exhaustive match failure");
        *done = some(fail_cx.llbb);
        ret fail_cx.llbb;
    }

    auto exit_map = ~[];
    auto er = trans::trans_expr(cx, expr);
    auto t = trans::node_id_type(cx.fcx.lcx.ccx, expr.id);
    auto v = trans::spill_if_immediate(er.bcx, er.val, t);
    compile_submatch(er.bcx, match, ~[v],
                     bind mk_fail(cx, expr.span, fail_cx), exit_map);

    auto i = 0u;
    auto arm_results = ~[];
    for (ast::arm a in arms) {
        auto body_cx = bodies.(i);
        if (make_phi_bindings(body_cx, exit_map,
                              ast::pat_id_map(a.pats.(0)))) {
            auto block_res = trans::trans_block(body_cx, a.block, output);
            arm_results += ~[block_res];
        } else { // Unreachable
            arm_results += ~[rslt(body_cx, C_nil())];
        }
        i += 1u;
    }
    ret rslt(trans::join_branches(cx, arm_results), C_nil());
}

// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
