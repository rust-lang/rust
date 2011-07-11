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
import trans::result;
import trans::rslt;
import trans::crate_ctxt;
import trans::block_ctxt;
import trans::new_sub_block_ctxt;
import trans::new_scope_block_ctxt;
import trans::load_if_immediate;
import trans::C_int;
import trans::C_uint;
import trans::C_nil;
import trans::val_ty;
import ty::pat_ty;
import syntax::ast;
import syntax::ast::def_id;
import syntax::codemap::span;
import util::common::lit_eq;

// An option identifying a branch (either a literal or a tag variant)
tag opt {
    lit(@ast::lit);
    var(uint /* variant id */, tup(def_id, def_id) /* variant def ids */);
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
fn trans_opt(&@crate_ctxt ccx, &opt o) -> ValueRef {
    alt (o) {
        lit(?l) { ret trans::trans_lit(ccx, *l); }
        var(?id, _) { ret C_int(id as int); }
    }
}

fn variant_opt(&@crate_ctxt ccx, ast::node_id pat_id) -> opt {
    auto vdef = ast::variant_def_ids(ccx.tcx.def_map.get(pat_id));
    auto variants = ty::tag_variants(ccx.tcx, vdef._0);
    auto i = 0u;
    for (ty::variant_info v in variants) {
        if (vdef._1 == v.id) { ret var(i, vdef); }
        i += 1u;
    }
    fail;
}

type bind_map = tup(ast::ident, ValueRef)[];
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
            br.bound += ~[tup(name, val)];
        }
        _ {}
    }
}

fn enter_default(&match m, uint col, ValueRef val) -> match {
    auto result = ~[];
    for (match_branch br in m) {
        if (matches_always(br.pats.(col))) {
            auto pats = ivec::slice(br.pats, 0u, col) +
                ivec::slice(br.pats, col + 1u, ivec::len(br.pats));
            auto new_br = @rec(pats=pats with *br);
            result += ~[new_br];
            bind_for_pat(br.pats.(col), new_br, val);
        }
    }
    ret result;
}

fn enter_opt(&@crate_ctxt ccx, &match m, &opt opt,
             uint col, uint tag_size, ValueRef val) -> match {
    auto result = ~[];
    auto dummy = @rec(id=0, node=ast::pat_wild, span=rec(lo=0u, hi=0u));
    for (match_branch br in m) {
        auto pats = ivec::slice(br.pats, 0u, col);
        auto include = true;
        alt (br.pats.(col).node) {
            ast::pat_tag(?ctor, _) {
                include = opt_eq(variant_opt(ccx, br.pats.(col).id), opt);
            }
            ast::pat_lit(?l) {
                include = opt_eq(lit(l), opt);
            }
            _ {}
        }
        if (include) {
            alt (br.pats.(col).node) {
                ast::pat_tag(_, ?subpats) {
                    assert ivec::len(subpats) == tag_size;
                    pats += subpats;
                }
                _ {
                    pats += ivec::init_elt(dummy, tag_size);
                }
            }
            pats += ivec::slice(br.pats, col + 1u, ivec::len(br.pats));
            auto new_br = @rec(pats=pats with *br);
            result += ~[new_br];
            bind_for_pat(br.pats.(col), new_br, val);
        }
    }
    ret result;
}

fn enter_rec(&@crate_ctxt ccx, &match m, uint col, &ast::ident[] fields,
             ValueRef val) -> match {
    auto result = ~[];
    auto dummy = @rec(id=0, node=ast::pat_wild, span=rec(lo=0u, hi=0u));
    for (match_branch br in m) {
        auto pats = ivec::slice(br.pats, 0u, col);
        alt (br.pats.(col).node) {
            ast::pat_rec(?fpats, _) {
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
            }
            _ {
                pats += ivec::init_elt(dummy, ivec::len(fields));
            }
        }
        pats += ivec::slice(br.pats, col + 1u, ivec::len(br.pats));
        auto new_br = @rec(pats=pats with *br);
        result += ~[new_br];
        bind_for_pat(br.pats.(col), new_br, val);
    }
    ret result;
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
                        &tup(def_id, def_id) vdefs, ValueRef val)
    -> tup(ValueRef[], @block_ctxt) {
    auto ccx = bcx.fcx.lcx.ccx;
    auto ty_param_substs = ty::node_id_to_type_params(ccx.tcx, pat_id);
    auto blobptr = val;
    auto variants = ty::tag_variants(ccx.tcx, vdefs._0);
    auto args = ~[];
    auto size = ivec::len(ty::tag_variant_with_id
                          (ccx.tcx, vdefs._0, vdefs._1).args);
    if (size > 0u && ivec::len(variants) != 1u) {
        auto tagptr = bcx.build.PointerCast
            (val, trans::T_opaque_tag_ptr(ccx.tn));
        blobptr = bcx.build.GEP(tagptr, ~[C_int(0), C_int(1)]);
    }
    auto i = 0u;
    while (i < size) {
        auto r = trans::GEP_tag(bcx, blobptr, vdefs._0, vdefs._1,
                                ty_param_substs, i as int);
        bcx = r.bcx;
        args += ~[r.val];
        i += 1u;
    }
    ret tup(args, bcx);
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

    auto rec_fields = ~[];
    for (match_branch br in m) {
        // Find a real id (we're adding placeholder wildcard patterns, but
        // each column is guaranteed to have at least one real pattern)
        if (pat_id == 0) { pat_id = br.pats.(col).id; }
        // Gather field names
        alt (br.pats.(col).node) {
            ast::pat_rec(?fs, _) {
                for (ast::field_pat f in fs) {
                    if (!ivec::any(bind str::eq(f.ident, _), rec_fields)) {
                        rec_fields += ~[f.ident];
                    }
                }
            }
            _ {}
        }
    }
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
        compile_submatch(bcx, enter_rec(ccx, m, col, rec_fields, val),
                         rec_vals + vals_left, f, exits);
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
                if (ivec::len(ty::tag_variants(ccx.tcx, vdef._0)) == 1u) {
                    kind = single;
                } else {
                    auto tagptr = bcx.build.PointerCast
                        (val, trans::T_opaque_tag_ptr(ccx.tn));
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
            switch { llvm::LLVMAddCase(sw, trans_opt(ccx, opt), opt_cx.llbb);}
            compare {
                auto t = ty::node_id_to_type(ccx.tcx, pat_id);
                auto eq = trans::trans_compare(bcx, ast::eq, t, test_val,
                                               trans_opt(ccx, opt));
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
                 size = ivec::len(args._0);
                 unpacked = args._0;
                 opt_cx = args._1;
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

// FIXME breaks on unreacheable cases
fn make_phi_bindings(&@block_ctxt bcx, &exit_node[] map,
                     &ast::pat_id_map ids) {
    fn assoc(str key, &tup(str, ValueRef)[] list) -> ValueRef {
        for (tup(str, ValueRef) elt in list) {
            if (str::eq(elt._0, key)) { ret elt._1; }
        }
        fail;
    }
    
    auto our_block = bcx.llbb as uint;
    for each (@tup(ast::ident, ast::node_id) item in ids.items()) {
        auto llbbs = ~[];
        auto vals = ~[];
        for (exit_node ex in map) {
            if (ex.to as uint == our_block) {
                llbbs += ~[ex.from];
                vals += ~[assoc(item._0, ex.bound)];
            }
        }
        auto phi = bcx.build.Phi(val_ty(vals.(0)), vals, llbbs);
        bcx.fcx.lllocals.insert(item._1, phi);
    }
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
        make_phi_bindings(body_cx, exit_map, ast::pat_id_map(a.pats.(0)));
        auto block_res = trans::trans_block(body_cx, a.block, output);
        arm_results += ~[block_res];
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
