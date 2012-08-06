import driver::session::session;
import lib::llvm::llvm;
import lib::llvm::{ValueRef, BasicBlockRef};
import pat_util::*;
import build::*;
import base::*;
import syntax::ast;
import syntax::ast_util;
import syntax::ast_util::{dummy_sp, path_to_ident};
import syntax::ast::def_id;
import syntax::codemap::span;
import syntax::print::pprust::pat_to_str;
import middle::resolve3::DefMap;
import back::abi;
import std::map::hashmap;
import dvec::{dvec, extensions};

import common::*;

// An option identifying a branch (either a literal, a enum variant or a
// range)
enum opt {
    lit(@ast::expr),
    var(/* disr val */int, /* variant dids */{enm: def_id, var: def_id}),
    range(@ast::expr, @ast::expr)
}
fn opt_eq(tcx: ty::ctxt, a: opt, b: opt) -> bool {
    match (a, b) {
      (lit(a), lit(b)) => const_eval::compare_lit_exprs(tcx, a, b) == 0,
      (range(a1, a2), range(b1, b2)) => {
        const_eval::compare_lit_exprs(tcx, a1, b1) == 0 &&
        const_eval::compare_lit_exprs(tcx, a2, b2) == 0
      }
      (var(a, _), var(b, _)) => a == b,
      _ => false
    }
}

enum opt_result {
    single_result(result),
    range_result(result, result),
}
fn trans_opt(bcx: block, o: opt) -> opt_result {
    let _icx = bcx.insn_ctxt(~"alt::trans_opt");
    let ccx = bcx.ccx();
    let mut bcx = bcx;
    match o {
      lit(l) => {
        match l.node {
          ast::expr_vstore(@{node: ast::expr_lit(
              @{node: ast::lit_str(s), _}), _},
                           ast::vstore_uniq) => {
            let strty = ty::mk_estr(bcx.tcx(), ty::vstore_uniq);
            let cell = empty_dest_cell();
            bcx = tvec::trans_estr(bcx, s, some(ast::vstore_uniq),
                                   by_val(cell));
            add_clean_temp(bcx, *cell, strty);
            return single_result(rslt(bcx, *cell));
          }
          _ => {
            return single_result(
                rslt(bcx, consts::const_expr(ccx, l)));
          }
        }
      }
      var(disr_val, _) => {
        return single_result(rslt(bcx, C_int(ccx, disr_val)));
      }
      range(l1, l2) => {
        return range_result(rslt(bcx, consts::const_expr(ccx, l1)),
                         rslt(bcx, consts::const_expr(ccx, l2)));
      }
    }
}

fn variant_opt(tcx: ty::ctxt, pat_id: ast::node_id) -> opt {
    let vdef = ast_util::variant_def_ids(tcx.def_map.get(pat_id));
    let variants = ty::enum_variants(tcx, vdef.enm);
    for vec::each(*variants) |v| {
        if vdef.var == v.id { return var(v.disr_val, vdef); }
    }
    core::unreachable();
}

struct binding {
    val: ValueRef;
    mode: ast::binding_mode;
    ty: ty::t;
}

type bind_map = ~[{
    ident: ast::ident,
    binding: binding
}];

fn assoc(key: ast::ident, list: bind_map) -> option<binding> {
    for vec::each(list) |elt| {
        if str::eq(elt.ident, key) {
            return some(elt.binding);
        }
    }
    return none;
}

type match_branch =
    @{pats: ~[@ast::pat],
      bound: bind_map,
      data: @{bodycx: block,
              guard: option<@ast::expr>,
              id_map: pat_id_map}};
type match_ = ~[match_branch];

fn has_nested_bindings(m: match_, col: uint) -> bool {
    for vec::each(m) |br| {
        match br.pats[col].node {
          ast::pat_ident(_, _, some(_)) => return true,
          _ => ()
        }
    }
    return false;
}

fn expand_nested_bindings(bcx: block, m: match_, col: uint, val: ValueRef)
                       -> match_ {

    let mut result = ~[];
    for vec::each(m) |br| {
      match br.pats[col].node {
          ast::pat_ident(mode, name, some(inner)) => {
            let pats = vec::append(
                vec::slice(br.pats, 0u, col),
                vec::append(~[inner],
                            vec::view(br.pats, col + 1u, br.pats.len())));
            vec::push(result,
                      @{pats: pats,
                        bound: vec::append(
                            br.bound, ~[{ident: path_to_ident(name),
                                         binding: binding {
                                            val: val,
                                            mode: mode,
                                            ty: node_id_type(bcx,
                                                             br.pats[col].id)
                                         }}])
                                with *br});
          }
          _ => vec::push(result, br)
        }
    }
    result
}

type enter_pat = fn(@ast::pat) -> option<~[@ast::pat]>;

fn enter_match(bcx: block, dm: DefMap, m: match_, col: uint, val: ValueRef,
               e: enter_pat) -> match_ {
    let mut result = ~[];
    for vec::each(m) |br| {
        match e(br.pats[col]) {
          some(sub) => {
            let pats = vec::append(
                vec::append(sub, vec::view(br.pats, 0u, col)),
                vec::view(br.pats, col + 1u, br.pats.len()));
            let self = br.pats[col];
            let bound = match self.node {
              ast::pat_ident(mode, name, none)
                  if !pat_is_variant(dm, self) => {
                vec::append(br.bound,
                            ~[{ident: path_to_ident(name),
                               binding: binding {
                                   val: val,
                                   mode: mode,
                                   ty: node_id_type(bcx, br.pats[col].id)
                               }}])
              }
              _ => br.bound
            };
            vec::push(result, @{pats: pats, bound: bound with *br});
          }
          none => ()
        }
    }
    return result;
}

fn enter_default(bcx: block, dm: DefMap, m: match_, col: uint, val: ValueRef)
              -> match_ {

    do enter_match(bcx, dm, m, col, val) |p| {
        match p.node {
          ast::pat_wild | ast::pat_rec(_, _) | ast::pat_tup(_) => some(~[]),
          ast::pat_ident(_, _, none) if !pat_is_variant(dm, p) => some(~[]),
          _ => none
        }
    }
}

fn enter_opt(bcx: block, m: match_, opt: opt, col: uint,
             variant_size: uint, val: ValueRef) -> match_ {
    let tcx = bcx.tcx();
    let dummy = @{id: 0, node: ast::pat_wild, span: dummy_sp()};
    do enter_match(bcx, tcx.def_map, m, col, val) |p| {
        match p.node {
          ast::pat_enum(_, subpats) => {
            if opt_eq(tcx, variant_opt(tcx, p.id), opt) {
              some(option::get_default(subpats,
                     vec::from_elem(variant_size, dummy))) }
            else { none }
          }
          ast::pat_ident(_, _, none) if pat_is_variant(tcx.def_map, p) => {
            if opt_eq(tcx, variant_opt(tcx, p.id), opt) { some(~[]) }
            else { none }
          }
          ast::pat_lit(l) => {
            if opt_eq(tcx, lit(l), opt) { some(~[]) } else { none }
          }
          ast::pat_range(l1, l2) => {
            if opt_eq(tcx, range(l1, l2), opt) { some(~[]) } else { none }
          }
          _ => some(vec::from_elem(variant_size, dummy))
        }
    }
}

fn enter_rec(bcx: block, dm: DefMap, m: match_, col: uint,
             fields: ~[ast::ident], val: ValueRef) -> match_ {
    let dummy = @{id: 0, node: ast::pat_wild, span: dummy_sp()};
    do enter_match(bcx, dm, m, col, val) |p| {
        match p.node {
          ast::pat_rec(fpats, _) => {
            let mut pats = ~[];
            for vec::each(fields) |fname| {
                let mut pat = dummy;
                for vec::each(fpats) |fpat| {
                    if str::eq(fpat.ident, fname) { pat = fpat.pat; break; }
                }
                vec::push(pats, pat);
            }
            some(pats)
          }
          _ => some(vec::from_elem(fields.len(), dummy))
        }
    }
}

fn enter_tup(bcx: block, dm: DefMap, m: match_, col: uint, val: ValueRef,
             n_elts: uint) -> match_ {
    let dummy = @{id: 0, node: ast::pat_wild, span: dummy_sp()};
    do enter_match(bcx, dm, m, col, val) |p| {
        match p.node {
          ast::pat_tup(elts) => some(elts),
          _ => some(vec::from_elem(n_elts, dummy))
        }
    }
}

fn enter_box(bcx: block, dm: DefMap, m: match_, col: uint, val: ValueRef)
          -> match_ {
    let dummy = @{id: 0, node: ast::pat_wild, span: dummy_sp()};
    do enter_match(bcx, dm, m, col, val) |p| {
        match p.node {
          ast::pat_box(sub) => some(~[sub]),
          _ => some(~[dummy])
        }
    }
}

fn enter_uniq(bcx: block, dm: DefMap, m: match_, col: uint, val: ValueRef)
           -> match_ {
    let dummy = @{id: 0, node: ast::pat_wild, span: dummy_sp()};
    do enter_match(bcx, dm, m, col, val) |p| {
        match p.node {
          ast::pat_uniq(sub) => some(~[sub]),
          _ => some(~[dummy])
        }
    }
}

fn get_options(ccx: @crate_ctxt, m: match_, col: uint) -> ~[opt] {
    fn add_to_set(tcx: ty::ctxt, &&set: dvec<opt>, val: opt) {
        if set.any(|l| opt_eq(tcx, l, val)) {return;}
        set.push(val);
    }

    let found = dvec();
    for vec::each(m) |br| {
        let cur = br.pats[col];
        if pat_is_variant(ccx.tcx.def_map, cur) {
            add_to_set(ccx.tcx, found, variant_opt(ccx.tcx, br.pats[col].id));
        } else {
            match cur.node {
              ast::pat_lit(l) => add_to_set(ccx.tcx, found, lit(l)),
              ast::pat_range(l1, l2) => {
                add_to_set(ccx.tcx, found, range(l1, l2));
              }
              _ => ()
            }
        }
    }
    return vec::from_mut(dvec::unwrap(found));
}

fn extract_variant_args(bcx: block, pat_id: ast::node_id,
                        vdefs: {enm: def_id, var: def_id}, val: ValueRef) ->
   {vals: ~[ValueRef], bcx: block} {
    let _icx = bcx.insn_ctxt(~"alt::extract_variant_args");
    let ccx = bcx.fcx.ccx;
    let enum_ty_substs = match check ty::get(node_id_type(bcx, pat_id))
        .struct {

      ty::ty_enum(id, substs) => { assert id == vdefs.enm; substs.tps }
    };
    let mut blobptr = val;
    let variants = ty::enum_variants(ccx.tcx, vdefs.enm);
    let size = ty::enum_variant_with_id(ccx.tcx, vdefs.enm,
                                        vdefs.var).args.len();
    if size > 0u && (*variants).len() != 1u {
        let enumptr =
            PointerCast(bcx, val, T_opaque_enum_ptr(ccx));
        blobptr = GEPi(bcx, enumptr, ~[0u, 1u]);
    }
    let vdefs_tg = vdefs.enm;
    let vdefs_var = vdefs.var;
    let args = do vec::from_fn(size) |i| {
        GEP_enum(bcx, blobptr, vdefs_tg, vdefs_var,
                 enum_ty_substs, i)
    };
    return {vals: args, bcx: bcx};
}

fn collect_record_fields(m: match_, col: uint) -> ~[ast::ident] {
    let mut fields: ~[ast::ident] = ~[];
    for vec::each(m) |br| {
        match br.pats[col].node {
          ast::pat_rec(fs, _) => {
            for vec::each(fs) |f| {
                if !vec::any(fields, |x| str::eq(f.ident, x)) {
                    vec::push(fields, f.ident);
                }
            }
          }
          _ => ()
        }
    }
    return fields;
}

fn root_pats_as_necessary(bcx: block, m: match_, col: uint, val: ValueRef) {
    for vec::each(m) |br| {
        let pat_id = br.pats[col].id;

        match bcx.ccx().maps.root_map.find({id:pat_id, derefs:0u}) {
          none => (),
          some(scope_id) => {
            // Note: the scope_id will always be the id of the alt.  See the
            // extended comment in rustc::middle::borrowck::preserve() for
            // details (look for the case covering cat_discr).

            let ty = node_id_type(bcx, pat_id);
            let val = load_if_immediate(bcx, val, ty);
            root_value(bcx, val, ty, scope_id);
            return; // if we kept going, we'd only be rooting same value again
          }
        }
    }
}

fn any_box_pat(m: match_, col: uint) -> bool {
    for vec::each(m) |br| {
        match br.pats[col].node {
          ast::pat_box(_) => return true,
          _ => ()
        }
    }
    return false;
}

fn any_uniq_pat(m: match_, col: uint) -> bool {
    for vec::each(m) |br| {
        match br.pats[col].node {
          ast::pat_uniq(_) => return true,
          _ => ()
        }
    }
    return false;
}

fn any_tup_pat(m: match_, col: uint) -> bool {
    for vec::each(m) |br| {
        match br.pats[col].node {
          ast::pat_tup(_) => return true,
          _ => ()
        }
    }
    return false;
}

type exit_node = {bound: bind_map, from: BasicBlockRef, to: BasicBlockRef};
type mk_fail = fn@() -> BasicBlockRef;

fn pick_col(m: match_) -> uint {
    fn score(p: @ast::pat) -> uint {
        match p.node {
          ast::pat_lit(_) | ast::pat_enum(_, _) | ast::pat_range(_, _) => 1u,
          ast::pat_ident(_, _, some(p)) => score(p),
          _ => 0u
        }
    }
    let scores = vec::to_mut(vec::from_elem(m[0].pats.len(), 0u));
    for vec::each(m) |br| {
        let mut i = 0u;
        for vec::each(br.pats) |p| { scores[i] += score(p); i += 1u; }
    }
    let mut max_score = 0u;
    let mut best_col = 0u;
    let mut i = 0u;
    for vec::each(scores) |score| {
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

fn compile_submatch(bcx: block, m: match_, vals: ~[ValueRef],
                    chk: option<mk_fail>, &exits: ~[exit_node]) {
    /*
      For an empty match, a fall-through case must exist
     */
    assert(m.len() > 0u || is_some(chk));
    let _icx = bcx.insn_ctxt(~"alt::compile_submatch");
    let mut bcx = bcx;
    let tcx = bcx.tcx(), dm = tcx.def_map;
    if m.len() == 0u { Br(bcx, option::get(chk)()); return; }
    if m[0].pats.len() == 0u {
        let data = m[0].data;
        match data.guard {
          some(e) => {
            // Temporarily set bindings. They'll be rewritten to PHI nodes
            // for the actual arm block.
            //
            // Also, in the case of by-value, do the copy now.

            for data.id_map.each |key, val| {
                let binding = assoc(key, m[0].bound).get();
                let (llval, mode) = (binding.val, binding.mode);
                let ty = binding.ty;

                if mode == ast::bind_by_value {
                    let llty = type_of::type_of(bcx.fcx.ccx, ty);
                    let alloc = alloca(bcx, llty);
                    bcx = copy_val(bcx, INIT, alloc,
                                   load_if_immediate(bcx, llval, ty), ty);
                    bcx.fcx.lllocals.insert(val, local_mem(alloc));
                    add_clean(bcx, alloc, ty);
                } else {
                    bcx.fcx.lllocals.insert(val, local_mem(llval));
                }
            };
            let {bcx: guard_cx, val} = {
                do with_scope_result(bcx, e.info(), ~"guard") |bcx| {
                    trans_temp_expr(bcx, e)
                }
            };
            bcx = do with_cond(guard_cx, Not(guard_cx, val)) |bcx| {
                compile_submatch(bcx, vec::tail(m), vals, chk, exits);
                bcx
            };
          }
          _ => ()
        }
        if !bcx.unreachable {
            vec::push(exits, {bound: m[0].bound, from: bcx.llbb,
                       to: data.bodycx.llbb});
        }
        Br(bcx, data.bodycx.llbb);
        return;
    }

    let col = pick_col(m);
    let val = vals[col];
    let m = if has_nested_bindings(m, col) {
                expand_nested_bindings(bcx, m, col, val)
            } else { m };

    let vals_left = vec::append(vec::slice(vals, 0u, col),
                                vec::view(vals, col + 1u, vals.len()));
    let ccx = bcx.fcx.ccx;
    let mut pat_id = 0;
    for vec::each(m) |br| {
        // Find a real id (we're adding placeholder wildcard patterns, but
        // each column is guaranteed to have at least one real pattern)
        if pat_id == 0 { pat_id = br.pats[col].id; }
    }

    root_pats_as_necessary(bcx, m, col, val);

    let rec_fields = collect_record_fields(m, col);
    // Separate path for extracting and binding record fields
    if rec_fields.len() > 0u {
        let fields = ty::get_fields(node_id_type(bcx, pat_id));
        let mut rec_vals = ~[];
        for vec::each(rec_fields) |field_name| {
            let ix = option::get(ty::field_idx(field_name, fields));
            vec::push(rec_vals, GEPi(bcx, val, ~[0u, ix]));
        }
        compile_submatch(bcx, enter_rec(bcx, dm, m, col, rec_fields, val),
                         vec::append(rec_vals, vals_left), chk, exits);
        return;
    }

    if any_tup_pat(m, col) {
        let tup_ty = node_id_type(bcx, pat_id);
        let n_tup_elts = match ty::get(tup_ty).struct {
          ty::ty_tup(elts) => elts.len(),
          _ => ccx.sess.bug(~"non-tuple type in tuple pattern")
        };
        let mut tup_vals = ~[], i = 0u;
        while i < n_tup_elts {
            vec::push(tup_vals, GEPi(bcx, val, ~[0u, i]));
            i += 1u;
        }
        compile_submatch(bcx, enter_tup(bcx, dm, m, col, val, n_tup_elts),
                         vec::append(tup_vals, vals_left), chk, exits);
        return;
    }

    // Unbox in case of a box field
    if any_box_pat(m, col) {
        let llbox = Load(bcx, val);
        let box_no_addrspace = non_gc_box_cast(bcx, llbox);
        let unboxed =
            GEPi(bcx, box_no_addrspace, ~[0u, abi::box_field_body]);
        compile_submatch(bcx, enter_box(bcx, dm, m, col, val),
                         vec::append(~[unboxed], vals_left), chk, exits);
        return;
    }

    if any_uniq_pat(m, col) {
        let llbox = Load(bcx, val);
        let box_no_addrspace = non_gc_box_cast(bcx, llbox);
        let unboxed =
            GEPi(bcx, box_no_addrspace, ~[0u, abi::box_field_body]);
        compile_submatch(bcx, enter_uniq(bcx, dm, m, col, val),
                         vec::append(~[unboxed], vals_left), chk, exits);
        return;
    }

    // Decide what kind of branch we need
    let opts = get_options(ccx, m, col);
    enum branch_kind { no_branch, single, switch, compare, }
    let mut kind = no_branch;
    let mut test_val = val;
    if opts.len() > 0u {
        match opts[0] {
          var(_, vdef) => {
            if (*ty::enum_variants(tcx, vdef.enm)).len() == 1u {
                kind = single;
            } else {
                let enumptr =
                    PointerCast(bcx, val, T_opaque_enum_ptr(ccx));
                let discrimptr = GEPi(bcx, enumptr, ~[0u, 0u]);
                test_val = Load(bcx, discrimptr);
                kind = switch;
            }
          }
          lit(l) => {
            test_val = Load(bcx, val);
            let pty = node_id_type(bcx, pat_id);
            kind = if ty::type_is_integral(pty) { switch }
                   else { compare };
          }
          range(_, _) => {
            test_val = Load(bcx, val);
            kind = compare;
          }
        }
    }
    for vec::each(opts) |o| {
        match o {
          range(_, _) => { kind = compare; break }
          _ => ()
        }
    }
    let else_cx = match kind {
      no_branch | single => bcx,
      _ => sub_block(bcx, ~"match_else")
    };
    let sw = if kind == switch {
        Switch(bcx, test_val, else_cx.llbb, opts.len())
    } else { C_int(ccx, 0) }; // Placeholder for when not using a switch

    let defaults = enter_default(bcx, dm, m, col, val);
    let exhaustive = option::is_none(chk) && defaults.len() == 0u;
    let len = opts.len();
    let mut i = 0u;
    // Compile subtrees for each option
    for vec::each(opts) |opt| {
        i += 1u;
        let mut opt_cx = else_cx;
        if !exhaustive || i < len {
            opt_cx = sub_block(bcx, ~"match_case");
            match kind {
              single => Br(bcx, opt_cx.llbb),
              switch => {
                match check trans_opt(bcx, opt) {
                  single_result(r) => {
                    llvm::LLVMAddCase(sw, r.val, opt_cx.llbb);
                    bcx = r.bcx;
                  }
                }
              }
              compare => {
                let t = node_id_type(bcx, pat_id);
                let {bcx: after_cx, val: matches} = {
                    do with_scope_result(bcx, none, ~"compare_scope") |bcx| {
                        match trans_opt(bcx, opt) {
                          single_result({bcx, val}) => {
                            trans_compare(bcx, ast::eq, test_val, t, val, t)
                          }
                          range_result(
                              {val: vbegin, _}, {bcx, val: vend}) => {
                            let {bcx, val: llge} = trans_compare(
                                bcx, ast::ge, test_val, t, vbegin, t);
                            let {bcx, val: llle} = trans_compare(
                                bcx, ast::le, test_val, t, vend, t);
                            {bcx: bcx, val: And(bcx, llge, llle)}
                          }
                        }
                    }
                };
                bcx = sub_block(after_cx, ~"compare_next");
                CondBr(after_cx, matches, opt_cx.llbb, bcx.llbb);
              }
              _ => ()
            }
        } else if kind == compare { Br(bcx, else_cx.llbb); }
        let mut size = 0u;
        let mut unpacked = ~[];
        match opt {
          var(_, vdef) => {
            let args = extract_variant_args(opt_cx, pat_id, vdef, val);
            size = args.vals.len();
            unpacked = args.vals;
            opt_cx = args.bcx;
          }
          lit(_) | range(_, _) => ()
        }
        compile_submatch(opt_cx, enter_opt(bcx, m, opt, col, size, val),
                         vec::append(unpacked, vals_left), chk, exits);
    }

    // Compile the fall-through case, if any
    if !exhaustive {
        if kind == compare { Br(bcx, else_cx.llbb); }
        if kind != single {
            compile_submatch(else_cx, defaults, vals_left, chk, exits);
        }
    }
}

struct phi_binding {
    pat_id: ast::node_id;
    phi_val: ValueRef;
    mode: ast::binding_mode;
    ty: ty::t;
}

type phi_bindings_list = ~[phi_binding];

// Returns false for unreachable blocks
fn make_phi_bindings(bcx: block,
                     map: ~[exit_node],
                     ids: pat_util::pat_id_map)
    -> option<phi_bindings_list> {
    let _icx = bcx.insn_ctxt(~"alt::make_phi_bindings");
    let our_block = bcx.llbb as uint;
    let mut phi_bindings = ~[];
    for ids.each |name, node_id| {
        let mut llbbs = ~[];
        let mut vals = ~[];
        let mut binding = none;
        for vec::each(map) |ex| {
            if ex.to as uint == our_block {
                match assoc(name, ex.bound) {
                  some(b) => {
                    vec::push(llbbs, ex.from);
                    vec::push(vals, b.val);
                    binding = some(b);
                  }
                  none => ()
                }
            }
        }

        let binding = match binding {
          some(binding) => binding,
          none => {
            Unreachable(bcx);
            return none;
          }
        };

        let phi_val = Phi(bcx, val_ty(vals[0]), vals, llbbs);
        vec::push(phi_bindings, phi_binding {
            pat_id: node_id,
            phi_val: phi_val,
            mode: binding.mode,
            ty: binding.ty
        });
    }
    return some(move phi_bindings);
}

// Copies by-value bindings into their homes.
fn make_pattern_bindings(bcx: block, phi_bindings: phi_bindings_list)
    -> block {
    let mut bcx = bcx;

    for phi_bindings.each |binding| {
        let phi_val = binding.phi_val;
        match binding.mode {
            ast::bind_by_implicit_ref => {
                // use local: phi is a ptr to the value
                bcx.fcx.lllocals.insert(binding.pat_id,
                                        local_mem(phi_val));
            }
            ast::bind_by_ref(_) => {
                // use local_imm: ptr is the value
                bcx.fcx.lllocals.insert(binding.pat_id,
                                        local_imm(phi_val));
            }
            ast::bind_by_value => {
                // by value: make a new temporary and copy the value out
                let lltype = type_of::type_of(bcx.fcx.ccx, binding.ty);
                let allocation = alloca(bcx, lltype);
                let ty = binding.ty;
                bcx = copy_val(bcx, INIT, allocation,
                               load_if_immediate(bcx, phi_val, ty), ty);
                bcx.fcx.lllocals.insert(binding.pat_id,
                                        local_mem(allocation));
                add_clean(bcx, allocation, ty);
            }
        }
    }

    return bcx;
}

fn trans_alt(bcx: block,
             alt_expr: @ast::expr,
             expr: @ast::expr,
             arms: ~[ast::arm],
             mode: ast::alt_mode,
             dest: dest) -> block {
    let _icx = bcx.insn_ctxt(~"alt::trans_alt");
    do with_scope(bcx, alt_expr.info(), ~"alt") |bcx| {
        trans_alt_inner(bcx, expr, arms, mode, dest)
    }
}

fn trans_alt_inner(scope_cx: block, expr: @ast::expr, arms: ~[ast::arm],
                   mode: ast::alt_mode, dest: dest) -> block {
    let _icx = scope_cx.insn_ctxt(~"alt::trans_alt_inner");
    let bcx = scope_cx, tcx = bcx.tcx();
    let mut bodies = ~[], matches = ~[];

    let {bcx, val, _} = trans_temp_expr(bcx, expr);
    if bcx.unreachable { return bcx; }

    for vec::each(arms) |a| {
        let body = scope_block(bcx, a.body.info(), ~"case_body");
        let id_map = pat_util::pat_id_map(tcx.def_map, a.pats[0]);
        vec::push(bodies, body);
        for vec::each(a.pats) |p| {
            vec::push(matches, @{pats: ~[p],
                        bound: ~[],
                        data: @{bodycx: body, guard: a.guard,
                                id_map: id_map}});
        }
    }

    fn mk_fail(bcx: block, sp: span, msg: ~str,
                   done: @mut option<BasicBlockRef>) -> BasicBlockRef {
            match *done { some(bb) => return bb, _ => () }
            let fail_cx = sub_block(bcx, ~"case_fallthrough");
            trans_fail(fail_cx, some(sp), msg);
            *done = some(fail_cx.llbb);
            return fail_cx.llbb;
    }
    let t = node_id_type(bcx, expr.id);
    let mk_fail = match mode {
      ast::alt_check => {
        let fail_cx = @mut none;
        // Cached fail-on-fallthrough block
        some(|| mk_fail(scope_cx, expr.span, ~"non-exhaustive match failure",
                        fail_cx))
      }
      ast::alt_exhaustive => {
          let fail_cx = @mut none;
          // special case for uninhabited type
          if ty::type_is_empty(tcx, t) {
                  some(|| mk_fail(scope_cx, expr.span,
                            ~"scrutinizing value that can't exist", fail_cx))
          }
          else {
              none
          }
      }
    };
    let mut exit_map = ~[];
    let spilled = spill_if_immediate(bcx, val, t);
    compile_submatch(bcx, matches, ~[spilled], mk_fail, exit_map);

    let mut arm_cxs = ~[], arm_dests = ~[], i = 0u;
    for vec::each(arms) |a| {
        let body_cx = bodies[i];
        let id_map = pat_util::pat_id_map(tcx.def_map, a.pats[0]);
        match make_phi_bindings(body_cx, exit_map, id_map) {
            none => {}
            some(phi_bindings) => {
                let body_cx = make_pattern_bindings(body_cx, phi_bindings);
                let arm_dest = dup_for_join(dest);
                vec::push(arm_dests, arm_dest);
                let mut arm_cx = trans_block(body_cx, a.body, arm_dest);
                arm_cx = trans_block_cleanups(arm_cx, body_cx);
                vec::push(arm_cxs, arm_cx);
            }
        }
        i += 1u;
    }
    join_returns(scope_cx, arm_cxs, arm_dests, dest)
}

// Not alt-related, but similar to the pattern-munging code above
fn bind_irrefutable_pat(bcx: block, pat: @ast::pat, val: ValueRef,
                        make_copy: bool) -> block {
    let _icx = bcx.insn_ctxt(~"alt::bind_irrefutable_pat");
    let ccx = bcx.fcx.ccx;
    let mut bcx = bcx;

    // Necessary since bind_irrefutable_pat is called outside trans_alt
    match pat.node {
      ast::pat_ident(_, _,inner) => {
        if pat_is_variant(bcx.tcx().def_map, pat) { return bcx; }
        if make_copy {
            let ty = node_id_type(bcx, pat.id);
            let llty = type_of::type_of(ccx, ty);
            let alloc = alloca(bcx, llty);
            bcx = copy_val(bcx, INIT, alloc,
                                  load_if_immediate(bcx, val, ty), ty);
            bcx.fcx.lllocals.insert(pat.id, local_mem(alloc));
            add_clean(bcx, alloc, ty);
        } else { bcx.fcx.lllocals.insert(pat.id, local_mem(val)); }
        match inner {
          some(pat) => { bcx = bind_irrefutable_pat(bcx, pat, val, true); }
          _ => ()
        }
      }
      ast::pat_enum(_, sub) => {
        let vdefs = ast_util::variant_def_ids(ccx.tcx.def_map.get(pat.id));
        let args = extract_variant_args(bcx, pat.id, vdefs, val);
        let mut i = 0;
        do option::iter(sub) |sub| { for vec::each(args.vals) |argval| {
            bcx = bind_irrefutable_pat(bcx, sub[i], argval, make_copy);
            i += 1;
        }}
      }
      ast::pat_rec(fields, _) => {
        let rec_fields = ty::get_fields(node_id_type(bcx, pat.id));
        for vec::each(fields) |f| {
            let ix = option::get(ty::field_idx(f.ident, rec_fields));
            let fldptr = GEPi(bcx, val, ~[0u, ix]);
            bcx = bind_irrefutable_pat(bcx, f.pat, fldptr, make_copy);
        }
      }
      ast::pat_tup(elems) => {
        let mut i = 0u;
        for vec::each(elems) |elem| {
            let fldptr = GEPi(bcx, val, ~[0u, i]);
            bcx = bind_irrefutable_pat(bcx, elem, fldptr, make_copy);
            i += 1u;
        }
      }
      ast::pat_box(inner) => {
        let llbox = Load(bcx, val);
        let unboxed =
            GEPi(bcx, llbox, ~[0u, abi::box_field_body]);
        bcx = bind_irrefutable_pat(bcx, inner, unboxed, true);
      }
      ast::pat_uniq(inner) => {
        let llbox = Load(bcx, val);
        let unboxed =
            GEPi(bcx, llbox, ~[0u, abi::box_field_body]);
        bcx = bind_irrefutable_pat(bcx, inner, unboxed, true);
      }
      ast::pat_wild | ast::pat_lit(_) | ast::pat_range(_, _) => ()
    }
    return bcx;
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
