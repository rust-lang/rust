use driver::session::session;
use lib::llvm::llvm;
use lib::llvm::{ValueRef, BasicBlockRef};
use pat_util::*;
use build::*;
use base::*;
use syntax::ast;
use syntax::ast_util;
use syntax::ast_util::{dummy_sp, path_to_ident};
use syntax::ast::def_id;
use syntax::codemap::span;
use syntax::print::pprust::pat_to_str;
use middle::resolve::DefMap;
use back::abi;
use std::map::hashmap;
use dvec::DVec;
use datum::*;
use common::*;
use expr::Dest;

fn macros() { include!("macros.rs"); } // FIXME(#3114): Macro import/export.

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
    single_result(Result),
    range_result(Result, Result),
}
fn trans_opt(bcx: block, o: opt) -> opt_result {
    let _icx = bcx.insn_ctxt("alt::trans_opt");
    let ccx = bcx.ccx();
    let mut bcx = bcx;
    match o {
        lit(lit_expr) => {
            let datumblock = expr::trans_to_datum(bcx, lit_expr);
            return single_result(datumblock.to_result());
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
    val: ValueRef,
    mode: ast::binding_mode,
    ty: ty::t
}

type bind_map = ~[{
    ident: ast::ident,
    binding: binding
}];

fn assoc(key: ast::ident, list: bind_map) -> Option<binding> {
    for vec::each(list) |elt| {
        if elt.ident == key {
            return Some(elt.binding);
        }
    }
    return None;
}

type match_branch =
    @{pats: ~[@ast::pat],
      bound: bind_map,
      data: @{bodycx: block,
              guard: Option<@ast::expr>,
              id_map: pat_id_map}};
type match_ = ~[match_branch];

fn has_nested_bindings(m: match_, col: uint) -> bool {
    for vec::each(m) |br| {
        match br.pats[col].node {
          ast::pat_ident(_, _, Some(_)) => return true,
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
          ast::pat_ident(mode, name, Some(inner)) => {
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
                                         }}]),
                        ..*br});
          }
          _ => vec::push(result, br)
        }
    }
    result
}

type enter_pat = fn(@ast::pat) -> Option<~[@ast::pat]>;

fn enter_match(bcx: block, dm: DefMap, m: match_, col: uint, val: ValueRef,
               e: enter_pat) -> match_ {
    let mut result = ~[];
    for vec::each(m) |br| {
        match e(br.pats[col]) {
          Some(sub) => {
            let pats = vec::append(
                vec::append(sub, vec::view(br.pats, 0u, col)),
                vec::view(br.pats, col + 1u, br.pats.len()));
            let self = br.pats[col];
            let bound = match self.node {
              ast::pat_ident(mode, name, None)
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
            vec::push(result, @{pats: pats, bound: bound, ..*br});
          }
          None => ()
        }
    }
    return result;
}

fn enter_default(bcx: block, dm: DefMap, m: match_, col: uint, val: ValueRef)
              -> match_ {

    do enter_match(bcx, dm, m, col, val) |p| {
        match p.node {
          ast::pat_wild | ast::pat_rec(_, _) | ast::pat_tup(_) |
          ast::pat_struct(*) => Some(~[]),
          ast::pat_ident(_, _, None) if !pat_is_variant(dm, p) => Some(~[]),
          _ => None
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
              Some(option::get_default(subpats,
                     vec::from_elem(variant_size, dummy))) }
            else { None }
          }
          ast::pat_ident(_, _, None) if pat_is_variant(tcx.def_map, p) => {
            if opt_eq(tcx, variant_opt(tcx, p.id), opt) { Some(~[]) }
            else { None }
          }
          ast::pat_lit(l) => {
            if opt_eq(tcx, lit(l), opt) { Some(~[]) } else { None }
          }
          ast::pat_range(l1, l2) => {
            if opt_eq(tcx, range(l1, l2), opt) { Some(~[]) } else { None }
          }
          _ => Some(vec::from_elem(variant_size, dummy))
        }
    }
}

fn enter_rec_or_struct(bcx: block, dm: DefMap, m: match_, col: uint,
                       fields: ~[ast::ident], val: ValueRef) -> match_ {
    let dummy = @{id: 0, node: ast::pat_wild, span: dummy_sp()};
    do enter_match(bcx, dm, m, col, val) |p| {
        match p.node {
          ast::pat_rec(fpats, _) | ast::pat_struct(_, fpats, _) => {
            let mut pats = ~[];
            for vec::each(fields) |fname| {
                match fpats.find(|p| p.ident == fname) {
                    None => vec::push(pats, dummy),
                    Some(pat) => vec::push(pats, pat.pat)
                }
            }
            Some(pats)
          }
          _ => Some(vec::from_elem(fields.len(), dummy))
        }
    }
}

fn enter_tup(bcx: block, dm: DefMap, m: match_, col: uint, val: ValueRef,
             n_elts: uint) -> match_ {
    let dummy = @{id: 0, node: ast::pat_wild, span: dummy_sp()};
    do enter_match(bcx, dm, m, col, val) |p| {
        match p.node {
          ast::pat_tup(elts) => Some(elts),
          _ => Some(vec::from_elem(n_elts, dummy))
        }
    }
}

fn enter_box(bcx: block, dm: DefMap, m: match_, col: uint, val: ValueRef)
          -> match_ {
    let dummy = @{id: 0, node: ast::pat_wild, span: dummy_sp()};
    do enter_match(bcx, dm, m, col, val) |p| {
        match p.node {
          ast::pat_box(sub) => Some(~[sub]),
          _ => Some(~[dummy])
        }
    }
}

fn enter_uniq(bcx: block, dm: DefMap, m: match_, col: uint, val: ValueRef)
           -> match_ {
    let dummy = @{id: 0, node: ast::pat_wild, span: dummy_sp()};
    do enter_match(bcx, dm, m, col, val) |p| {
        match p.node {
          ast::pat_uniq(sub) => Some(~[sub]),
          _ => Some(~[dummy])
        }
    }
}

fn get_options(ccx: @crate_ctxt, m: match_, col: uint) -> ~[opt] {
    fn add_to_set(tcx: ty::ctxt, &&set: DVec<opt>, val: opt) {
        if set.any(|l| opt_eq(tcx, l, val)) {return;}
        set.push(val);
    }

    let found = DVec();
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
    return vec::from_mut(dvec::unwrap(move found));
}

fn extract_variant_args(bcx: block, pat_id: ast::node_id,
                        vdefs: {enm: def_id, var: def_id}, val: ValueRef) ->
   {vals: ~[ValueRef], bcx: block} {
    let _icx = bcx.insn_ctxt("alt::extract_variant_args");
    let ccx = bcx.fcx.ccx;
    let enum_ty_substs = match ty::get(node_id_type(bcx, pat_id))
        .struct {
      ty::ty_enum(id, substs) => { assert id == vdefs.enm; substs.tps }
      _ => bcx.sess().bug(~"extract_variant_args: pattern has non-enum type")
    };
    let mut blobptr = val;
    let variants = ty::enum_variants(ccx.tcx, vdefs.enm);
    let size = ty::enum_variant_with_id(ccx.tcx, vdefs.enm,
                                        vdefs.var).args.len();
    if size > 0u && (*variants).len() != 1u {
        let enumptr =
            PointerCast(bcx, val, T_opaque_enum_ptr(ccx));
        blobptr = GEPi(bcx, enumptr, [0u, 1u]);
    }
    let vdefs_tg = vdefs.enm;
    let vdefs_var = vdefs.var;
    let args = do vec::from_fn(size) |i| {
        GEP_enum(bcx, blobptr, vdefs_tg, vdefs_var,
                 enum_ty_substs, i)
    };
    return {vals: args, bcx: bcx};
}

fn collect_record_or_struct_fields(m: match_, col: uint) -> ~[ast::ident] {
    let mut fields: ~[ast::ident] = ~[];
    for vec::each(m) |br| {
        match br.pats[col].node {
          ast::pat_rec(fs, _) => extend(&mut fields, fs),
          ast::pat_struct(_, fs, _) => extend(&mut fields, fs),
          _ => ()
        }
    }
    return fields;

    fn extend(idents: &mut ~[ast::ident], field_pats: &[ast::field_pat]) {
        for field_pats.each |field_pat| {
            let field_ident = field_pat.ident;
            if !vec::any(*idents, |x| x == field_ident) {
                vec::push(*idents, field_ident);
            }
        }
    }
}

fn root_pats_as_necessary(bcx: block, m: match_, col: uint, val: ValueRef) {
    for vec::each(m) |br| {
        let pat_id = br.pats[col].id;

        match bcx.ccx().maps.root_map.find({id:pat_id, derefs:0u}) {
            None => (),
            Some(scope_id) => {
                // Note: the scope_id will always be the id of the alt.  See
                // the extended comment in rustc::middle::borrowck::preserve()
                // for details (look for the case covering cat_discr).

                let datum = Datum {val: val, ty: node_id_type(bcx, pat_id),
                                   mode: ByRef, source: FromLvalue};
                datum.root(bcx, scope_id);
                return; // if we kept going, we'd only re-root the same value
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
          ast::pat_ident(_, _, Some(p)) => score(p),
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

enum branch_kind { no_branch, single, switch, compare, }

impl branch_kind : cmp::Eq {
    pure fn eq(&&other: branch_kind) -> bool {
        (self as uint) == (other as uint)
    }
    pure fn ne(&&other: branch_kind) -> bool { !self.eq(other) }
}

// Compiles a comparison between two things.
fn trans_compare(cx: block, op: ast::binop, lhs: ValueRef,
                 _lhs_t: ty::t, rhs: ValueRef, rhs_t: ty::t) -> Result {
    let _icx = cx.insn_ctxt("trans_compare");
    if ty::type_is_scalar(rhs_t) {
      let rs = compare_scalar_types(cx, lhs, rhs, rhs_t, op);
      return rslt(rs.bcx, rs.val);
    }

    // Determine the operation we need.
    let llop = {
        match op {
          ast::eq | ast::ne => C_u8(abi::cmp_glue_op_eq),
          ast::lt | ast::ge => C_u8(abi::cmp_glue_op_lt),
          ast::le | ast::gt => C_u8(abi::cmp_glue_op_le),
          _ => cx.tcx().sess.bug(~"trans_compare got non-comparison-op")
        }
    };

    let cmpval = glue::call_cmp_glue(cx, lhs, rhs, rhs_t, llop);

    // Invert the result if necessary.
    match op {
      ast::eq | ast::lt | ast::le => rslt(cx, cmpval),
      ast::ne | ast::ge | ast::gt => rslt(cx, Not(cx, cmpval)),
      _ => cx.tcx().sess.bug(~"trans_compare got non-comparison-op")
    }
}

fn compile_submatch(bcx: block, m: match_, vals: ~[ValueRef],
                    chk: Option<mk_fail>, &exits: ~[exit_node]) {
    /*
      For an empty match, a fall-through case must exist
     */
    assert(m.len() > 0u || is_some(chk));
    let _icx = bcx.insn_ctxt("alt::compile_submatch");
    let mut bcx = bcx;
    let tcx = bcx.tcx(), dm = tcx.def_map;
    if m.len() == 0u {
        Br(bcx, option::get(chk)());
        return;
    }
    if m[0].pats.len() == 0u {
        let data = m[0].data;
        match data.guard {
          Some(e) => {
            // Temporarily set bindings. They'll be rewritten to PHI nodes
            // for the actual arm block.
            //
            // Also, in the case of by-value, do the copy now.

            for data.id_map.each |key, val| {
                let binding = assoc(key, m[0].bound).get();
                let datum = Datum {val: binding.val, ty: binding.ty,
                                   mode: ByRef, source: FromLvalue};

                if binding.mode == ast::bind_by_value {
                    let llty = type_of::type_of(bcx.fcx.ccx, binding.ty);
                    let alloc = alloca(bcx, llty);
                    bcx = datum.copy_to(bcx, INIT, alloc);
                    bcx.fcx.lllocals.insert(val, local_mem(alloc));
                    add_clean(bcx, alloc, binding.ty);
                } else if binding.mode == ast::bind_by_move {
                    fail ~"can't translate bind_by_move into a pattern guard";
                } else {
                    bcx.fcx.lllocals.insert(val, local_mem(datum.val));
                }
            };

            let Result {bcx: guard_cx, val} = {
                do with_scope_result(bcx, e.info(), ~"guard") |bcx| {
                    expr::trans_to_appropriate_llval(bcx, e)
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

    let rec_fields = collect_record_or_struct_fields(m, col);
    if rec_fields.len() > 0 {
        let pat_ty = node_id_type(bcx, pat_id);
        do expr::with_field_tys(tcx, pat_ty) |_has_dtor, field_tys| {
            let mut rec_vals = ~[];
            for vec::each(rec_fields) |field_name| {
                let ix = ty::field_idx_strict(tcx, field_name, field_tys);
                vec::push(rec_vals, GEPi(bcx, val, struct_field(ix)));
            }
            compile_submatch(
                bcx,
                enter_rec_or_struct(bcx, dm, m, col, rec_fields, val),
                vec::append(rec_vals, vals_left),
                chk,
                exits);
        }
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
            vec::push(tup_vals, GEPi(bcx, val, [0u, i]));
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
            GEPi(bcx, box_no_addrspace, [0u, abi::box_field_body]);
        compile_submatch(bcx, enter_box(bcx, dm, m, col, val),
                         vec::append(~[unboxed], vals_left), chk, exits);
        return;
    }

    if any_uniq_pat(m, col) {
        let llbox = Load(bcx, val);
        let box_no_addrspace = non_gc_box_cast(bcx, llbox);
        let unboxed =
            GEPi(bcx, box_no_addrspace, [0u, abi::box_field_body]);
        compile_submatch(bcx, enter_uniq(bcx, dm, m, col, val),
                         vec::append(~[unboxed], vals_left), chk, exits);
        return;
    }

    // Decide what kind of branch we need
    let opts = get_options(ccx, m, col);
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
                let discrimptr = GEPi(bcx, enumptr, [0u, 0u]);
                test_val = Load(bcx, discrimptr);
                kind = switch;
            }
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
                match trans_opt(bcx, opt) {
                  single_result(r) => {
                    llvm::LLVMAddCase(sw, r.val, opt_cx.llbb);
                    bcx = r.bcx;
                  }
                  _ => bcx.sess().bug(~"in compile_submatch, expected \
                         trans_opt to return a single_result")
                }
              }
              compare => {
                let t = node_id_type(bcx, pat_id);
                let Result {bcx: after_cx, val: matches} = {
                    do with_scope_result(bcx, None, ~"compare_scope") |bcx| {
                        match trans_opt(bcx, opt) {
                            single_result(
                                Result {bcx, val}) =>
                            {
                                trans_compare(bcx, ast::eq, test_val,
                                              t, val, t)
                            }
                            range_result(
                                Result {val: vbegin, _},
                                Result {bcx, val: vend}) =>
                            {
                                let Result {bcx, val: llge} = trans_compare(
                                    bcx, ast::ge, test_val, t, vbegin, t);
                                let Result {bcx, val: llle} = trans_compare(
                                    bcx, ast::le, test_val, t, vend, t);
                                rslt(bcx, And(bcx, llge, llle))
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
    pat_id: ast::node_id,
    phi_val: ValueRef,
    mode: ast::binding_mode,
    ty: ty::t
}

type phi_bindings_list = ~[phi_binding];

// Returns false for unreachable blocks
fn make_phi_bindings(bcx: block,
                     map: ~[exit_node],
                     ids: pat_util::pat_id_map)
    -> Option<phi_bindings_list> {
    let _icx = bcx.insn_ctxt("alt::make_phi_bindings");
    let our_block = bcx.llbb as uint;
    let mut phi_bindings = ~[];
    for ids.each |name, node_id| {
        let mut llbbs = ~[];
        let mut vals = ~[];
        let mut binding = None;
        for vec::each(map) |ex| {
            if ex.to as uint == our_block {
                match assoc(name, ex.bound) {
                  Some(b) => {
                    vec::push(llbbs, ex.from);
                    vec::push(vals, b.val);
                    binding = Some(b);
                  }
                  None => ()
                }
            }
        }

        let binding = match binding {
          Some(binding) => binding,
          None => {
            Unreachable(bcx);
            return None;
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
    return Some(move phi_bindings);
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
            ast::bind_by_value | ast::bind_by_move => {
                // by value: make a new temporary and copy the value out
                let phi_datum = Datum {val: phi_val, ty: binding.ty,
                                       mode: ByRef, source: FromLvalue};
                let scratch = scratch_datum(bcx, binding.ty, false);
                if binding.mode == ast::bind_by_value {
                    phi_datum.copy_to_datum(bcx, INIT, scratch);
                } else {
                    phi_datum.move_to_datum(bcx, INIT, scratch);
                }
                bcx.fcx.lllocals.insert(binding.pat_id,
                                        local_mem(scratch.val));
                add_clean(bcx, scratch.val, binding.ty);
            }
        }
    }

    return bcx;
}

fn trans_alt(bcx: block,
             alt_expr: @ast::expr,
             discr_expr: @ast::expr,
             arms: ~[ast::arm],
             dest: Dest) -> block {
    let _icx = bcx.insn_ctxt("alt::trans_alt");
    do with_scope(bcx, alt_expr.info(), ~"alt") |bcx| {
        trans_alt_inner(bcx, discr_expr, arms, dest)
    }
}

fn trans_alt_inner(scope_cx: block,
                   discr_expr: @ast::expr,
                   arms: ~[ast::arm],
                   dest: Dest) -> block {
    let _icx = scope_cx.insn_ctxt("alt::trans_alt_inner");
    let mut bcx = scope_cx;
    let tcx = bcx.tcx();

    let discr_datum = unpack_datum!(bcx, {
        expr::trans_to_datum(bcx, discr_expr)
    });
    if bcx.unreachable {
        return bcx;
    }

    let mut bodies = ~[], matches = ~[];
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

    let t = node_id_type(bcx, discr_expr.id);
    let chk = {
        if ty::type_is_empty(tcx, t) {
            // Special case for empty types
            let fail_cx = @mut None;
            Some(|| mk_fail(scope_cx, discr_expr.span,
                            ~"scrutinizing value that can't exist", fail_cx))
        } else {
            None
        }
    };
    let mut exit_map = ~[];
    let lldiscr = discr_datum.to_ref_llval(bcx);
    compile_submatch(bcx, matches, ~[lldiscr], chk, exit_map);

    let mut arm_cxs = ~[], i = 0u;
    for vec::each(arms) |a| {
        let body_cx = bodies[i];
        let id_map = pat_util::pat_id_map(tcx.def_map, a.pats[0]);
        match make_phi_bindings(body_cx, exit_map, id_map) {
            None => {}
            Some(phi_bindings) => {
                let body_cx = make_pattern_bindings(body_cx, phi_bindings);
                let mut arm_cx =
                    controlflow::trans_block(body_cx, a.body, dest);
                arm_cx = trans_block_cleanups(arm_cx,
                                              block_cleanups(body_cx));
                vec::push(arm_cxs, arm_cx);
            }
        }
        i += 1u;
    }
    return controlflow::join_blocks(scope_cx, arm_cxs);

    fn mk_fail(bcx: block, sp: span, msg: ~str,
               done: @mut Option<BasicBlockRef>) -> BasicBlockRef {
        match *done { Some(bb) => return bb, _ => () }
        let fail_cx = sub_block(bcx, ~"case_fallthrough");
        controlflow::trans_fail(fail_cx, Some(sp), msg);
        *done = Some(fail_cx.llbb);
        return fail_cx.llbb;
    }
}

// Not alt-related, but similar to the pattern-munging code above
fn bind_irrefutable_pat(bcx: block, pat: @ast::pat, val: ValueRef,
                        make_copy: bool) -> block {
    let _icx = bcx.insn_ctxt("alt::bind_irrefutable_pat");
    let ccx = bcx.fcx.ccx;
    let mut bcx = bcx;

    // Necessary since bind_irrefutable_pat is called outside trans_alt
    match pat.node {
        ast::pat_ident(_, _,inner) => {
            if pat_is_variant(bcx.tcx().def_map, pat) {
                return bcx;
            }

            if make_copy {
                let binding_ty = node_id_type(bcx, pat.id);
                let datum = Datum {val: val, ty: binding_ty,
                                   mode: ByRef, source: FromRvalue};
                let scratch = scratch_datum(bcx, binding_ty, false);
                datum.copy_to_datum(bcx, INIT, scratch);
                bcx.fcx.lllocals.insert(pat.id, local_mem(scratch.val));
                add_clean(bcx, scratch.val, binding_ty);
            } else {
                bcx.fcx.lllocals.insert(pat.id, local_mem(val));
            }

            for inner.each |inner_pat| {
                bcx = bind_irrefutable_pat(bcx, inner_pat, val, true);
            }
      }
        ast::pat_enum(_, sub_pats) => {
            let pat_def = ccx.tcx.def_map.get(pat.id);
            let vdefs = ast_util::variant_def_ids(pat_def);
            let args = extract_variant_args(bcx, pat.id, vdefs, val);
            for sub_pats.each |sub_pat| {
                for vec::eachi(args.vals) |i, argval| {
                    bcx = bind_irrefutable_pat(bcx, sub_pat[i],
                                               argval, make_copy);
                }
            }
        }
        ast::pat_rec(fields, _) | ast::pat_struct(_, fields, _) => {
            let tcx = bcx.tcx();
            let pat_ty = node_id_type(bcx, pat.id);
            do expr::with_field_tys(tcx, pat_ty) |_has_dtor, field_tys| {
                for vec::each(fields) |f| {
                    let ix = ty::field_idx_strict(tcx, f.ident, field_tys);
                    let fldptr = GEPi(bcx, val, struct_field(ix));
                    bcx = bind_irrefutable_pat(bcx, f.pat, fldptr, make_copy);
                }
            }
        }
        ast::pat_tup(elems) => {
            for vec::eachi(elems) |i, elem| {
                let fldptr = GEPi(bcx, val, [0u, i]);
                bcx = bind_irrefutable_pat(bcx, elem, fldptr, make_copy);
            }
        }
        ast::pat_box(inner) | ast::pat_uniq(inner) |
        ast::pat_region(inner) => {
            let llbox = Load(bcx, val);
            let unboxed = GEPi(bcx, llbox, [0u, abi::box_field_body]);
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
