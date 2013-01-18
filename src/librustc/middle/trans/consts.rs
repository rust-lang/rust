// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use middle::const_eval;
use middle::trans::base::get_insn_ctxt;
use middle::trans::common::*;
use middle::trans::consts;
use middle::trans::expr;
use middle::ty;

use syntax::{ast, ast_util, codemap, ast_map};

fn const_lit(cx: @crate_ctxt, e: @ast::expr, lit: ast::lit)
    -> ValueRef {
    let _icx = cx.insn_ctxt("trans_lit");
    match lit.node {
      ast::lit_int(i, t) => C_integral(T_int_ty(cx, t), i as u64, True),
      ast::lit_uint(u, t) => C_integral(T_uint_ty(cx, t), u, False),
      ast::lit_int_unsuffixed(i) => {
        let lit_int_ty = ty::node_id_to_type(cx.tcx, e.id);
        match ty::get(lit_int_ty).sty {
          ty::ty_int(t) => {
            C_integral(T_int_ty(cx, t), i as u64, True)
          }
          ty::ty_uint(t) => {
            C_integral(T_uint_ty(cx, t), i as u64, False)
          }
          _ => cx.sess.span_bug(lit.span,
                                ~"integer literal doesn't have a type")
        }
      }
      ast::lit_float(fs, t) => C_floating(/*bad*/copy *fs, T_float_ty(cx, t)),
      ast::lit_float_unsuffixed(fs) => {
        let lit_float_ty = ty::node_id_to_type(cx.tcx, e.id);
        match ty::get(lit_float_ty).sty {
          ty::ty_float(t) => {
            C_floating(/*bad*/copy *fs, T_float_ty(cx, t))
          }
          _ => {
            cx.sess.span_bug(lit.span,
                             ~"floating point literal doesn't have the right \
                               type");
          }
        }
      }
      ast::lit_bool(b) => C_bool(b),
      ast::lit_nil => C_nil(),
      ast::lit_str(s) => C_estr_slice(cx, /*bad*/copy *s)
    }
}

fn const_ptrcast(cx: @crate_ctxt, a: ValueRef, t: TypeRef) -> ValueRef {
    unsafe {
        let b = llvm::LLVMConstPointerCast(a, T_ptr(t));
        assert cx.const_globals.insert(b as int, a);
        b
    }
}

fn const_vec(cx: @crate_ctxt, e: @ast::expr, es: &[@ast::expr])
    -> (ValueRef, ValueRef, TypeRef) {
    unsafe {
        let vec_ty = ty::expr_ty(cx.tcx, e);
        let unit_ty = ty::sequence_element_type(cx.tcx, vec_ty);
        let llunitty = type_of::type_of(cx, unit_ty);
        let v = C_array(llunitty, es.map(|e| const_expr(cx, *e)));
        let unit_sz = shape::llsize_of(cx, llunitty);
        let sz = llvm::LLVMConstMul(C_uint(cx, es.len()), unit_sz);
        return (v, sz, llunitty);
    }
}

fn const_deref(cx: @crate_ctxt, v: ValueRef) -> ValueRef {
    unsafe {
        let v = match cx.const_globals.find(v as int) {
            Some(v) => v,
            None => v
        };
        assert llvm::LLVMIsGlobalConstant(v) == True;
        let v = llvm::LLVMGetInitializer(v);
        v
    }
}

fn const_get_elt(cx: @crate_ctxt, v: ValueRef, us: &[c_uint]) -> ValueRef {
    unsafe {
        let r = do vec::as_imm_buf(us) |p, len| {
            llvm::LLVMConstExtractValue(v, p, len as c_uint)
        };

        debug!("const_get_elt(v=%s, us=%?, r=%s)",
               val_str(cx.tn, v), us, val_str(cx.tn, r));

        return r;
    }
}

fn const_autoderef(cx: @crate_ctxt, ty: ty::t, v: ValueRef)
    -> (ty::t, ValueRef) {
    let mut t1 = ty;
    let mut v1 = v;
    loop {
        // Only rptrs can be autoderef'ed in a const context.
        match ty::get(ty).sty {
            ty::ty_rptr(_, mt) => {
                t1 = mt.ty;
                v1 = const_deref(cx, v1);
            }
            _ => return (t1,v1)
        }
    }
}

fn get_const_val(cx: @crate_ctxt, def_id: ast::def_id) -> ValueRef {
    if !ast_util::is_local(def_id) {
        cx.tcx.sess.bug(~"cross-crate constants");
    }
    if !cx.const_values.contains_key(def_id.node) {
        match cx.tcx.items.get(def_id.node) {
            ast_map::node_item(@ast::item {
                node: ast::item_const(_, subexpr), _
            }, _) => {
                trans_const(cx, subexpr, def_id.node);
            }
            _ => cx.tcx.sess.bug(~"expected a const to be an item")
        }
    }
    cx.const_values.get(def_id.node)
}

fn const_expr(cx: @crate_ctxt, e: @ast::expr) -> ValueRef {
    unsafe {
        let _icx = cx.insn_ctxt("const_expr");
        return match /*bad*/copy e.node {
          ast::expr_lit(lit) => consts::const_lit(cx, e, *lit),
          ast::expr_binary(b, e1, e2) => {
            let te1 = const_expr(cx, e1);
            let te2 = const_expr(cx, e2);

            let te2 = base::cast_shift_const_rhs(b, te1, te2);

            /* Neither type is bottom, and we expect them to be unified
             * already, so the following is safe. */
            let ty = ty::expr_ty(cx.tcx, e1);
            let is_float = ty::type_is_fp(ty);
            let signed = ty::type_is_signed(ty);
            return match b {
              ast::add   => {
                if is_float { llvm::LLVMConstFAdd(te1, te2) }
                else        { llvm::LLVMConstAdd(te1, te2) }
              }
              ast::subtract => {
                if is_float { llvm::LLVMConstFSub(te1, te2) }
                else        { llvm::LLVMConstSub(te1, te2) }
              }
              ast::mul    => {
                if is_float { llvm::LLVMConstFMul(te1, te2) }
                else        { llvm::LLVMConstMul(te1, te2) }
              }
              ast::div    => {
                if is_float    { llvm::LLVMConstFDiv(te1, te2) }
                else if signed { llvm::LLVMConstSDiv(te1, te2) }
                else           { llvm::LLVMConstUDiv(te1, te2) }
              }
              ast::rem    => {
                if is_float    { llvm::LLVMConstFRem(te1, te2) }
                else if signed { llvm::LLVMConstSRem(te1, te2) }
                else           { llvm::LLVMConstURem(te1, te2) }
              }
              ast::and    |
              ast::or     => cx.sess.span_unimpl(e.span, ~"binop logic"),
              ast::bitxor => llvm::LLVMConstXor(te1, te2),
              ast::bitand => llvm::LLVMConstAnd(te1, te2),
              ast::bitor  => llvm::LLVMConstOr(te1, te2),
              ast::shl    => llvm::LLVMConstShl(te1, te2),
              ast::shr    => {
                if signed { llvm::LLVMConstAShr(te1, te2) }
                else      { llvm::LLVMConstLShr(te1, te2) }
              }
              ast::eq     |
              ast::lt     |
              ast::le     |
              ast::ne     |
              ast::ge     |
              ast::gt     => cx.sess.span_unimpl(e.span, ~"binop comparator")
            }
          }
          ast::expr_unary(u, e) => {
            let te = const_expr(cx, e);
            let ty = ty::expr_ty(cx.tcx, e);
            let is_float = ty::type_is_fp(ty);
            return match u {
              ast::box(_)  |
              ast::uniq(_) |
              ast::deref  => const_deref(cx, te),
              ast::not    => llvm::LLVMConstNot(te),
              ast::neg    => {
                if is_float { llvm::LLVMConstFNeg(te) }
                else        { llvm::LLVMConstNeg(te) }
              }
            }
          }
          ast::expr_field(base, field, _) => {
              let bt = ty::expr_ty(cx.tcx, base);
              let bv = const_expr(cx, base);
              let (bt, bv) = const_autoderef(cx, bt, bv);
              do expr::with_field_tys(cx.tcx, bt, None) |_, field_tys| {
                  let ix = ty::field_idx_strict(cx.tcx, field, field_tys);

                  // Note: ideally, we'd use `struct_field()` here instead
                  // of hardcoding [0, ix], but we can't because it yields
                  // the wrong type and also inserts an extra 0 that is
                  // not needed in the constant variety:
                  const_get_elt(cx, bv, [0, ix as c_uint])
              }
          }

          ast::expr_index(base, index) => {
              let bt = ty::expr_ty(cx.tcx, base);
              let bv = const_expr(cx, base);
              let (bt, bv) = const_autoderef(cx, bt, bv);
              let iv = match const_eval::eval_const_expr(cx.tcx, index) {
                  const_eval::const_int(i) => i as u64,
                  const_eval::const_uint(u) => u,
                  _ => cx.sess.span_bug(index.span,
                                        ~"index is not an integer-constant \
                                          expression")
              };
              let (arr, _len) = match ty::get(bt).sty {
                  ty::ty_evec(_, vstore) | ty::ty_estr(vstore) =>
                      match vstore {
                      ty::vstore_fixed(u) =>
                          (bv, C_uint(cx, u)),

                      ty::vstore_slice(_) => {
                          let unit_ty = ty::sequence_element_type(cx.tcx, bt);
                          let llunitty = type_of::type_of(cx, unit_ty);
                          let unit_sz = shape::llsize_of(cx, llunitty);

                          (const_deref(cx, const_get_elt(cx, bv, [0])),
                           llvm::LLVMConstUDiv(const_get_elt(cx, bv, [1]),
                                               unit_sz))
                      },
                      _ => cx.sess.span_bug(base.span,
                                            ~"index-expr base must be \
                                              fixed-size or slice")
                  },
                  _ =>  cx.sess.span_bug(base.span,
                                         ~"index-expr base must be \
                                           a vector or string type")
              };

              // FIXME #3169: This is a little odd but it arises due to a
              // weird wrinkle in LLVM: it doesn't appear willing to let us
              // call LLVMConstIntGetZExtValue on the size element of the
              // slice, or seemingly any integer-const involving a sizeof()
              // call. Despite that being "a const", it's not the kind of
              // const you can ask for the integer-value of, evidently. This
              // might be an LLVM bug, not sure. In any case, to work around
              // this we drop down to the array-type level here and just ask
              // how long the array-type itself is, ignoring the length we
              // pulled out of the slice. This in turn only works because we
              // picked out the original globalvar via const_deref and so can
              // recover the array-size of the underlying array, and all this
              // will hold together exactly as long as we _don't_ support
              // const sub-slices (that is, slices that represent something
              // other than a whole array).  At that point we'll have more and
              // uglier work to do here, but for now this should work.
              //
              // In the future, what we should be doing here is the
              // moral equivalent of:
              //
              // let len = llvm::LLVMConstIntGetZExtValue(len) as u64;
              //
              // but we might have to do substantially more magic to
              // make it work. Or figure out what is causing LLVM to
              // not want to consider sizeof() a constant expression
              // we can get the value (as a number) out of.

              let len = llvm::LLVMGetArrayLength(val_ty(arr)) as u64;
              let len = match ty::get(bt).sty {
                  ty::ty_estr(*) => {assert len > 0; len - 1},
                  _ => len
              };
              if iv >= len {
                  // FIXME #3170: report this earlier on in the const-eval
                  // pass. Reporting here is a bit late.
                  cx.sess.span_err(e.span,
                                   ~"const index-expr is out of bounds");
              }
              const_get_elt(cx, arr, [iv as c_uint])
          }
          ast::expr_cast(base, _) => {
            let ety = ty::expr_ty(cx.tcx, e);
            let llty = type_of::type_of(cx, ety);
            let basety = ty::expr_ty(cx.tcx, base);
            let v = const_expr(cx, base);
            match (expr::cast_type_kind(basety),
                   expr::cast_type_kind(ety)) {

              (expr::cast_integral, expr::cast_integral) => {
                let s = if ty::type_is_signed(basety) { True } else { False };
                llvm::LLVMConstIntCast(v, llty, s)
              }
              (expr::cast_integral, expr::cast_float) => {
                if ty::type_is_signed(basety) {
                    llvm::LLVMConstSIToFP(v, llty)
                } else {
                    llvm::LLVMConstUIToFP(v, llty)
                }
              }
              (expr::cast_float, expr::cast_float) => {
                llvm::LLVMConstFPCast(v, llty)
              }
              (expr::cast_float, expr::cast_integral) => {
                if ty::type_is_signed(ety) { llvm::LLVMConstFPToSI(v, llty) }
                else { llvm::LLVMConstFPToUI(v, llty) }
              }
              _ => {
                cx.sess.impossible_case(e.span,
                                        ~"bad combination of types for cast")
              }
            }
          }
          ast::expr_addr_of(ast::m_imm, sub) => {
            let cv = const_expr(cx, sub);
            let subty = ty::expr_ty(cx.tcx, sub),
            llty = type_of::type_of(cx, subty);
            let gv = do str::as_c_str("const") |name| {
                llvm::LLVMAddGlobal(cx.llmod, llty, name)
            };
            llvm::LLVMSetInitializer(gv, cv);
            llvm::LLVMSetGlobalConstant(gv, True);
            gv
          }
          ast::expr_tup(es) => {
            C_struct(es.map(|e| const_expr(cx, *e)))
          }
          ast::expr_rec(ref fs, None) => {
              C_struct([C_struct(
                  (*fs).map(|f| const_expr(cx, f.node.expr)))])
          }
          ast::expr_struct(_, ref fs, _) => {
              let ety = ty::expr_ty(cx.tcx, e);
              let cs = do expr::with_field_tys(cx.tcx,
                                               ety,
                                               None) |_hd, field_tys| {
                  field_tys.map(|field_ty| {
                      match fs.find(|f| field_ty.ident == f.node.ident) {
                          Some(ref f) => const_expr(cx, (*f).node.expr),
                          None => {
                              cx.tcx.sess.span_bug(
                                  e.span, ~"missing struct field");
                          }
                      }
                  })
              };
              let llty = type_of::type_of(cx, ety);
              C_named_struct(llty, [C_struct(cs)])
          }
          ast::expr_vec(es, ast::m_imm) => {
            let (v, _, _) = const_vec(cx, e, es);
            v
          }
          ast::expr_vstore(e, ast::expr_vstore_fixed(_)) => {
            const_expr(cx, e)
          }
          ast::expr_vstore(sub, ast::expr_vstore_slice) => {
            match /*bad*/copy sub.node {
              ast::expr_lit(lit) => {
                match lit.node {
                  ast::lit_str(*) => { const_expr(cx, sub) }
                  _ => { cx.sess.span_bug(e.span,
                                          ~"bad const-slice lit") }
                }
              }
              ast::expr_vec(es, ast::m_imm) => {
                let (cv, sz, llunitty) = const_vec(cx, e, es);
                let llty = val_ty(cv);
                let gv = do str::as_c_str("const") |name| {
                    llvm::LLVMAddGlobal(cx.llmod, llty, name)
                };
                llvm::LLVMSetInitializer(gv, cv);
                llvm::LLVMSetGlobalConstant(gv, True);
                let p = const_ptrcast(cx, gv, llunitty);
                C_struct(~[p, sz])
              }
              _ => cx.sess.span_bug(e.span,
                                    ~"bad const-slice expr")
            }
          }
          ast::expr_path(pth) => {
            assert pth.types.len() == 0;
            match cx.tcx.def_map.find(e.id) {
                Some(ast::def_fn(def_id, purity)) => {
                    assert ast_util::is_local(def_id);
                    let f = base::get_item_val(cx, def_id.node);
                    match purity {
                      ast::extern_fn => llvm::LLVMConstPointerCast(f, T_ptr(T_i8())),
                      _ => C_struct(~[f, C_null(T_opaque_box_ptr(cx))])
                    }
                }
                Some(ast::def_const(def_id)) => {
                    get_const_val(cx, def_id)
                }
                Some(ast::def_variant(enum_did, variant_did)) => {
                    // Note that we know this is a C-like (nullary) enum
                    // variant or we wouldn't have gotten here -- the constant
                    // checker forbids paths that don't map to C-like enum
                    // variants.
                    let lldiscrim = base::get_discrim_val(cx, e.span,
                                                          enum_did,
                                                          variant_did);
                    C_struct(~[lldiscrim])
                }
                Some(ast::def_struct(_)) => {
                    let ety = ty::expr_ty(cx.tcx, e);
                    let llty = type_of::type_of(cx, ety);
                    C_null(llty)
                }
                _ => {
                    cx.sess.span_bug(e.span,
                                     ~"expected a const, fn, or variant def")
                }
            }
          }
          ast::expr_call(callee, args, _) => {
            match cx.tcx.def_map.find(callee.id) {
                Some(ast::def_struct(def_id)) => {
                    let ety = ty::expr_ty(cx.tcx, e);
                    let llty = type_of::type_of(cx, ety);
                    let llstructbody =
                        C_struct(args.map(|a| const_expr(cx, *a)));
                    if ty::ty_dtor(cx.tcx, def_id).is_present() {
                        C_named_struct(llty, ~[ llstructbody, C_u8(0) ])
                    } else {
                        C_named_struct(llty, ~[ llstructbody ])
                    }
                }
            Some(ast::def_variant(tid, vid)) => {
                let ety = ty::expr_ty(cx.tcx, e);
                let degen = ty::enum_is_univariant(cx.tcx, tid);
                let size = shape::static_size_of_enum(cx, ety);

                let discrim = base::get_discrim_val(cx, e.span, tid, vid);
                let c_args = C_struct(args.map(|a| const_expr(cx, *a)));

                let fields = if !degen {
                    ~[discrim, c_args]
                } else if size == 0 {
                    ~[discrim]
                } else {
                    ~[c_args]
                };

                C_struct(fields)
            }
                _ => cx.sess.span_bug(e.span, ~"expected a struct def")
            }
          }
          ast::expr_paren(e) => { return const_expr(cx, e); }
          _ => cx.sess.span_bug(e.span,
                ~"bad constant expression type in consts::const_expr")
        };
    }
}

fn trans_const(ccx: @crate_ctxt, _e: @ast::expr, id: ast::node_id) {
    unsafe {
        let _icx = ccx.insn_ctxt("trans_const");
        let g = base::get_item_val(ccx, id);
        // At this point, get_item_val has already translated the
        // constant's initializer to determine its LLVM type.
        let v = ccx.const_values.get(id);
        llvm::LLVMSetInitializer(g, v);
        llvm::LLVMSetGlobalConstant(g, True);
    }
}
