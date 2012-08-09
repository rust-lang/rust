import common::*;
import syntax::{ast, ast_util, codemap, ast_map};
import base::get_insn_ctxt;

fn const_lit(cx: @crate_ctxt, e: @ast::expr, lit: ast::lit)
    -> ValueRef {
    let _icx = cx.insn_ctxt(~"trans_lit");
    match lit.node {
      ast::lit_int(i, t) => C_integral(T_int_ty(cx, t), i as u64, True),
      ast::lit_uint(u, t) => C_integral(T_uint_ty(cx, t), u, False),
      ast::lit_int_unsuffixed(i) => {
        let lit_int_ty = ty::node_id_to_type(cx.tcx, e.id);
        match ty::get(lit_int_ty).struct {
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
      ast::lit_float(fs, t) => C_floating(*fs, T_float_ty(cx, t)),
      ast::lit_bool(b) => C_bool(b),
      ast::lit_nil => C_nil(),
      ast::lit_str(s) => C_estr_slice(cx, *s)
    }
}

// FIXME (#2530): this should do some structural hash-consing to avoid
// duplicate constants. I think. Maybe LLVM has a magical mode that does so
// later on?

fn const_vec(cx: @crate_ctxt, e: @ast::expr, es: &[@ast::expr])
    -> (ValueRef, ValueRef, TypeRef) {
    let vec_ty = ty::expr_ty(cx.tcx, e);
    let unit_ty = ty::sequence_element_type(cx.tcx, vec_ty);
    let llunitty = type_of::type_of(cx, unit_ty);
    let v = C_array(llunitty, es.map(|e| const_expr(cx, e)));
    let unit_sz = shape::llsize_of(cx, llunitty);
    let sz = llvm::LLVMConstMul(C_uint(cx, es.len()), unit_sz);
    return (v, sz, llunitty);
}

fn const_deref(v: ValueRef) -> ValueRef {
    assert llvm::LLVMIsGlobalConstant(v) == True;
    llvm::LLVMGetInitializer(v)
}

fn const_get_elt(v: ValueRef, u: uint) -> ValueRef {
    let u = u;
    llvm::LLVMConstExtractValue(v, ptr::addr_of(u), 1 as c_uint)
}

fn const_autoderef(ty: ty::t, v: ValueRef)
    -> (ty::t, ValueRef) {
    let mut t1 = ty;
    let mut v1 = v;
    loop {
        // Only rptrs can be autoderef'ed in a const context.
        match ty::get(ty).struct {
            ty::ty_rptr(_, mt) => {
                t1 = mt.ty;
                v1 = const_deref(v1);
            }
            _ => return (t1,v1)
        }
    }
}


fn const_expr(cx: @crate_ctxt, e: @ast::expr) -> ValueRef {
    let _icx = cx.insn_ctxt(~"const_expr");
    match e.node {
      ast::expr_lit(lit) => consts::const_lit(cx, e, *lit),
      ast::expr_binary(b, e1, e2) => {
        let te1 = const_expr(cx, e1);
        let te2 = const_expr(cx, e2);

        let te2 = base::cast_shift_const_rhs(b, te1, te2);

        /* Neither type is bottom, and we expect them to be unified already,
         * so the following is safe. */
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
          ast::deref  => const_deref(te),
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
          let (bt, bv) = const_autoderef(bt, bv);
          let fields = match ty::get(bt).struct {
              ty::ty_rec(fs) => fs,
              ty::ty_class(did, substs) =>
                  ty::class_items_as_mutable_fields(cx.tcx, did, substs),
              _ => cx.sess.span_bug(e.span,
                                    ~"field access on unknown type in const"),
          };
          let ix = field_idx_strict(cx.tcx, e.span, field, fields);
          const_get_elt(bv, ix)
      }

      ast::expr_index(base, index) => {
          let bt = ty::expr_ty(cx.tcx, base);
          let bv = const_expr(cx, base);
          let (bt, bv) = const_autoderef(bt, bv);
          let iv = match const_eval::eval_const_expr(cx.tcx, index) {
              const_eval::const_int(i) => i as u64,
              const_eval::const_uint(u) => u,
              _ => cx.sess.span_bug(index.span,
                                    ~"index is not an integer-constant \
                                      expression")
          };
          let (arr,len) = match ty::get(bt).struct {
              ty::ty_evec(_, vstore) | ty::ty_estr(vstore) =>
                  match vstore {
                  ty::vstore_fixed(u) =>
                      (bv, C_uint(cx, u)),

                  ty::vstore_slice(_) => {
                      let unit_ty = ty::sequence_element_type(cx.tcx, bt);
                      let llunitty = type_of::type_of(cx, unit_ty);
                      let unit_sz = shape::llsize_of(cx, llunitty);
                      (const_deref(const_get_elt(bv, 0)),
                       llvm::LLVMConstUDiv(const_get_elt(bv, 1),
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
          let len = llvm::LLVMConstIntGetZExtValue(len) as u64;
          let len = match ty::get(bt).struct {
              ty::ty_estr(*) => {assert len > 0; len - 1},
              _ => len
          };
          if iv >= len {
              // Better late than never for reporting this?
              cx.sess.span_err(e.span,
                               ~"const index-expr is out of bounds");
          }
          const_get_elt(arr, iv as uint)
      }
      ast::expr_cast(base, tp) => {
        let ety = ty::expr_ty(cx.tcx, e), llty = type_of::type_of(cx, ety);
        let basety = ty::expr_ty(cx.tcx, base);
        let v = const_expr(cx, base);
        match check (base::cast_type_kind(basety),
                     base::cast_type_kind(ety)) {

          (base::cast_integral, base::cast_integral) => {
            let s = if ty::type_is_signed(basety) { True } else { False };
            llvm::LLVMConstIntCast(v, llty, s)
          }
          (base::cast_integral, base::cast_float) => {
            if ty::type_is_signed(basety) { llvm::LLVMConstSIToFP(v, llty) }
            else { llvm::LLVMConstUIToFP(v, llty) }
          }
          (base::cast_float, base::cast_float) => {
            llvm::LLVMConstFPCast(v, llty)
          }
          (base::cast_float, base::cast_integral) => {
            if ty::type_is_signed(ety) { llvm::LLVMConstFPToSI(v, llty) }
            else { llvm::LLVMConstFPToUI(v, llty) }
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
        C_struct(es.map(|e| const_expr(cx, e)))
      }
      ast::expr_struct(_, fs, _) => {
          let ety = ty::expr_ty(cx.tcx, e);
          let llty = type_of::type_of(cx, ety);
          let class_fields =
              match ty::get(ety).struct {
              ty::ty_class(clsid, _) =>
                  ty::lookup_class_fields(cx.tcx, clsid),
              _ =>
                  cx.tcx.sess.span_bug(e.span,
                                       ~"didn't resolve to a struct")
          };
          let mut cs = ~[];
          for class_fields.each |class_field| {
              let mut found = false;
              for fs.each |field| {
                  if class_field.ident == field.node.ident  {
                      found = true;
                      vec::push(cs, const_expr(cx, field.node.expr));
                  }
              }
              if !found {
                  cx.tcx.sess.span_bug(e.span, ~"missing struct field");
              }
          }
          C_named_struct(llty, cs)
      }
      ast::expr_rec(fs, none) => {
        C_struct(fs.map(|f| const_expr(cx, f.node.expr)))
      }
      ast::expr_vec(es, m_imm) => {
        let (v, _, _) = const_vec(cx, e, es);
        v
      }
      ast::expr_vstore(e, ast::vstore_fixed(_)) => {
        const_expr(cx, e)
      }
      ast::expr_vstore(sub, ast::vstore_slice(_)) => {
        match sub.node {
          ast::expr_lit(lit) => {
            match lit.node {
              ast::lit_str(*) => { const_expr(cx, sub) }
              _ => { cx.sess.span_bug(e.span,
                                      ~"bad const-slice lit") }
            }
          }
          ast::expr_vec(es, m_imm) => {
            let (cv, sz, llunitty) = const_vec(cx, e, es);
            let llty = val_ty(cv);
            let gv = do str::as_c_str("const") |name| {
                llvm::LLVMAddGlobal(cx.llmod, llty, name)
            };
            llvm::LLVMSetInitializer(gv, cv);
            llvm::LLVMSetGlobalConstant(gv, True);
            let p = llvm::LLVMConstPointerCast(gv, T_ptr(llunitty));

            C_struct(~[p, sz])
          }
          _ => cx.sess.span_bug(e.span,
                                ~"bad const-slice expr")
        }
      }
      ast::expr_path(path) => {
        match cx.tcx.def_map.find(e.id) {
          some(ast::def_const(def_id)) => {
            // Don't know how to handle external consts
            assert ast_util::is_local(def_id);
            match cx.tcx.items.get(def_id.node) {
              ast_map::node_item(@{
                node: ast::item_const(_, subexpr), _
              }, _) => {
                // FIXME (#2530): Instead of recursing here to regenerate
                // the values for other constants, we should just look up
                // the already-defined value.
                const_expr(cx, subexpr)
              }
              _ => cx.sess.span_bug(e.span, ~"expected item")
            }
          }
          _ => cx.sess.span_bug(e.span, ~"expected to find a const def")
        }
      }
      _ => cx.sess.span_bug(e.span,
            ~"bad constant expression type in consts::const_expr")
    }
}

fn trans_const(ccx: @crate_ctxt, e: @ast::expr, id: ast::node_id) {
    let _icx = ccx.insn_ctxt(~"trans_const");
    let v = const_expr(ccx, e);

    // The scalars come back as 1st class LLVM vals
    // which we have to stick into global constants.
    let g = base::get_item_val(ccx, id);
    llvm::LLVMSetInitializer(g, v);
    llvm::LLVMSetGlobalConstant(g, True);
}
