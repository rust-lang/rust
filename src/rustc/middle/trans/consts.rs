import common::*;
import syntax::{ast, ast_util, codemap, ast_map};
import base::get_insn_ctxt;

fn const_lit(cx: @crate_ctxt, e: @ast::expr, lit: ast::lit)
    -> ValueRef {
    let _icx = cx.insn_ctxt(~"trans_lit");
    alt lit.node {
      ast::lit_int(i, t) { C_integral(T_int_ty(cx, t), i as u64, True) }
      ast::lit_uint(u, t) { C_integral(T_uint_ty(cx, t), u, False) }
      ast::lit_int_unsuffixed(i) {
        let lit_int_ty = ty::node_id_to_type(cx.tcx, e.id);
        alt ty::get(lit_int_ty).struct {
          ty::ty_int(t) {
            C_integral(T_int_ty(cx, t), i as u64, True)
          }
          ty::ty_uint(t) {
            C_integral(T_uint_ty(cx, t), i as u64, False)
          }
          _ { cx.sess.span_bug(lit.span,
                               ~"integer literal doesn't have a type");
            }
        }
      }
      ast::lit_float(fs, t) { C_floating(*fs, T_float_ty(cx, t)) }
      ast::lit_bool(b) { C_bool(b) }
      ast::lit_nil { C_nil() }
      ast::lit_str(s) {
        cx.sess.span_unimpl(lit.span, ~"unique string in this context");
      }
    }
}

// FIXME (#2530): this should do some structural hash-consing to avoid
// duplicate constants. I think. Maybe LLVM has a magical mode that does so
// later on?
fn const_expr(cx: @crate_ctxt, e: @ast::expr) -> ValueRef {
    let _icx = cx.insn_ctxt(~"const_expr");
    alt e.node {
      ast::expr_lit(lit) { consts::const_lit(cx, e, *lit) }
      // If we have a vstore, just keep going; it has to be a string
      ast::expr_vstore(e, _) { const_expr(cx, e) }
      ast::expr_binary(b, e1, e2) {
        let te1 = const_expr(cx, e1);
        let te2 = const_expr(cx, e2);

        let te2 = base::cast_shift_const_rhs(b, te1, te2);

        /* Neither type is bottom, and we expect them to be unified already,
         * so the following is safe. */
        let ty = ty::expr_ty(cx.tcx, e1);
        let is_float = ty::type_is_fp(ty);
        let signed = ty::type_is_signed(ty);
        ret alt b {
          ast::add    {
            if is_float { llvm::LLVMConstFAdd(te1, te2) }
            else        { llvm::LLVMConstAdd(te1, te2) }
          }
          ast::subtract {
            if is_float { llvm::LLVMConstFSub(te1, te2) }
            else        { llvm::LLVMConstSub(te1, te2) }
          }
          ast::mul    {
            if is_float { llvm::LLVMConstFMul(te1, te2) }
            else        { llvm::LLVMConstMul(te1, te2) }
          }
          ast::div    {
            if is_float    { llvm::LLVMConstFDiv(te1, te2) }
            else if signed { llvm::LLVMConstSDiv(te1, te2) }
            else           { llvm::LLVMConstUDiv(te1, te2) }
          }
          ast::rem    {
            if is_float    { llvm::LLVMConstFRem(te1, te2) }
            else if signed { llvm::LLVMConstSRem(te1, te2) }
            else           { llvm::LLVMConstURem(te1, te2) }
          }
          ast::and    |
          ast::or     { cx.sess.span_unimpl(e.span, ~"binop logic"); }
          ast::bitxor { llvm::LLVMConstXor(te1, te2) }
          ast::bitand { llvm::LLVMConstAnd(te1, te2) }
          ast::bitor  { llvm::LLVMConstOr(te1, te2) }
          ast::shl    { llvm::LLVMConstShl(te1, te2) }
          ast::shr    {
            if signed { llvm::LLVMConstAShr(te1, te2) }
            else      { llvm::LLVMConstLShr(te1, te2) }
          }
          ast::eq     |
          ast::lt     |
          ast::le     |
          ast::ne     |
          ast::ge     |
          ast::gt     { cx.sess.span_unimpl(e.span, ~"binop comparator"); }
        }
      }
      ast::expr_unary(u, e) {
        let te = const_expr(cx, e);
        let ty = ty::expr_ty(cx.tcx, e);
        let is_float = ty::type_is_fp(ty);
        ret alt u {
          ast::box(_)  |
          ast::uniq(_) |
          ast::deref   { cx.sess.span_bug(e.span,
                           ~"bad unop type in const_expr"); }
          ast::not    { llvm::LLVMConstNot(te) }
          ast::neg    {
            if is_float { llvm::LLVMConstFNeg(te) }
            else        { llvm::LLVMConstNeg(te) }
          }
        }
      }
      ast::expr_cast(base, tp) {
        let ety = ty::expr_ty(cx.tcx, e), llty = type_of::type_of(cx, ety);
        let basety = ty::expr_ty(cx.tcx, base);
        let v = const_expr(cx, base);
        alt check (base::cast_type_kind(basety), base::cast_type_kind(ety)) {
          (base::cast_integral, base::cast_integral) {
            let s = if ty::type_is_signed(basety) { True } else { False };
            llvm::LLVMConstIntCast(v, llty, s)
          }
          (base::cast_integral, base::cast_float) {
            if ty::type_is_signed(basety) { llvm::LLVMConstSIToFP(v, llty) }
            else { llvm::LLVMConstUIToFP(v, llty) }
          }
          (base::cast_float, base::cast_float) {
            llvm::LLVMConstFPCast(v, llty)
          }
          (base::cast_float, base::cast_integral) {
            if ty::type_is_signed(ety) { llvm::LLVMConstFPToSI(v, llty) }
            else { llvm::LLVMConstFPToUI(v, llty) }
          }
        }
      }
      ast::expr_tup(es) {
        C_struct(es.map(|e| const_expr(cx, e)))
      }
      ast::expr_rec(fs, none) {
        C_struct(fs.map(|f| const_expr(cx, f.node.expr)))
      }
      ast::expr_path(path) {
        alt cx.tcx.def_map.find(e.id) {
          some(ast::def_const(def_id)) {
            // Don't know how to handle external consts
            assert ast_util::is_local(def_id);
            alt cx.tcx.items.get(def_id.node) {
              ast_map::node_item(@{
                node: ast::item_const(_, subexpr), _
              }, _) {
                // FIXME (#2530): Instead of recursing here to regenerate
                // the values for other constants, we should just look up
                // the already-defined value.
                const_expr(cx, subexpr)
              }
              _ {
                cx.sess.span_bug(e.span, ~"expected item");
              }
            }
          }
          _ { cx.sess.span_bug(e.span, ~"expected to find a const def") }
        }
      }
      _ { cx.sess.span_bug(e.span,
            ~"bad constant expression type in consts::const_expr"); }
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
