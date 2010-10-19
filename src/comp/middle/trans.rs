import std._str;
import std._vec;
import std._str.rustrt.sbuf;
import std._vec.rustrt.vbuf;
import std.map.hashmap;
import std.util.option;
import std.util.some;
import std.util.none;

import front.ast;
import driver.session;
import back.x86;
import back.abi;

import util.common.istr;
import util.common.new_str_hash;

import lib.llvm.llvm;
import lib.llvm.builder;
import lib.llvm.llvm.ModuleRef;
import lib.llvm.llvm.ValueRef;
import lib.llvm.llvm.TypeRef;
import lib.llvm.llvm.BuilderRef;
import lib.llvm.llvm.BasicBlockRef;

import lib.llvm.False;
import lib.llvm.True;

state obj namegen(mutable int i) {
    fn next(str prefix) -> str {
        i += 1;
        ret prefix + istr(i);
    }
}

type glue_fns = rec(ValueRef activate_glue,
                    ValueRef yield_glue,
                    ValueRef exit_task_glue,
                    vec[ValueRef] upcall_glues);

state type trans_ctxt = rec(session.session sess,
                            ModuleRef llmod,
                            hashmap[str,ValueRef] upcalls,
                            hashmap[str,ValueRef] fns,
                            @glue_fns glues,
                            namegen names,
                            str path);

state type fn_ctxt = rec(ValueRef llfn,
                         ValueRef lloutptr,
                         ValueRef lltaskptr,
                         @trans_ctxt tcx);

type terminator = fn(@fn_ctxt cx, builder build);

tag cleanup {
    clean(fn(@block_ctxt cx) -> result);
}

state type block_ctxt = rec(BasicBlockRef llbb,
                            builder build,
                            terminator term,
                            mutable vec[cleanup] cleanups,
                            @fn_ctxt fcx);


state type result = rec(mutable @block_ctxt bcx,
                        mutable ValueRef val);

fn res(@block_ctxt bcx, ValueRef val) -> result {
    ret rec(mutable bcx = bcx,
            mutable val = val);
}

fn ty_str(TypeRef t) -> str {
    ret lib.llvm.type_to_str(t);
}

fn val_ty(ValueRef v) -> TypeRef {
    ret llvm.LLVMTypeOf(v);
}

fn val_str(ValueRef v) -> str {
    ret ty_str(val_ty(v));
}


// LLVM type constructors.

fn T_void() -> TypeRef {
    // Note: For the time being llvm is kinda busted here, it has the notion
    // of a 'void' type that can only occur as part of the signature of a
    // function, but no general unit type of 0-sized value. This is, afaict,
    // vestigial from its C heritage, and we'll be attempting to submit a
    // patch upstream to fix it. In the mean time we only model function
    // outputs (Rust functions and C functions) using T_void, and model the
    // Rust general purpose nil type you can construct as 1-bit (always
    // zero). This makes the result incorrect for now -- things like a tuple
    // of 10 nil values will have 10-bit size -- but it doesn't seem like we
    // have any other options until it's fixed upstream.
    ret llvm.LLVMVoidType();
}

fn T_nil() -> TypeRef {
    // NB: See above in T_void().
    ret llvm.LLVMInt1Type();
}

fn T_i1() -> TypeRef {
    ret llvm.LLVMInt1Type();
}

fn T_i8() -> TypeRef {
    ret llvm.LLVMInt8Type();
}

fn T_i16() -> TypeRef {
    ret llvm.LLVMInt16Type();
}

fn T_i32() -> TypeRef {
    ret llvm.LLVMInt32Type();
}

fn T_i64() -> TypeRef {
    ret llvm.LLVMInt64Type();
}

fn T_int() -> TypeRef {
    // FIXME: switch on target type.
    ret T_i32();
}

fn T_fn(vec[TypeRef] inputs, TypeRef output) -> TypeRef {
    ret llvm.LLVMFunctionType(output,
                              _vec.buf[TypeRef](inputs),
                              _vec.len[TypeRef](inputs),
                              False);
}

fn T_ptr(TypeRef t) -> TypeRef {
    ret llvm.LLVMPointerType(t, 0u);
}

fn T_struct(vec[TypeRef] elts) -> TypeRef {
    ret llvm.LLVMStructType(_vec.buf[TypeRef](elts),
                            _vec.len[TypeRef](elts),
                            False);
}

fn T_opaque() -> TypeRef {
    ret llvm.LLVMOpaqueType();
}

fn T_task() -> TypeRef {
    ret T_struct(vec(T_int(),      // Refcount
                     T_int(),      // Delegate pointer
                     T_int(),      // Stack segment pointer
                     T_int(),      // Runtime SP
                     T_int(),      // Rust SP
                     T_int(),      // GC chain
                     T_int(),      // Domain pointer
                     T_int()       // Crate cache pointer
                     ));
}

fn T_array(TypeRef t, uint n) -> TypeRef {
    ret llvm.LLVMArrayType(t, n);
}

fn T_vec(TypeRef t, uint n) -> TypeRef {
    ret T_struct(vec(T_int(),      // Refcount
                     T_int(),      // Alloc
                     T_int(),      // Fill
                     T_array(t, n) // Body elements
                     ));
}

fn T_str(uint n) -> TypeRef {
    ret T_vec(T_i8(), n);
}

fn T_crate() -> TypeRef {
    ret T_struct(vec(T_int(),      // ptrdiff_t image_base_off
                     T_int(),      // uintptr_t self_addr
                     T_int(),      // ptrdiff_t debug_abbrev_off
                     T_int(),      // size_t debug_abbrev_sz
                     T_int(),      // ptrdiff_t debug_info_off
                     T_int(),      // size_t debug_info_sz
                     T_int(),      // size_t activate_glue_off
                     T_int(),      // size_t yield_glue_off
                     T_int(),      // size_t unwind_glue_off
                     T_int(),      // size_t gc_glue_off
                     T_int(),      // size_t main_exit_task_glue_off
                     T_int(),      // int n_rust_syms
                     T_int(),      // int n_c_syms
                     T_int()       // int n_libs
                     ));
}

fn T_double() -> TypeRef {
    ret llvm.LLVMDoubleType();
}

fn T_taskptr() -> TypeRef {
    ret T_ptr(T_task());
}

// LLVM constant constructors.

fn C_null(TypeRef t) -> ValueRef {
    ret llvm.LLVMConstNull(t);
}

fn C_integral(int i, TypeRef t) -> ValueRef {
    // FIXME. We can't use LLVM.ULongLong with our existing minimal native
    // API, which only knows word-sized args.  Lucky for us LLVM has a "take a
    // string encoding" version.  Hilarious. Please fix to handle:
    //
    // ret llvm.LLVMConstInt(T_int(), t as LLVM.ULongLong, False);
    //
    ret llvm.LLVMConstIntOfString(t, _str.buf(istr(i)), 10);
}

fn C_nil() -> ValueRef {
    // NB: See comment above in T_void().
    ret C_integral(0, T_i1());
}

fn C_bool(bool b) -> ValueRef {
    if (b) {
        ret C_integral(1, T_i1());
    } else {
        ret C_integral(0, T_i1());
    }
}

fn C_int(int i) -> ValueRef {
    ret C_integral(i, T_int());
}

fn C_str(@trans_ctxt cx, str s) -> ValueRef {
    auto sc = llvm.LLVMConstString(_str.buf(s), _str.byte_len(s), False);
    auto g = llvm.LLVMAddGlobal(cx.llmod, val_ty(sc),
                                _str.buf(cx.names.next("str")));
    llvm.LLVMSetInitializer(g, sc);
    ret g;
}

fn C_struct(vec[ValueRef] elts) -> ValueRef {
    ret llvm.LLVMConstStruct(_vec.buf[ValueRef](elts),
                             _vec.len[ValueRef](elts),
                             False);
}

fn decl_cdecl_fn(ModuleRef llmod, str name,
                 vec[TypeRef] inputs, TypeRef output) -> ValueRef {
    let TypeRef llty = T_fn(inputs, output);
    log "declaring " + name + " with type " + ty_str(llty);
    let ValueRef llfn =
        llvm.LLVMAddFunction(llmod, _str.buf(name), llty);
    llvm.LLVMSetFunctionCallConv(llfn, lib.llvm.LLVMCCallConv);
    ret llfn;
}

fn decl_glue(ModuleRef llmod, str s) -> ValueRef {
    ret decl_cdecl_fn(llmod, s, vec(T_taskptr()), T_void());
}

fn decl_upcall(ModuleRef llmod, uint _n) -> ValueRef {
    // It doesn't actually matter what type we come up with here, at the
    // moment, as we cast the upcall function pointers to int before passing
    // them to the indirect upcall-invocation glue.  But eventually we'd like
    // to call them directly, once we have a calling convention worked out.
    let int n = _n as int;
    let str s = abi.upcall_glue_name(n);
    let vec[TypeRef] args =
        vec(T_taskptr(), // taskptr
            T_int())     // callee
        + _vec.init_elt[TypeRef](T_int(), n as uint);

    ret decl_cdecl_fn(llmod, s, args, T_int());
}

fn get_upcall(@trans_ctxt cx, str name, int n_args) -> ValueRef {
    if (cx.upcalls.contains_key(name)) {
        ret cx.upcalls.get(name);
    }
    auto inputs = vec(T_taskptr());
    inputs += _vec.init_elt[TypeRef](T_int(), n_args as uint);
    auto output = T_int();
    auto f = decl_cdecl_fn(cx.llmod, name, inputs, output);
    cx.upcalls.insert(name, f);
    ret f;
}

fn trans_upcall(@block_ctxt cx, str name, vec[ValueRef] args) -> result {
    let int n = _vec.len[ValueRef](args) as int;
    let ValueRef llupcall = get_upcall(cx.fcx.tcx, name, n);
    llupcall = llvm.LLVMConstPointerCast(llupcall, T_int());

    let ValueRef llglue = cx.fcx.tcx.glues.upcall_glues.(n);
    let vec[ValueRef] call_args = vec(cx.fcx.lltaskptr, llupcall);
    for (ValueRef a in args) {
        call_args += cx.build.ZExtOrBitCast(a, T_int());
    }
    /*
     log "emitting indirect-upcall via " + abi.upcall_glue_name(n);
     for (ValueRef v in call_args) {
       log "arg: " + val_str(v);
     }
     log "emitting call to llglue of type: " + val_str(llglue);
    */

    ret res(cx, cx.build.Call(llglue, call_args));
}

fn trans_non_gc_free(@block_ctxt cx, ValueRef v) -> result {
    ret trans_upcall(cx, "upcall_free", vec(cx.build.PtrToInt(v, T_int()),
                                            C_int(0)));
}

fn decr_refcnt_and_if_zero(@block_ctxt cx,
                           ValueRef box_ptr,
                           fn(@block_ctxt cx) -> result inner,
                           TypeRef t_else, ValueRef v_else) -> result {
    auto rc_ptr = cx.build.GEP(box_ptr, vec(C_int(0),
                                            C_int(abi.box_rc_field_refcnt)));
    auto rc = cx.build.Load(rc_ptr);
    rc = cx.build.Sub(rc, C_int(1));
    cx.build.Store(rc, rc_ptr);
    auto test = cx.build.ICmp(lib.llvm.LLVMIntEQ, C_int(0), rc);
    auto next_cx = new_extension_block_ctxt(cx);
    auto then_cx = new_empty_block_ctxt(cx.fcx);
    auto then_res = inner(then_cx);
    then_res.bcx.build.Br(next_cx.llbb);
    cx.build.CondBr(test, then_res.bcx.llbb, next_cx.llbb);
    auto phi = next_cx.build.Phi(t_else,
                                 vec(v_else, then_res.val),
                                 vec(cx.llbb, then_res.bcx.llbb));
    ret res(next_cx, phi);
}

fn trans_drop_str(@block_ctxt cx, ValueRef v) -> result {
    ret decr_refcnt_and_if_zero(cx, v,
                                bind trans_non_gc_free(_, v),
                                T_int(), C_int(0));
}

fn trans_lit(@block_ctxt cx, &ast.lit lit) -> result {
    alt (lit.node) {
        case (ast.lit_int(?i)) {
            ret res(cx, C_int(i));
        }
        case (ast.lit_uint(?u)) {
            ret res(cx, C_int(u as int));
        }
        case (ast.lit_char(?c)) {
            ret res(cx, C_integral(c as int, T_i32()));
        }
        case (ast.lit_bool(?b)) {
            ret res(cx, C_bool(b));
        }
        case (ast.lit_nil) {
            ret res(cx, C_nil());
        }
        case (ast.lit_str(?s)) {
            auto len = (_str.byte_len(s) as int) + 1;
            auto sub = trans_upcall(cx, "upcall_new_str",
                                    vec(p2i(C_str(cx.fcx.tcx, s)),
                                        C_int(len)));
            sub.val = sub.bcx.build.IntToPtr(sub.val,
                                             T_ptr(T_str(len as uint)));
            cx.cleanups += vec(clean(bind trans_drop_str(_, sub.val)));
            ret sub;
        }
    }
}

fn trans_unary(@block_ctxt cx, ast.unop op, &ast.expr e) -> result {

    auto sub = trans_expr(cx, e);

    alt (op) {
        case (ast.bitnot) {
            sub.val = cx.build.Not(sub.val);
            ret sub;
        }
        case (ast.not) {
            sub.val = cx.build.Not(sub.val);
            ret sub;
        }
        case (ast.neg) {
            // FIXME: switch by signedness.
            sub.val = cx.build.Neg(sub.val);
            ret sub;
        }
    }
    cx.fcx.tcx.sess.unimpl("expr variant in trans_unary");
    fail;
}

fn trans_binary(@block_ctxt cx, ast.binop op,
                &ast.expr a, &ast.expr b) -> result {

    auto lhs = trans_expr(cx, a);
    auto sub = trans_expr(lhs.bcx, b);

    alt (op) {
        case (ast.add) {
            sub.val = cx.build.Add(lhs.val, sub.val);
            ret sub;
        }

        case (ast.sub) {
            sub.val = cx.build.Sub(lhs.val, sub.val);
            ret sub;
        }

        case (ast.mul) {
            // FIXME: switch by signedness.
            sub.val = cx.build.Mul(lhs.val, sub.val);
            ret sub;
        }

        case (ast.div) {
            // FIXME: switch by signedness.
            sub.val = cx.build.SDiv(lhs.val, sub.val);
            ret sub;
        }

        case (ast.rem) {
            // FIXME: switch by signedness.
            sub.val = cx.build.SRem(lhs.val, sub.val);
            ret sub;
        }

        case (ast.bitor) {
            sub.val = cx.build.Or(lhs.val, sub.val);
            ret sub;
        }

        case (ast.bitand) {
            sub.val = cx.build.And(lhs.val, sub.val);
            ret sub;
        }

        case (ast.bitxor) {
            sub.val = cx.build.Xor(lhs.val, sub.val);
            ret sub;
        }

        case (ast.lsl) {
            sub.val = cx.build.Shl(lhs.val, sub.val);
            ret sub;
        }

        case (ast.lsr) {
            sub.val = cx.build.LShr(lhs.val, sub.val);
            ret sub;
        }

        case (ast.asr) {
            sub.val = cx.build.AShr(lhs.val, sub.val);
            ret sub;
        }

        case (ast.eq) {
            sub.val = cx.build.ICmp(lib.llvm.LLVMIntEQ, lhs.val, sub.val);
            ret sub;
        }

        case (ast.ne) {
            sub.val = cx.build.ICmp(lib.llvm.LLVMIntNE, lhs.val, sub.val);
            ret sub;
        }

        case (ast.lt) {
            // FIXME: switch by signedness.
            sub.val = cx.build.ICmp(lib.llvm.LLVMIntSLT, lhs.val, sub.val);
            ret sub;
        }

        case (ast.le) {
            // FIXME: switch by signedness.
            sub.val = cx.build.ICmp(lib.llvm.LLVMIntSLE, lhs.val, sub.val);
            ret sub;
        }

        case (ast.ge) {
            // FIXME: switch by signedness.
            sub.val = cx.build.ICmp(lib.llvm.LLVMIntSGE, lhs.val, sub.val);
            ret sub;
        }

        case (ast.gt) {
            // FIXME: switch by signedness.
            sub.val = cx.build.ICmp(lib.llvm.LLVMIntSGT, lhs.val, sub.val);
            ret sub;
        }
    }
    cx.fcx.tcx.sess.unimpl("expr variant in trans_binary");
    fail;
}

fn trans_if(@block_ctxt cx, &ast.expr cond,
            &ast.block thn, &option[ast.block] els) -> result {

    auto cond_res = trans_expr(cx, cond);

    auto then_cx = new_empty_block_ctxt(cx.fcx);
    auto then_res = trans_block(then_cx, thn);

    auto next_cx = new_extension_block_ctxt(cx);
    then_res.bcx.build.Br(next_cx.llbb);
    auto phi;

    alt (els) {
        case (some[ast.block](?eblk)) {
            auto else_cx = new_empty_block_ctxt(cx.fcx);
            auto else_res = trans_block(else_cx, eblk);
            cond_res.bcx.build.CondBr(cond_res.val,
                                      then_cx.llbb,
                                      else_cx.llbb);
            else_res.bcx.build.Br(next_cx.llbb);
            phi = next_cx.build.Phi(T_nil(),
                                    vec(then_res.val,
                                        else_res.val),
                                    vec(then_res.bcx.llbb,
                                        else_res.bcx.llbb));
        }

        case (_) {
            cond_res.bcx.build.CondBr(cond_res.val,
                                      then_cx.llbb,
                                      next_cx.llbb);
            phi = next_cx.build.Phi(T_nil(),
                                    vec(then_res.val, C_nil()),
                                    vec(then_res.bcx.llbb,
                                        cond_res.bcx.llbb));
        }
    }

    ret res(next_cx, phi);
}

fn trans_expr(@block_ctxt cx, &ast.expr e) -> result {
    alt (e.node) {
        case (ast.expr_lit(?lit, _)) {
            ret trans_lit(cx, *lit);
        }

        case (ast.expr_unary(?op, ?x, _)) {
            ret trans_unary(cx, op, *x);
        }

        case (ast.expr_binary(?op, ?x, ?y, _)) {
            ret trans_binary(cx, op, *x, *y);
        }

        case (ast.expr_if(?cond, ?thn, ?els, _)) {
            ret trans_if(cx, *cond, thn, els);
        }

        case (ast.expr_block(?blk, _)) {
            auto sub_cx = new_empty_block_ctxt(cx.fcx);
            auto next_cx = new_extension_block_ctxt(cx);
            auto sub = trans_block(sub_cx, blk);

            cx.build.Br(sub_cx.llbb);
            sub.bcx.build.Br(next_cx.llbb);

            ret res(next_cx, sub.val);
        }
    }
    cx.fcx.tcx.sess.unimpl("expr variant in trans_expr");
    fail;
}

fn trans_log(@block_ctxt cx, &ast.expr e) -> result {
    alt (e.node) {
        case (ast.expr_lit(?lit, _)) {
            alt (lit.node) {
                case (ast.lit_str(_)) {
                    auto sub = trans_expr(cx, e);
                    auto v = sub.bcx.build.PtrToInt(sub.val, T_int());
                    ret trans_upcall(sub.bcx,
                                     "upcall_log_str",
                                     vec(v));
                }

                case (_) {
                    auto sub = trans_expr(cx, e);
                    ret trans_upcall(sub.bcx,
                                     "upcall_log_int",
                                     vec(sub.val));
                }
            }
        }

        case (_) {
            auto sub = trans_expr(cx, e);
            ret trans_upcall(sub.bcx, "upcall_log_int", vec(sub.val));
        }
    }
}

fn trans_stmt(@block_ctxt cx, &ast.stmt s) -> result {
    auto sub = res(cx, C_nil());
    alt (s.node) {
        case (ast.stmt_log(?a)) {
            sub.bcx = trans_log(cx, *a).bcx;
        }

        case (ast.stmt_expr(?e)) {
            sub.bcx = trans_expr(cx, *e).bcx;
        }

        case (_) {
            cx.fcx.tcx.sess.unimpl("stmt variant");
        }
    }
    ret sub;
}

fn new_builder(BasicBlockRef llbb) -> builder {
    let BuilderRef llbuild = llvm.LLVMCreateBuilder();
    llvm.LLVMPositionBuilderAtEnd(llbuild, llbb);
    ret builder(llbuild);
}

// You probably don't want to use this one. See the
// next three functions instead.
fn new_block_ctxt(@fn_ctxt cx, terminator term,
                  vec[cleanup] cleanups) -> @block_ctxt {
    let BasicBlockRef llbb =
        llvm.LLVMAppendBasicBlock(cx.llfn, _str.buf(""));
    ret @rec(llbb=llbb,
             build=new_builder(llbb),
             term=term,
             mutable cleanups=cleanups,
             fcx=cx);
}

// Use this when you are making a block_ctxt to replace the
// current one, i.e. when chaining together sequences of stmts
// or making sub-blocks you will branch back out of and wish to
// "carry on" in the parent block's context.
fn new_extension_block_ctxt(@block_ctxt bcx) -> @block_ctxt {
    ret new_block_ctxt(bcx.fcx, bcx.term, bcx.cleanups);
}

// Use this when you're at the top block of a function or the like.
fn new_top_block_ctxt(@fn_ctxt fcx) -> @block_ctxt {
    fn terminate_ret_void(@fn_ctxt cx, builder build) {
        build.RetVoid();
    }
    auto term = terminate_ret_void;
    let vec[cleanup] cleanups = vec();
    ret new_block_ctxt(fcx, term, cleanups);

}

// Use this when you are making a block_ctxt to replace the
// current one, i.e. when chaining together sequences of stmts
// or making sub-blocks you will branch back out of and wish to
// "carry on" in the parent block's context.
fn new_empty_block_ctxt(@fn_ctxt fcx) -> @block_ctxt {
    fn terminate_no_op(@fn_ctxt cx, builder build) {
    }
    auto term = terminate_no_op;
    let vec[cleanup] cleanups = vec();
    ret new_block_ctxt(fcx, term, cleanups);
}

fn trans_block_cleanups(@block_ctxt cx) -> @block_ctxt {
    auto bcx = cx;
    for (cleanup c in cx.cleanups) {
        alt (c) {
            case (clean(?cfn)) {
                bcx = cfn(bcx).bcx;
            }
        }
    }
    ret bcx;
}

fn trans_block(@block_ctxt cx, &ast.block b) -> result {
    auto bcx = cx;

    for (@ast.stmt s in b.node.stmts) {
        bcx = trans_stmt(bcx, *s).bcx;
    }

    bcx = trans_block_cleanups(bcx);
    bcx.term(bcx.fcx, bcx.build);
    ret res(bcx, C_nil());
}

fn new_fn_ctxt(@trans_ctxt cx,
               str name,
               TypeRef T_out,
               vec[TypeRef] T_explicit_args) -> @fn_ctxt {
    let vec[TypeRef] args = vec(T_ptr(T_out), // outptr.
                                T_taskptr()   // taskptr
                                );
    args += T_explicit_args;
    let ValueRef llfn = decl_cdecl_fn(cx.llmod, name, args, T_void());
    cx.fns.insert(cx.path, llfn);
    let ValueRef lloutptr = llvm.LLVMGetParam(llfn, 0u);
    let ValueRef lltaskptr = llvm.LLVMGetParam(llfn, 1u);
    ret @rec(llfn=llfn,
             lloutptr=lloutptr,
             lltaskptr=lltaskptr,
             tcx=cx);
}

fn trans_fn(@trans_ctxt cx, &ast._fn f) {
    let TypeRef out = T_int();
    let vec[TypeRef] args = vec();

    auto fcx = new_fn_ctxt(cx, cx.path, out, args);

    trans_block(new_top_block_ctxt(fcx), f.body);
}

fn trans_item(@trans_ctxt cx, &ast.item item) {
    alt (item.node) {
        case (ast.item_fn(?name, ?f, _)) {
            auto sub_cx = @rec(path=cx.path + "." + name with *cx);
            trans_fn(sub_cx, f);
        }
        case (ast.item_mod(?name, ?m, _)) {
            auto sub_cx = @rec(path=cx.path + "." + name with *cx);
            trans_mod(sub_cx, m);
        }
    }
}

fn trans_mod(@trans_ctxt cx, &ast._mod m) {
    for (@ast.item item in m.items) {
        trans_item(cx, *item);
    }
}


fn p2i(ValueRef v) -> ValueRef {
    ret llvm.LLVMConstPtrToInt(v, T_int());
}

fn trans_exit_task_glue(@trans_ctxt cx) {
    let vec[TypeRef] T_args = vec();
    let vec[ValueRef] V_args = vec();

    auto llfn = cx.glues.exit_task_glue;
    let ValueRef lloutptr = C_null(T_int());
    let ValueRef lltaskptr = llvm.LLVMGetParam(llfn, 0u);
    auto fcx = @rec(llfn=llfn,
                    lloutptr=lloutptr,
                    lltaskptr=lltaskptr,
                    tcx=cx);

    auto bcx = new_top_block_ctxt(fcx);
    trans_upcall(bcx, "upcall_exit", V_args);
    bcx.term(fcx, bcx.build);
}

fn crate_constant(@trans_ctxt cx) -> ValueRef {

    let ValueRef crate_ptr =
        llvm.LLVMAddGlobal(cx.llmod, T_crate(),
                           _str.buf("rust_crate"));

    let ValueRef crate_addr = p2i(crate_ptr);

    let ValueRef activate_glue_off =
        llvm.LLVMConstSub(p2i(cx.glues.activate_glue), crate_addr);

    let ValueRef yield_glue_off =
        llvm.LLVMConstSub(p2i(cx.glues.yield_glue), crate_addr);

    let ValueRef exit_task_glue_off =
        llvm.LLVMConstSub(p2i(cx.glues.exit_task_glue), crate_addr);

    let ValueRef crate_val =
        C_struct(vec(C_null(T_int()),     // ptrdiff_t image_base_off
                     p2i(crate_ptr),      // uintptr_t self_addr
                     C_null(T_int()),     // ptrdiff_t debug_abbrev_off
                     C_null(T_int()),     // size_t debug_abbrev_sz
                     C_null(T_int()),     // ptrdiff_t debug_info_off
                     C_null(T_int()),     // size_t debug_info_sz
                     activate_glue_off,   // size_t activate_glue_off
                     yield_glue_off,      // size_t yield_glue_off
                     C_null(T_int()),     // size_t unwind_glue_off
                     C_null(T_int()),     // size_t gc_glue_off
                     exit_task_glue_off,  // size_t main_exit_task_glue_off
                     C_null(T_int()),     // int n_rust_syms
                     C_null(T_int()),     // int n_c_syms
                     C_null(T_int())      // int n_libs
                     ));

    llvm.LLVMSetInitializer(crate_ptr, crate_val);
    ret crate_ptr;
}

fn trans_main_fn(@trans_ctxt cx, ValueRef llcrate) {
    auto T_main_args = vec(T_int(), T_int());
    auto T_rust_start_args = vec(T_int(), T_int(), T_int(), T_int());

    auto llmain =
        decl_cdecl_fn(cx.llmod, "main", T_main_args, T_int());

    auto llrust_start =
        decl_cdecl_fn(cx.llmod, "rust_start", T_rust_start_args, T_int());

    auto llargc = llvm.LLVMGetParam(llmain, 0u);
    auto llargv = llvm.LLVMGetParam(llmain, 1u);
    auto llrust_main = cx.fns.get("_rust.main");

    //
    // Emit the moral equivalent of:
    //
    // main(int argc, char **argv) {
    //     rust_start(&_rust.main, &crate, argc, argv);
    // }
    //

    let BasicBlockRef llbb =
        llvm.LLVMAppendBasicBlock(llmain, _str.buf(""));
    auto b = new_builder(llbb);

    auto start_args = vec(p2i(llrust_main), p2i(llcrate), llargc, llargv);

    b.Ret(b.Call(llrust_start, start_args));

}

fn trans_crate(session.session sess, @ast.crate crate) {
    auto llmod =
        llvm.LLVMModuleCreateWithNameInContext(_str.buf("rust_out"),
                                               llvm.LLVMGetGlobalContext());

    llvm.LLVMSetModuleInlineAsm(llmod, _str.buf(x86.get_module_asm()));

    auto glues = @rec(activate_glue = decl_glue(llmod,
                                                abi.activate_glue_name()),
                      yield_glue = decl_glue(llmod, abi.yield_glue_name()),
                      /*
                       * Note: the signature passed to decl_cdecl_fn here
                       * looks unusual because it is. It corresponds neither
                       * to an upcall signature nor a normal rust-ABI
                       * signature. In fact it is a fake signature, that
                       * exists solely to acquire the task pointer as an
                       * argument to the upcall. It so happens that the
                       * runtime sets up the task pointer as the sole incoming
                       * argument to the frame that we return into when
                       * returning to the exit task glue. So this is the
                       * signature required to retrieve it.
                       */
                      exit_task_glue =
                      decl_cdecl_fn(llmod, abi.exit_task_glue_name(),
                                    vec(T_taskptr()), T_void()),

                      upcall_glues =
                      _vec.init_fn[ValueRef](bind decl_upcall(llmod, _),
                                             abi.n_upcall_glues as uint));

    auto cx = @rec(sess = sess,
                   llmod = llmod,
                   upcalls = new_str_hash[ValueRef](),
                   fns = new_str_hash[ValueRef](),
                   glues = glues,
                   names = namegen(0),
                   path = "_rust");

    trans_mod(cx, crate.node.module);
    trans_exit_task_glue(cx);
    trans_main_fn(cx, crate_constant(cx));

    llvm.LLVMWriteBitcodeToFile(llmod, _str.buf("rust_out.bc"));
    llvm.LLVMDisposeModule(llmod);
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
