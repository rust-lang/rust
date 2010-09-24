import std._str;
import std._vec;
import std._str.rustrt.sbuf;
import std._vec.rustrt.vbuf;

import front.ast;
import driver.session;
import back.x86;
import back.abi;

import util.common.istr;

import lib.llvm.llvm;
import lib.llvm.builder;
import lib.llvm.llvm.ModuleRef;
import lib.llvm.llvm.ValueRef;
import lib.llvm.llvm.TypeRef;
import lib.llvm.llvm.BuilderRef;
import lib.llvm.llvm.BasicBlockRef;

import lib.llvm.False;
import lib.llvm.True;

type glue_fns = rec(ValueRef activate_glue,
                    ValueRef yield_glue,
                    vec[ValueRef] upcall_glues);

type trans_ctxt = rec(session.session sess,
                      ModuleRef llmod,
                      @glue_fns glues,
                      str path);

fn T_nil() -> TypeRef {
    ret llvm.LLVMVoidType();
}

fn T_int() -> TypeRef {
    ret llvm.LLVMInt32Type();
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
                     T_opaque())); // Rest is opaque for now
}

fn decl_cdecl_fn(ModuleRef llmod, str name,
                 vec[TypeRef] inputs, TypeRef output) -> ValueRef {
    let TypeRef llty = T_fn(inputs, output);
    let ValueRef llfn =
        llvm.LLVMAddFunction(llmod, _str.buf(name), llty);
    llvm.LLVMSetFunctionCallConv(llfn, lib.llvm.LLVMCCallConv);
    ret llfn;
}

fn decl_glue(ModuleRef llmod, str s) -> ValueRef {
    ret decl_cdecl_fn(llmod, s, vec(T_ptr(T_task())), T_nil());
}

fn decl_upcall(ModuleRef llmod, uint _n) -> ValueRef {
    let int n = _n as int;
    let str s = "rust_upcall_" + istr(n);
    let vec[TypeRef] args =
        vec(T_ptr(T_task()), // taskptr
            T_int())         // callee
        + _vec.init_elt[TypeRef](T_int(), n as uint);

    ret decl_cdecl_fn(llmod, s, args, T_int());
}


type terminator = fn(&trans_ctxt cx, builder b);

fn trans_log(&trans_ctxt cx, builder b, &ast.atom a) {
}

fn trans_stmt(&trans_ctxt cx, builder b, &ast.stmt s, terminator t) {
    alt (s) {
        case (ast.stmt_log(?a)) {
            trans_log(cx, b, *a);
        }
        case (_) {
            cx.sess.unimpl("stmt variant");
        }
    }
}

fn default_terminate(&trans_ctxt cx, builder b) {
    b.RetVoid();
}

fn trans_block(&trans_ctxt cx, ValueRef llfn, &ast.block b, terminator t) {
    let BasicBlockRef llbb =
        llvm.LLVMAppendBasicBlock(llfn, _str.buf(""));
    let BuilderRef llbuild = llvm.LLVMCreateBuilder();
    llvm.LLVMPositionBuilderAtEnd(llbuild, llbb);
    auto bld = builder(llbuild);
    for (@ast.stmt s in b) {
        trans_stmt(cx, bld, *s, t);
    }
    t(cx, bld);
}

fn trans_fn(&trans_ctxt cx, &ast._fn f) {
    let vec[TypeRef] args = vec();
    let ValueRef llfn = decl_cdecl_fn(cx.llmod, cx.path, args, T_nil());
    auto term = default_terminate;

    trans_block(cx, llfn, f.body, term);
}

fn trans_item(&trans_ctxt cx, &str name, &ast.item item) {
    auto sub_cx = rec(path=cx.path + "." + name with cx);
    alt (item) {
        case (ast.item_fn(?f)) {
            trans_fn(sub_cx, *f);
        }
        case (ast.item_mod(?m)) {
            trans_mod(sub_cx, *m);
        }
    }
}

fn trans_mod(&trans_ctxt cx, &ast._mod m) {
    for each (tup(str, ast.item) pair in m.items()) {
        trans_item(cx, pair._0, pair._1);
    }
}

fn trans_crate(session.session sess, ast.crate crate) {
    auto llmod =
        llvm.LLVMModuleCreateWithNameInContext(_str.buf("rust_out"),
                                               llvm.LLVMGetGlobalContext());

    llvm.LLVMSetModuleInlineAsm(llmod, _str.buf(x86.get_module_asm()));

    auto glues = @rec(activate_glue = decl_glue(llmod, "rust_activate_glue"),
                      yield_glue = decl_glue(llmod, "rust_yield_glue"),
                      upcall_glues =
                      _vec.init_fn[ValueRef](bind decl_upcall(llmod, _),
                                             abi.n_upcall_glues as uint));

    auto cx = rec(sess=sess, llmod=llmod, glues=glues, path="");
    trans_mod(cx, crate.module);

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
