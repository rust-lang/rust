import std._str;
import std._vec;
import std._str.rustrt.sbuf;
import std._vec.rustrt.vbuf;

import fe.ast;
import driver.session;

import lib.llvm.llvm;
import lib.llvm.builder;
import lib.llvm.llvm.ModuleRef;
import lib.llvm.llvm.ValueRef;
import lib.llvm.llvm.TypeRef;
import lib.llvm.llvm.BuilderRef;
import lib.llvm.llvm.BasicBlockRef;

import lib.llvm.False;
import lib.llvm.True;

type trans_ctxt = rec(session.session sess,
                      ModuleRef llmod,
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
                              False());
}

fn trans_log(&trans_ctxt cx, builder b, &ast.atom a) {
}

fn trans_stmt(&trans_ctxt cx, builder b, &ast.stmt s) {
    alt (s) {
        case (ast.stmt_log(?a)) {
            trans_log(cx, b, *a);
        }
        case (_) {
            cx.sess.unimpl("stmt variant");
        }
    }
}

fn trans_block(&trans_ctxt cx, ValueRef llfn, &ast.block b) {
    let BasicBlockRef llbb =
        llvm.LLVMAppendBasicBlock(llfn, _str.buf(""));
    let BuilderRef llbuild = llvm.LLVMCreateBuilder();
    llvm.LLVMPositionBuilderAtEnd(llbuild, llbb);
    auto bld = builder(llbuild);
    for (@ast.stmt s in b) {
        trans_stmt(cx, bld, *s);
    }
}

fn trans_fn(&trans_ctxt cx, &ast._fn f) {
    let vec[TypeRef] args = vec();
    let TypeRef llty = T_fn(args, T_nil());
    let ValueRef llfn =
        llvm.LLVMAddFunction(cx.llmod, _str.buf(cx.path), llty);
    trans_block(cx, llfn, f.body);
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

    auto cx = rec(sess=sess, llmod=llmod, path="");
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
