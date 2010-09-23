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

fn trans_fn(ModuleRef llmod, str name, &ast._fn f) {
    let vec[TypeRef] args = vec();
    let TypeRef llty = T_fn(args, T_nil());
    let ValueRef llfn =
        llvm.LLVMAddFunction(llmod, _str.buf(name), llty);
}

fn trans_block(ast.block b, ValueRef llfn) {
    let BasicBlockRef llbb =
        llvm.LLVMAppendBasicBlock(llfn, 0 as sbuf);
    let BuilderRef llbuild = llvm.LLVMCreateBuilder();
    llvm.LLVMPositionBuilderAtEnd(llbuild, llbb);
    auto b = builder(llbuild);
}

fn trans_mod_item(ModuleRef llmod, str name, &ast.item item) {
    alt (item) {
        case (ast.item_fn(?f)) {
            trans_fn(llmod, name, *f);
        }
        case (ast.item_mod(?m)) {
            trans_mod(llmod, name, *m);
        }
    }
}

fn trans_mod(ModuleRef llmod, str name, &ast._mod m) {
    for each (tup(str, ast.item) pair in m.items()) {
        trans_mod_item(llmod, name + "." + pair._0, pair._1);
    }
}

fn trans_crate(session.session sess, ast.crate crate) {
    auto llmod =
        llvm.LLVMModuleCreateWithNameInContext(_str.buf("rust_out"),
                                               llvm.LLVMGetGlobalContext());


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
