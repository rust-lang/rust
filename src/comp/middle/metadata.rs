import std._str;
import front.ast;
import middle.trans;
import back.x86;

import lib.llvm.llvm;
import lib.llvm.llvm.ValueRef;
import lib.llvm.False;

// Returns a Plain Old LLVM String.
fn C_postr(str s) -> ValueRef {
    ret llvm.LLVMConstString(_str.buf(s), _str.byte_len(s), False);
}

fn collect_meta_directives(@trans.crate_ctxt cx, @ast.crate crate)
        -> ValueRef {
    ret C_postr("Hello world!");    // TODO
}

fn write_metadata(@trans.crate_ctxt cx, @ast.crate crate) {
    auto llmeta = collect_meta_directives(cx, crate);

    auto llconst = trans.C_struct(vec(llmeta));
    auto llglobal = llvm.LLVMAddGlobal(cx.llmod, trans.val_ty(llconst),
                                       _str.buf("rust_metadata"));
    llvm.LLVMSetInitializer(llglobal, llconst);
    llvm.LLVMSetSection(llglobal, _str.buf(x86.get_meta_sect_name()));
}

