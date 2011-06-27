
import std::str;
import middle::trans;
import front::ast::crate;
import back::x86;
import lib::llvm::llvm;
import lib::llvm::llvm::ValueRef;
import lib::llvm::False;

export write_metadata;

// Returns a Plain Old LLVM String:
fn C_postr(&str s) -> ValueRef {
    ret llvm::LLVMConstString(str::buf(s), str::byte_len(s), False);
}

fn write_metadata(&@trans::crate_ctxt cx, &@crate crate) {
    if (!cx.sess.get_opts().shared) { ret; }
    auto llmeta = C_postr(encoder::encode_metadata(cx, crate));
    auto llconst = trans::C_struct([llmeta]);
    auto llglobal =
        llvm::LLVMAddGlobal(cx.llmod, trans::val_ty(llconst),
                            str::buf("rust_metadata"));
    llvm::LLVMSetInitializer(llglobal, llconst);
    llvm::LLVMSetSection(llglobal, str::buf(x86::get_meta_sect_name()));
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
