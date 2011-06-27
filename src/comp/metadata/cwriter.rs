// Writing metadata into crate files

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
