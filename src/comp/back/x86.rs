
import lib::llvm::llvm;
import lib::llvm::llvm::ModuleRef;
import std::str;
import std::istr;
import std::os::target_os;

fn get_module_asm() -> str { ret ""; }

fn get_meta_sect_name() -> str {
    if istr::eq(target_os(), ~"macos") { ret "__DATA,__note.rustc"; }
    if istr::eq(target_os(), ~"win32") { ret ".note.rustc"; }
    ret ".note.rustc";
}

fn get_data_layout() -> str {
    if istr::eq(target_os(), ~"macos") {
        ret "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16" + "-i32:32:32-i64:32:64" +
                "-f32:32:32-f64:32:64-v64:64:64" +
                "-v128:128:128-a0:0:64-f80:128:128" + "-n8:16:32";
    }
    if istr::eq(target_os(), ~"win32") {
        ret "e-p:32:32-f64:64:64-i64:64:64-f80:32:32-n8:16:32";
    }
    ret "e-p:32:32-f64:32:64-i64:32:64-f80:32:32-n8:16:32";
}

fn get_target_triple() -> str {
    if istr::eq(target_os(), ~"macos") { ret "i686-apple-darwin"; }
    if istr::eq(target_os(), ~"win32") { ret "i686-pc-mingw32"; }
    ret "i686-unknown-linux-gnu";
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
