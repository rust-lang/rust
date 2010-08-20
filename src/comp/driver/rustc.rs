// -*- rust -*-

import std._str;
import lib.llvm.llvm;
import lib.llvm.builder;
import fe.parser;
import fe.token;

fn write_module() {
    auto llmod =
        llvm.LLVMModuleCreateWithNameInContext(_str.buf("rust_out"),
                                               llvm.LLVMGetGlobalContext());

    auto b = builder(llvm.LLVMCreateBuilder());

    llvm.LLVMWriteBitcodeToFile(llmod, _str.buf("rust_out.bc"));
    llvm.LLVMDisposeModule(llmod);
}

fn main(vec[str] args) {

  log "This is the rust 'self-hosted' compiler.";
  log "The one written in rust.";
  log "It does nothing yet, it's a placeholder.";
  log "You want rustboot, the compiler next door.";

  auto i = 0;
  for (str filename in args) {
    if (i > 0) {
        auto p = parser.new_parser(filename);
        log "opened file: " + filename;
        auto tok = p.peek();
        while (true) {
            alt (tok) {
                case (token.EOF()) { ret; }
                case (_) {
                    log token.to_str(tok);
                    p.bump();
                    tok = p.peek();
                }
            }
        }
    }
    i += 1;
  }

  // Test LLVM module-writing. Nothing interesting yet.
  write_module();

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
