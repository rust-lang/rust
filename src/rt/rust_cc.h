// Rust cycle collector. Temporary, but will probably stick around for some
// time until LLVM's GC infrastructure is more mature.

#ifndef RUST_CC_H
#define RUST_CC_H

struct rust_task;

namespace cc {

void do_cc(rust_task *task);

// performs a cycle coll then asserts that there is nothing left
void do_final_cc(rust_task *task);

void maybe_cc(rust_task *task);

}   // end namespace cc

#endif

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
