// Rust cycle collector. Temporary, but will probably stick around for some
// time until LLVM's GC infrastructure is more mature.

#ifndef RUST_CC_H
#define RUST_CC_H

struct rust_task;

namespace cc {

void do_cc(rust_task *task);
void maybe_cc(rust_task *task);

}   // end namespace cc

#endif

