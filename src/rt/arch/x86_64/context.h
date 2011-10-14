// -*- mode: c++ -*-

#ifndef CONTEXT_H
#define CONTEXT_H

#include <cstdlib>
#include <inttypes.h>
#include <stdint.h>

#ifdef HAVE_VALGRIND
#include <valgrind/memcheck.h>
#endif

template<typename T>
T align_down(T sp)
{
    // There is no platform we care about that needs more than a
    // 16-byte alignment.
    return (T)((uint64_t)sp & ~(16 - 1));
}

struct registers_t {
    uint64_t regs[7]; // Space for the volatile regs: rbx, rsp, rbp, r12:r15
    uint64_t xmms[6]; // Space for the volatile regs: xmm0:xmm5
    uint64_t ip;
};

class context {
public:
  static const int RSP = 1;

  registers_t regs;

  context();

  context *next;

  void swap(context &out);
  void call(void *f, void *arg, void *sp);
  void call(void *f, void *sp);

  // Note that this doesn't actually adjust esp. Instead, we adjust esp when
  // we actually do the call. This is needed for exception safety -- if the
  // function being called causes the task to fail, then we have to avoid
  // leaking space on the C stack.
  inline void *alloc_stack(size_t nbytes) {
    uint32_t bot = regs.regs[RSP];
    uint32_t top = align_down(bot - nbytes);

#ifdef HAVE_VALGRIND
    (void)VALGRIND_MAKE_MEM_UNDEFINED(top - 4, bot - top + 4);
#endif

    return reinterpret_cast<void *>(top);
  }
};

#endif
