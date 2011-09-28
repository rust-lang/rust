// -*- mode: c++ -*-

#ifndef CONTEXT_H
#define CONTEXT_H

#include <cstdlib>
#include <inttypes.h>
#include <stdint.h>

template<typename T>
T align_down(T sp)
{
    // There is no platform we care about that needs more than a
    // 16-byte alignment.
    return (T)((uint32_t)sp & ~(16 - 1));
}

struct registers_t {
  // general purpose registers
  uint32_t eax, ebx, ecx, edx, ebp, esi, edi, esp;

  // segment registers
  uint16_t cs, ds, ss, es, fs, gs;

  uint32_t eflags;

  uint32_t eip;
};

class context {
public:
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
    return (void *)(align_down(regs.esp - nbytes));
  }
};

#endif
