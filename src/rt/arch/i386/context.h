// -*- mode: c++ -*-

#ifndef CONTEXT_H
#define CONTEXT_H

#include <inttypes.h>

struct registers_t {
  // general purpose registers
  uint32_t eax, ebx, ecx, edx, ebp, esi, edi, esp;

  // segment registers
  uint16_t cs, ds, ss, es, fs, gs;

  uint32_t eflags;

  uint32_t eip;
};

class context {
  registers_t regs;

public:
  context();
  
  context *next;

  void set();
  
  void swap(context &out);

  void call(void *f, void *arg, void *sp);
};

template<typename T>
T align_down(T sp)
{
    // There is no platform we care about that needs more than a
    // 16-byte alignment.
    return (T)((int)sp & ~(16 - 1));
}

#endif
