// -*- mode: c++ -*-

#ifndef CONTEXT_H
#define CONTEXT_H

#include <cstdlib>
#include <inttypes.h>
#include <stdint.h>

#include "vg/memcheck.h"

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
} __attribute__((aligned(16)));

extern "C" void __morestack(void *args, void *fn_ptr, uintptr_t stack_ptr);

class context {
public:
  registers_t regs;

  context();

  context *next;

  void swap(context &out);
  void call(void *f, void *arg, void *sp);

  void call_and_change_stacks(void *args, void *fn_ptr) {
      __morestack(args, fn_ptr, regs.esp);
  }
};

#endif
