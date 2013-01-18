// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


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

class context {
public:
  registers_t regs;

  context();

  context *next;

  void swap(context &out);
  void call(void *f, void *arg, void *sp);
};

#endif
