// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include "context.h"
#include "../../rust_globals.h"

extern "C" void CDECL swap_registers(registers_t *oregs,
                                     registers_t *regs)
asm ("swap_registers");

context::context()
{
    assert((void*)&regs == (void*)this);
    memset(&regs, 0, sizeof(regs));
}

void context::swap(context &out)
{
    swap_registers(&out.regs, &regs);
}

void context::call(void *f, void *arg, void *stack)
{
  // Get the current context, which we will then modify to call the
  // given function.
  swap(*this);

  // set up the stack
  uint32_t *sp = (uint32_t *)stack;
  sp = align_down(sp);
  // The final return address. 0 indicates the bottom of the stack
  // sp of mips o32 is 8-byte aligned
  sp -= 2;
  *sp = 0;

  regs.data[4] = (uint32_t)arg;
  regs.data[29] = (uint32_t)sp;
  regs.data[25] = (uint32_t)f;
  regs.data[31] = (uint32_t)f;

  // Last base pointer on the stack should be 0
}
