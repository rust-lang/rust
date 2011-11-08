#include "context.h"

#include "../../rust.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

extern "C" void CDECL swap_registers(registers_t *oregs,
                                     registers_t *regs)
asm ("swap_registers");

context::context()
{
    assert((void*)&regs == (void*)this);
}

void context::swap(context &out)
{
    swap_registers(&out.regs, &regs);
}

void context::call(void *f, void *arg, void *stack) {
  // Get the current context, which we will then modify to call the
  // given function.
  swap(*this);

  // set up the trampoline frame
  uint64_t *sp = (uint64_t *)stack;

  // Shift the stack pointer so the alignment works out right.
  sp = align_down(sp) - 3;
  *--sp = (uint64_t)arg;
  *--sp = 0xdeadbeef;

  regs.regs[RSP] = (uint64_t)sp;
  regs.ip = (uint64_t)f;
}
