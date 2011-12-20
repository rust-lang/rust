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

  // set up the stack
  uint64_t *sp = (uint64_t *)stack;
  sp = align_down(sp);
  // The final return address. 0 indicates the bottom of the stack
  *--sp = 0;

  regs.data[RUSTRT_ARG0] = (uint64_t)arg;
  regs.data[RUSTRT_RSP] = (uint64_t)sp;
  regs.data[RUSTRT_IP] = (uint64_t)f;

  // Last base pointer on the stack should be 0
  regs.data[RUSTRT_RBP] = 0;
}
