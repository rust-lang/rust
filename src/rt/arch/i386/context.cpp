#include "context.h"

#include "../../rust.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

extern "C" uint32_t CDECL swap_registers(registers_t *oregs,
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
  uint32_t *sp = (uint32_t *)stack;

  // Shift the stack pointer so the alignment works out right.
  sp = align_down(sp) - 3;
  *--sp = (uint32_t)arg;
  *--sp = 0xdeadbeef;

  regs.esp = (uint32_t)sp;
  regs.eip = (uint32_t)f;
}

#if 0
// This is some useful code to check how the registers struct got
// layed out in memory.
int main() {
  registers_t regs;

  printf("Register offsets\n");

#define REG(r) \
  printf("  %6s: +%ld\n", #r, (intptr_t)&regs.r - (intptr_t)&regs);

  REG(eax);
  REG(ebx);
  REG(ecx);
  REG(edx);
  REG(ebp);
  REG(esi);
  REG(edi);
  REG(esp);

  REG(cs);
  REG(ds);
  REG(ss);
  REG(es);
  REG(fs);
  REG(gs);

  REG(eflags);

  REG(eip);

  return 0;
}
#endif
