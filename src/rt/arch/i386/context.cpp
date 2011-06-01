#include "context.h"

#include "../../rust.h"

#include <stdio.h>
#include <stdlib.h>

extern "C" uint32_t CDECL get_registers(registers_t *regs) 
  asm ("get_registers");
extern "C" uint32_t CDECL set_registers(registers_t *regs)
  asm ("set_registers");

context::context()
  : next(NULL)
{
  get_registers(&regs);
}

void context::set()
{
  //printf("Activating %p...\n", this);
  set_registers(&regs);
}

void context::swap(context &out)
{
  //printf("Swapping to %p and saving in %p\n", this, &out);
  uint32_t r = get_registers(&out.regs);
  //printf("get_registers = %d, sp = 0x%x\n", r, out.regs.esp);
  if(!r) {
    set();
  }
  //printf("Resumed %p...\n", &out);
}

void context::call(void *f, void *arg, void *stack) {
  // set up the trampoline frame
  uint32_t *sp = (uint32_t *)stack;
  *--sp = (uint32_t)this;
  *--sp = (uint32_t)arg;
  *--sp = 0xdeadbeef; //(uint32_t)ctx_trampoline1;
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
