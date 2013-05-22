// xfail-license

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
  uint32_t *sp = ( uint32_t *)stack;
  sp = align_down(sp);
  // The final return address. 0 indicates the bottom of the stack
  // sp of arm eabi is 8-byte aligned
  sp -= 2;
  *sp = 0;

  regs.data[0] = ( uint32_t )arg; // r0
  regs.data[13] = ( uint32_t )sp; //#52 sp, r13
  regs.data[14] = ( uint32_t )f;  //#60 pc, r15 --> lr,
  // Last base pointer on the stack should be 0
}
