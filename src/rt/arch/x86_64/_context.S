#include "x86_64_regs.h"

        .text

/*
According to ABI documentation found at
http://www.x86-64.org/documentation.html
and Microsoft discussion at
http://msdn.microsoft.com/en-US/library/9z1stfyw%28v=VS.80%29.aspx.

BOTH CALLING CONVENTIONS
        
Callee save registers:
	R12--R15, RDI, RSI, RBX, RBP, RSP
        XMM0--XMM5

Caller save registers:
	RAX, RCX, RDX, R8--R11
        XMM6--XMM15
        Floating point stack

MAC/AMD CALLING CONVENTIONS

Integer arguments go in registers:
        rdi, rsi, rdx, rcx, r8, r9

User flags have no specified role and are not preserved
        across calls, with the exception of DF in %rFLAGS,
        which must be clear (set to "forward" direction)
        on function entry and return.
        
MICROSOFT CALLING CONVENTIONS

Return value: RAX

First four arguments:
        RCX, RDX, R8, R9
        XMM0, XMM1, XMM2, XMM3
*/
	
/*
        Stores current registers into arg0/RCX and restores
        registers found in arg1/RDX. This is used by our
	    implementation of getcontext.
*/	

// swap_registers(registers_t *oregs, registers_t *regs)
.globl swap_registers
swap_registers:
        // n.b. when we enter, the return address is at the top of
        // the stack (i.e., 0(%RSP)) and the argument is in
        // RUSTRT_ARG0_S.  We
        // simply save all NV registers into oregs.
        // We then restore all NV registers from regs.  This restores
        // the old stack pointer, which should include the proper
        // return address. We can therefore just return normally to
        // jump back into the old code.

#       if defined(RUSTRT_ARG0_S)
        mov RUSTRT_ARG0_S, %rdi
        mov RUSTRT_ARG1_S, %rsi
#       endif
        
        // Save instruction pointer:
        pop %rax
        mov %rax, (RUSTRT_IP*8)(%rdi)

        // Save non-volatile integer registers:
        //   (including RSP)
        mov %rbx, (RUSTRT_RBX*8)(%rdi)
        mov %rsp, (RUSTRT_RSP*8)(%rdi)
        mov %rbp, (RUSTRT_RBP*8)(%rdi)
        mov %r12, (RUSTRT_R12*8)(%rdi)
        mov %r13, (RUSTRT_R13*8)(%rdi)
        mov %r14, (RUSTRT_R14*8)(%rdi)
        mov %r15, (RUSTRT_R15*8)(%rdi)

        // Save relevant argument registers:
        mov %rcx, (RUSTRT_RCX*8)(%rdi)
        mov %rdi, (RUSTRT_RDI*8)(%rdi)

        // Save non-volatile XMM registers:
        movapd %xmm0, (RUSTRT_XMM0*8)(%rdi)
        movapd %xmm1, (RUSTRT_XMM1*8)(%rdi)
        movapd %xmm2, (RUSTRT_XMM2*8)(%rdi)
        movapd %xmm3, (RUSTRT_XMM3*8)(%rdi)
        movapd %xmm4, (RUSTRT_XMM4*8)(%rdi)
        movapd %xmm5, (RUSTRT_XMM5*8)(%rdi)

        // Restore non-volatile integer registers:
        //   (including RSP)
        mov (RUSTRT_RBX*8)(%rsi), %rbx
        mov (RUSTRT_RSP*8)(%rsi), %rsp
        mov (RUSTRT_RBP*8)(%rsi), %rbp
        mov (RUSTRT_R12*8)(%rsi), %r12
        mov (RUSTRT_R13*8)(%rsi), %r13
        mov (RUSTRT_R14*8)(%rsi), %r14
        mov (RUSTRT_R15*8)(%rsi), %r15

        // Restore relevant argument registers:
        mov (RUSTRT_RCX*8)(%rsi), %rcx
        mov (RUSTRT_RDI*8)(%rsi), %rdi

        // Restore non-volatile XMM registers:
        movapd (RUSTRT_XMM0*8)(%rsi), %xmm0
        movapd (RUSTRT_XMM1*8)(%rsi), %xmm1
        movapd (RUSTRT_XMM2*8)(%rsi), %xmm2
        movapd (RUSTRT_XMM3*8)(%rsi), %xmm3
        movapd (RUSTRT_XMM4*8)(%rsi), %xmm4
        movapd (RUSTRT_XMM5*8)(%rsi), %xmm5

        // Jump to the instruction pointer
        // found in regs:
        jmp *(RUSTRT_IP*8)(%rsi)

