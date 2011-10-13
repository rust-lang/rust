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
        // the stack (i.e., 0(%RSP)).  We
        // simply save all NV registers into oregs.
        // We then restore all NV registers from regs.  This restores
        // the old stack pointer, which should include the proper
        // return address. We can therefore just return normally to
        // jump back into the old code.
        
        // Save instruction pointer:
        popl %rax
        movl %rax, 104(%rdi)

        // Save non-volatile integer registers:
        //   (including RSP)
        movl %rbx,   0(%rdi)
        movl %rsp,   8(%rdi)
        movl %rbp,  16(%rdi)
        movl %r12,  24(%rdi)
        movl %r13,  32(%rdi)
        movl %r14,  40(%rdi)
        movl %r15,  48(%rdi)

        // Save non-volatile XMM registers:
        movl %xmm0, 56(%rdi)
        movl %xmm1, 64(%rdi)
        movl %xmm2, 72(%rdi)
        movl %xmm3, 80(%rdi)
        movl %xmm4, 88(%rdi)
        movl %xmm5, 96(%rdi)

        // Restore non-volatile integer registers:
        //   (including RSP)
        movl  0(%rsi), %rbx
        movl  8(%rsi), %rsp
        movl 16(%rsi), %rbp
        movl 24(%rsi), %r12
        movl 32(%rsi), %r13
        movl 40(%rsi), %r14
        movl 48(%rsi), %r15

        // Restore non-volatile XMM registers:
        movl 56(%rsi), %xmm0
        movl 64(%rsi), %xmm1
        movl 72(%rsi), %xmm2
        movl 80(%rsi), %xmm3
        movl 88(%rsi), %xmm4
        movl 96(%rsi), %xmm5

        // Jump to the instruction pointer
        // found in regs:
        jmp *104(%rsi)

