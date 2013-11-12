// Mark stack as non-executable
#if defined(__linux__) && defined(__ELF__)
.section	.note.GNU-stack, "", @progbits
#endif

        .text

/*
Callee save registers:
        ebp, ebx, esi, edi

Caller save registers:
        eax, ecx, edx
*/

/*
Saves a set of registers. This is used by our implementation of
getcontext.

The registers_t variable is in (%esp)
*/

#if defined(__APPLE__) || defined(_WIN32)
#define SWAP_REGISTERS _rust_swap_registers
#else
#define SWAP_REGISTERS rust_swap_registers
#endif

// swap_registers(registers_t *oregs, registers_t *regs)
.globl SWAP_REGISTERS
SWAP_REGISTERS:
    // save the old context
    movl 4(%esp), %eax
    movl %ebx, 4(%eax)
    movl %ebp, 16(%eax)
    movl %esi, 20(%eax)
    movl %edi, 24(%eax)

    // save the flags
    pushf
    popl %ecx
    movl %ecx, 44(%eax)

    // save the return address as the instruction pointer
    // and save the stack pointer of the caller
    popl %ecx
    movl %esp, 28(%eax)
    movl %ecx, 48(%eax)

    // restore the new context
    movl 4(%esp), %eax

    movl 4(%eax), %ebx
    movl 16(%eax), %ebp
    movl 20(%eax), %esi
    movl 24(%eax), %edi
    movl 28(%eax), %esp

    // restore the flags
    movl 44(%eax), %ecx
    push %ecx
    popf

    // Return!
    jmp *48(%eax)
