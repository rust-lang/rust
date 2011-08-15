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

// swap_registers(registers_t *oregs, registers_t *regs)
.globl swap_registers
swap_registers:
    // save the old context
    movl 4(%esp), %eax
	//movl %eax, 0(%eax)
	movl %ebx, 4(%eax)
	movl %ecx, 8(%eax)
	movl %edx, 12(%eax)
	movl %ebp, 16(%eax)
	movl %esi, 20(%eax)
	movl %edi, 24(%eax)
	//movl %cs, 32(%eax)
	//movl %ds, 34(%eax)
	//movl %ss, 36(%eax)
	//movl %es, 38(%eax)
	//movl %fs, 40(%eax)
	//movl %gs, 42(%eax)

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
	// save ecx for later...
	movl 12(%eax), %edx
	movl 16(%eax), %ebp
	movl 20(%eax), %esi
	movl 24(%eax), %edi
	movl 28(%eax), %esp
	// We can't actually change this...
	//movl 32(%eax), %cs
	//movl 34(%eax), %ds
	//movl 36(%eax), %ss
	//movl 38(%eax), %es
	//movl 40(%eax), %fs
	//movl 42(%eax), %gs

	// restore the flags
	movl 44(%eax), %ecx
	push %ecx
	popf

    // ok, now we can restore ecx
    movl 8(%eax), %ecx
	
    // Return!
    jmp *48(%eax)


.globl task_trampoline
task_trampoline:
    // This gets set up by std::task::_spawn.
#if defined(__APPLE__) || defined(__WIN32__)
    call _task_exit
#else
    call task_exit
#endif
