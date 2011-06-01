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
	
.globl get_registers
get_registers:
	movl 4(%esp), %eax
	movl %eax, 0(%eax)
	movl %ebx, 4(%eax)
	movl %ecx, 8(%eax)
	movl %edx, 12(%eax)
	movl %ebp, 16(%eax)
	movl %esi, 20(%eax)
	movl %edi, 24(%eax)
	movl %esp, 28(%eax)
	movw %cs, 32(%eax)
	movw %ds, 34(%eax)
	movw %ss, 36(%eax)
	movw %es, 38(%eax)
	movw %fs, 40(%eax)
	movw %gs, 42(%eax)

	// save the flags
	pushf
	popl %ecx
	movl %ecx, 44(%eax)

	// save the return address as the instruction pointer
	movl 0(%esp), %ecx
	movl %ecx, 48(%eax)

	// return 0
	xor %eax, %eax
	ret

.globl set_registers
set_registers:
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
	movw 34(%eax), %ds
	movw 36(%eax), %ss
	movw 38(%eax), %es
	movw 40(%eax), %fs
	movw 42(%eax), %gs

	// restore the flags
	movl 44(%eax), %ecx
	push %ecx
	popf

	// get ready to return back to the old eip
	// We could write this directly to 0(%esp), but Valgrind on OS X
	// complains.
	pop %ecx
	mov 48(%eax), %ecx
	push %ecx	
	//movl %ecx, 0(%esp)
	
	// okay, now we can restore ecx.
	movl 8(%eax), %ecx

	// return 1 to the saved eip
	movl $1, %eax
	ret
