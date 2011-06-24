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
    // and save the stack pointer of the caller
    popl %ecx
    movl %esp, 28(%eax)
	movl %ecx, 48(%eax)

	// return 0
	xor %eax, %eax
	jmp *%ecx

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

    // get ready to return.
	mov 48(%eax), %ecx
	push %ecx	
	
	// okay, now we can restore ecx.
	movl 8(%eax), %ecx

	// return 1 to the saved eip
	movl $1, %eax
	ret

// swap_registers(registers_t *oregs, registers_t *regs)
.globl swap_registers
swap_registers:
    // %eax = get_registers(oregs);
    movl 4(%esp), %eax
    push %eax
    call get_registers
        
    // if(!%eax) goto call_set
    test %eax, %eax
    jz call_set

    // else
    addl $4, %esp
    ret
        
call_set:
    // set_registers(regs)
    movl 12(%esp), %eax
    movl %eax, 0(%esp)
    call set_registers
    // set_registers never returns