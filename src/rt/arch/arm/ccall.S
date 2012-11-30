.text
.code 32
.arm
.align

.globl __morestack
.hidden __morestack
__morestack:
	mov r3, sp
	mov sp, r2
	
	str r3, [sp]
	str lr, [sp, #-4]
	
	sub sp, #8
	
	blx r1

	add sp, #8
	
	ldr lr, [sp, #-4]
	ldr r3, [sp] 
	
	mov sp, r3
	mov pc, lr
	

