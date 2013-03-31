// Mark stack as non-executable
#if defined(__linux__) && defined(__ELF__)
.section	.note.GNU-stack, "", @progbits
#endif

.text
.code 32
.arm
.align


.globl record_sp_limit
.globl get_sp_limit
.globl get_sp

record_sp_limit:
	mov r3, r0
	ldr r0, =my_cpu
	mov r1, #0
	mov r2, #0
    stmfd   sp!, {r3, r7}
    ldr     r7, =345
    swi     #0
    ldmfd   sp!, {r3, r7}
    movs    r0, r0
	movmi	r0, #0

	ldr r1, =my_array
	str r3, [r1, r0]
	mov pc, lr


get_sp_limit:
    ldr r0, =my_cpu
	mov r1, #0
	mov r2, #0
    stmfd   sp!, {r4, r7}
    ldr     r7, =345
    swi     #0
    ldmfd   sp!, {r4, r7}
    movs    r0, r0
	movmi	r0, #0
	mov r3, r0

	ldr r1, =my_array
	ldr r0, [r1, r3]
	mov pc, lr


get_sp:
	mov r0, sp
	mov pc, lr

.data
my_cpu:	.long	0
.global my_array
my_array:
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
.end
