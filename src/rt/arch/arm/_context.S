// Mark stack as non-executable
#if defined(__linux__) && defined(__ELF__)
.section	.note.GNU-stack, "", %progbits
#endif

.text
.code 32
.arm
#if defined(__APPLE__)
.align 2
#else
.align
#endif

.globl rust_swap_registers
rust_swap_registers:
	str r0, [r0, #0]
	str r3, [r0, #12]
	str r4, [r0, #16]
	str r5, [r0, #20]
	str r6, [r0, #24]
	str r7, [r0, #28]
	str r8, [r0, #32]
	str r9, [r0, #36]
	str r10, [r0, #40]
	str r11, [r0, #44]
	str r12, [r0, #48]
	str sp, [r0, #52]
	str lr, [r0, #56]

	mrs r2, cpsr
	str r2, [r0, #64]


	ldr r0, [r1, #0]
	ldr r3, [r1, #12]
	ldr r4, [r1, #16]
	ldr r5, [r1, #20]
	ldr r6, [r1, #24]
	ldr r7, [r1, #28]
	ldr r8, [r1, #32]
	ldr r9, [r1, #36]
	ldr r10, [r1, #40]
	ldr r11, [r1, #44]
	ldr r12, [r1, #48]

	ldr sp, [r1, #52]
	ldr lr, [r1, #56]

	ldr r2, [r1, #64]
	msr cpsr_cxsf, r2

	mov pc, lr
