.macro	func _name
	.text
	.align	2
	.globl	\_name
	.type	\_name, %function
\_name:
.endm

.macro	endfunc _name
	.size	\_name, .-\_name
.endm
