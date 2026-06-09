
	.macro ABI2_FUNCTION_BEGIN name
	.p2align 2
	.section .text.\name,"ax",@progbits
	.globl \name
	.type  \name, @function
\name:
	.endm

	.macro ABI2_FUNCTION_END name
	.size  \name, . - \name
	.endm


ABI2_FUNCTION_BEGIN __save_r16_through_r27
        {
                memd(fp+#-48) = r27:26
                memd(fp+#-40) = r25:24
        }
        {
                memd(fp+#-32) = r23:22
                memd(fp+#-24) = r21:20
        }
        {
                memd(fp+#-16) = r19:18
                memd(fp+#-8) = r17:16
                jumpr lr
        }
ABI2_FUNCTION_END __save_r16_through_r27

ABI2_FUNCTION_BEGIN __save_r16_through_r25
        {
                memd(fp+#-40) = r25:24
                memd(fp+#-32) = r23:22
        }
        {
                memd(fp+#-24) = r21:20
                memd(fp+#-16) = r19:18
        }
        {
                memd(fp+#-8) = r17:16
                jumpr lr
        }
ABI2_FUNCTION_END __save_r16_through_r25

ABI2_FUNCTION_BEGIN __save_r16_through_r23
        {
                memd(fp+#-32) = r23:22
                memd(fp+#-24) = r21:20
        }
        {
                memd(fp+#-16) = r19:18
                memd(fp+#-8) = r17:16
                jumpr lr
        }
ABI2_FUNCTION_END __save_r16_through_r23

ABI2_FUNCTION_BEGIN __save_r16_through_r21
        {
                memd(fp+#-24) = r21:20
                memd(fp+#-16) = r19:18
        }
        {
                memd(fp+#-8) = r17:16
                jumpr lr
        }
ABI2_FUNCTION_END __save_r16_through_r21

ABI2_FUNCTION_BEGIN __save_r16_through_r19
        {
                memd(fp+#-16) = r19:18
                memd(fp+#-8) = r17:16
                jumpr lr
        }
ABI2_FUNCTION_END __save_r16_through_r19

ABI2_FUNCTION_BEGIN __save_r16_through_r17
        {
                memd(fp+#-8) = r17:16
                jumpr lr
        }
ABI2_FUNCTION_END __save_r16_through_r17


ABI2_FUNCTION_BEGIN __restore_r16_through_r27_and_deallocframe_before_tailcall
                r27:26 = memd(fp+#-48)
        {
                r25:24 = memd(fp+#-40)
                r23:22 = memd(fp+#-32)
        }
        {
                r21:20 = memd(fp+#-24)
                r19:18 = memd(fp+#-16)
        }
        {
                r17:16 = memd(fp+#-8)
                deallocframe
                jumpr lr
        }
ABI2_FUNCTION_END __restore_r16_through_r27_and_deallocframe_before_tailcall

ABI2_FUNCTION_BEGIN __restore_r16_through_r25_and_deallocframe_before_tailcall
        {
                r25:24 = memd(fp+#-40)
                r23:22 = memd(fp+#-32)
        }
        {
                r21:20 = memd(fp+#-24)
                r19:18 = memd(fp+#-16)
        }
        {
                r17:16 = memd(fp+#-8)
                deallocframe
                jumpr lr
        }
ABI2_FUNCTION_END __restore_r16_through_r25_and_deallocframe_before_tailcall

ABI2_FUNCTION_BEGIN __restore_r16_through_r23_and_deallocframe_before_tailcall
        {
                r23:22 = memd(fp+#-32)
                r21:20 = memd(fp+#-24)
        }
                r19:18 = memd(fp+#-16)
        {
                r17:16 = memd(fp+#-8)
                deallocframe
                jumpr lr
        }
ABI2_FUNCTION_END __restore_r16_through_r23_and_deallocframe_before_tailcall


ABI2_FUNCTION_BEGIN __restore_r16_through_r21_and_deallocframe_before_tailcall
        {
                r21:20 = memd(fp+#-24)
                r19:18 = memd(fp+#-16)
        }
        {
                r17:16 = memd(fp+#-8)
                deallocframe
                jumpr lr
        }
ABI2_FUNCTION_END __restore_r16_through_r21_and_deallocframe_before_tailcall

ABI2_FUNCTION_BEGIN __restore_r16_through_r19_and_deallocframe_before_tailcall
                r19:18 = memd(fp+#-16)
        {
                r17:16 = memd(fp+#-8)
                deallocframe
                jumpr lr
        }
ABI2_FUNCTION_END __restore_r16_through_r19_and_deallocframe_before_tailcall

ABI2_FUNCTION_BEGIN __restore_r16_through_r17_and_deallocframe_before_tailcall
        {
                r17:16 = memd(fp+#-8)
                deallocframe
                jumpr lr
        }
ABI2_FUNCTION_END __restore_r16_through_r17_and_deallocframe_before_tailcall


ABI2_FUNCTION_BEGIN __restore_r16_through_r27_and_deallocframe
                r27:26 = memd(fp+#-48)
        {
                r25:24 = memd(fp+#-40)
                r23:22 = memd(fp+#-32)
        }
        {
                r21:20 = memd(fp+#-24)
                r19:18 = memd(fp+#-16)
        }
	{
		r17:16 = memd(fp+#-8)
		dealloc_return
	}
ABI2_FUNCTION_END __restore_r16_through_r27_and_deallocframe

ABI2_FUNCTION_BEGIN __restore_r16_through_r25_and_deallocframe
        {
                r25:24 = memd(fp+#-40)
                r23:22 = memd(fp+#-32)
        }
        {
                r21:20 = memd(fp+#-24)
                r19:18 = memd(fp+#-16)
        }
	{
		r17:16 = memd(fp+#-8)
		dealloc_return
	}
ABI2_FUNCTION_END __restore_r16_through_r25_and_deallocframe

ABI2_FUNCTION_BEGIN __restore_r16_through_r23_and_deallocframe
        {
                r23:22 = memd(fp+#-32)
        }
        {
                r21:20 = memd(fp+#-24)
                r19:18 = memd(fp+#-16)
        }
	{
		r17:16 = memd(fp+#-8)
		dealloc_return
	}
ABI2_FUNCTION_END __restore_r16_through_r23_and_deallocframe

ABI2_FUNCTION_BEGIN __restore_r16_through_r21_and_deallocframe
        {
                r21:20 = memd(fp+#-24)
                r19:18 = memd(fp+#-16)
        }
	{
		r17:16 = memd(fp+#-8)
		dealloc_return
	}
ABI2_FUNCTION_END __restore_r16_through_r21_and_deallocframe

ABI2_FUNCTION_BEGIN __restore_r16_through_r19_and_deallocframe
	{
                r19:18 = memd(fp+#-16)
		r17:16 = memd(fp+#-8)
        }
        {
		dealloc_return
	}
ABI2_FUNCTION_END __restore_r16_through_r19_and_deallocframe

ABI2_FUNCTION_BEGIN __restore_r16_through_r17_and_deallocframe
	{
		r17:16 = memd(fp+#-8)
		dealloc_return
	}
ABI2_FUNCTION_END __restore_r16_through_r17_and_deallocframe

ABI2_FUNCTION_BEGIN __deallocframe
        dealloc_return
ABI2_FUNCTION_END __deallocframe
