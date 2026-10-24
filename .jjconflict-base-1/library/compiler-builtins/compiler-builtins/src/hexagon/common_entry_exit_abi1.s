
FUNCTION_BEGIN __save_r24_through_r27
		memd(fp+#-16) = r27:26
FALLTHROUGH_TAIL_CALL __save_r24_through_r27 __save_r24_through_r25
	{
		memd(fp+#-8) = r25:24
		jumpr lr
	}
FUNCTION_END __save_r24_through_r25


FUNCTION_BEGIN __restore_r24_through_r27_and_deallocframe_before_tailcall
		r27:26 = memd(fp+#-16)
FALLTHROUGH_TAIL_CALL __restore_r24_through_r27_and_deallocframe_before_tailcall __restore_r24_through_r25_and_deallocframe_before_tailcall
	{
		r25:24 = memd(fp+#-8)
		deallocframe
		jumpr lr
	}
FUNCTION_END __restore_r24_through_r25_and_deallocframe_before_tailcall


FUNCTION_BEGIN __restore_r24_through_r27_and_deallocframe
	{
		lr = memw(fp+#4)
		r27:26 = memd(fp+#-16)
	}
	{
		r25:24 = memd(fp+#-8)
		deallocframe
		jumpr lr
	}
FUNCTION_END __restore_r24_through_r27_and_deallocframe


FUNCTION_BEGIN __restore_r24_through_r25_and_deallocframe
	{
		r25:24 = memd(fp+#-8)
		deallocframe
	}
		jumpr lr
FUNCTION_END __restore_r24_through_r25_and_deallocframe
