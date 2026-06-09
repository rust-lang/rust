
FUNCTION_BEGIN __save_r27_through_r16
		memd(fp+#-48) = r17:16
FALLTHROUGH_TAIL_CALL __save_r27_through_r16 __save_r27_through_r18
		memd(fp+#-40) = r19:18
FALLTHROUGH_TAIL_CALL __save_r27_through_r18 __save_r27_through_r20
		memd(fp+#-32) = r21:20
FALLTHROUGH_TAIL_CALL __save_r27_through_r20 __save_r27_through_r22
		memd(fp+#-24) = r23:22
FALLTHROUGH_TAIL_CALL __save_r27_through_r22 __save_r27_through_r24
		memd(fp+#-16) = r25:24
	{
		memd(fp+#-8) = r27:26
		jumpr lr
	}
FUNCTION_END __save_r27_through_r24


FUNCTION_BEGIN __restore_r27_through_r20_and_deallocframe_before_sibcall
	{
		r21:20 = memd(fp+#-32)
		r23:22 = memd(fp+#-24)
	}
FALLTHROUGH_TAIL_CALL __restore_r27_through_r20_and_deallocframe_before_sibcall __restore_r27_through_r24_and_deallocframe_before_sibcall
	{
		r25:24 = memd(fp+#-16)
		jump __restore_r27_through_r26_and_deallocframe_before_sibcall
	}
FUNCTION_END __restore_r27_through_r24_and_deallocframe_before_sibcall


FUNCTION_BEGIN __restore_r27_through_r16_and_deallocframe_before_sibcall
		r17:16 = memd(fp+#-48)
FALLTHROUGH_TAIL_CALL __restore_r27_through_r16_and_deallocframe_before_sibcall __restore_r27_through_r18_and_deallocframe_before_sibcall
	{
		r19:18 = memd(fp+#-40)
		r21:20 = memd(fp+#-32)
	}
FALLTHROUGH_TAIL_CALL __restore_r27_through_r18_and_deallocframe_before_sibcall __restore_r27_through_r22_and_deallocframe_before_sibcall
	{
		r23:22 = memd(fp+#-24)
		r25:24 = memd(fp+#-16)
	}
FALLTHROUGH_TAIL_CALL __restore_r27_through_r22_and_deallocframe_before_sibcall __restore_r27_through_r26_and_deallocframe_before_sibcall
	{
		r27:26 = memd(fp+#-8)
		deallocframe
		jumpr lr
	}
FUNCTION_END __restore_r27_through_r26_and_deallocframe_before_sibcall


FUNCTION_BEGIN __restore_r27_through_r16_and_deallocframe
	{
		r17:16 = memd(fp+#-48)
		r19:18 = memd(fp+#-40)
	}
FALLTHROUGH_TAIL_CALL __restore_r27_through_r16_and_deallocframe __restore_r27_through_r20_and_deallocframe
	{
		r21:20 = memd(fp+#-32)
		r23:22 = memd(fp+#-24)
	}
FALLTHROUGH_TAIL_CALL __restore_r27_through_r20_and_deallocframe __restore_r27_through_r24_and_deallocframe
	{
		lr = memw(fp+#4)
		r25:24 = memd(fp+#-16)
	}
	{
		r27:26 = memd(fp+#-8)
		deallocframe
		jumpr lr
	}
FUNCTION_END __restore_r27_through_r24_and_deallocframe


FUNCTION_BEGIN __restore_r27_through_r18_and_deallocframe
	{
		r19:18 = memd(fp+#-40)
		r21:20 = memd(fp+#-32)
	}
FALLTHROUGH_TAIL_CALL __restore_r27_through_r18_and_deallocframe __restore_r27_through_r22_and_deallocframe
	{
		r23:22 = memd(fp+#-24)
		r25:24 = memd(fp+#-16)
	}
FALLTHROUGH_TAIL_CALL __restore_r27_through_r22_and_deallocframe __restore_r27_through_r26_and_deallocframe
	{
		r27:26 = memd(fp+#-8)
		deallocframe
	}
		jumpr lr
FUNCTION_END __restore_r27_through_r26_and_deallocframe
