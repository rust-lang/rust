declare i32 @eh_rust_personality_catch(...)

define i8* @rust_try(void (i8*,i8*)* %f, i8* %fptr, i8* %env) {

	invoke void %f(i8* %fptr, i8* %env)
		to label %normal
		unwind label %catch

normal:
	ret i8* null

catch:
	%1 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @eh_rust_personality_catch to i8*)
			catch i8* null

	; return pointer to the exception object
    %2 = extractvalue { i8*, i32 } %1, 0
	ret i8* %2
}
