; Copyright 2013 The Rust Project Developers. See the COPYRIGHT
; file at the top-level directory of this distribution and at
; http://rust-lang.org/COPYRIGHT.
;
; Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
; http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
; <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
; option. This file may not be copied, modified, or distributed
; except according to those terms.

; Rust's try-catch
; When f(...) returns normally, the return value is null.
; When f(...) throws, the return value is a pointer to the caught exception object.

; See also: librustrt/unwind.rs

define i8* @rust_try(void (i8*,i8*)* %f, i8* %fptr, i8* %env) {

    %1 = invoke i8* @rust_try_inner(void (i8*,i8*)* %f, i8* %fptr, i8* %env)
        to label %normal
        unwind label %catch

normal:
    ret i8* %1

catch:
    landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @rust_eh_personality_catch to i8*)
        catch i8* null
    ; rust_try_inner's landing pad does not resume unwinds, so execution will never reach here
    ret i8* null
}

define internal i8* @rust_try_inner(void (i8*,i8*)* %f, i8* %fptr, i8* %env) {

    invoke void %f(i8* %fptr, i8* %env)
        to label %normal
        unwind label %catch

normal:
    ret i8* null

catch:
    %1 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @rust_eh_personality to i8*)
        catch i8* null
    ; extract and return pointer to the exception object
    %2 = extractvalue { i8*, i32 } %1, 0
    ret i8* %2
}

declare i32 @rust_eh_personality(...)
declare i32 @rust_eh_personality_catch(...)
