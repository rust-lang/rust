; Copyright 2015 The Rust Project Developers. See the COPYRIGHT
; file at the top-level directory of this distribution and at
; http://rust-lang.org/COPYRIGHT.
;
; Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
; http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
; <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
; option. This file may not be copied, modified, or distributed
; except according to those terms.

; For more comments about what's going on here see rust_try_msvc_64.ll. The only
; difference between that and this file is the personality function used as it's
; different for 32-bit MSVC than it is for 64-bit.

define i8* @rust_try(void (i8*)* %f, i8* %env)
    personality i8* bitcast (i32 (...)* @_except_handler3 to i8*)
{
    invoke void %f(i8* %env)
        to label %normal
        unwind label %catch

normal:
    ret i8* null
catch:
    %vals = landingpad { i8*, i32 }
              catch i8* bitcast (i32 (i8*, i8*)* @__rust_try_filter to i8*)
    %ehptr = extractvalue { i8*, i32 } %vals, 0
    %sel = extractvalue { i8*, i32 } %vals, 1
    %filter_sel = call i32 @llvm.eh.typeid.for(i8* bitcast (i32 (i8*, i8*)* @__rust_try_filter to i8*))
    %is_filter = icmp eq i32 %sel, %filter_sel
    br i1 %is_filter, label %catch-return, label %catch-resume

catch-return:
    ret i8* %ehptr

catch-resume:
    resume { i8*, i32 } %vals
}

declare i32 @_except_handler3(...)
declare i32 @__rust_try_filter(i8*, i8*)
declare i32 @llvm.eh.typeid.for(i8*) readnone nounwind
