; Copyright 2015 The Rust Project Developers. See the COPYRIGHT
; file at the top-level directory of this distribution and at
; http://rust-lang.org/COPYRIGHT.
;
; Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
; http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
; <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
; option. This file may not be copied, modified, or distributed
; except according to those terms.

; 64-bit MSVC's definition of the `rust_try` function. This function can't be
; defined in Rust as it's a "try-catch" block that's not expressible in Rust's
; syntax, so we're using LLVM to produce an object file with the associated
; handler.
;
; To use the correct system implementation details, this file is separate from
; the standard rust_try.ll as we need specifically use the __C_specific_handler
; personality function or otherwise LLVM doesn't emit SEH handling tables.
; There's also a few fiddly bits about SEH right now in LLVM that require us to
; structure this a fairly particular way!
;
; See also: src/libstd/rt/unwind/seh.rs

define i8* @rust_try(void (i8*)* %f, i8* %env) {
    invoke void %f(i8* %env)
        to label %normal
        unwind label %catch

normal:
    ret i8* null

; Here's where most of the magic happens, this is the only landing pad in rust
; tagged with "catch" to indicate that we're catching an exception. The other
; catch handlers in rust_try.ll just catch *all* exceptions, but that's because
; most exceptions are already filtered out by their personality function.
;
; For MSVC we're just using a standard personality function that we can't
; customize, so we need to do the exception filtering ourselves, and this is
; currently performed by the `__rust_try_filter` function. This function,
; specified in the landingpad instruction, will be invoked by Windows SEH
; routines and will return whether the exception in question can be caught (aka
; the Rust runtime is the one that threw the exception).
;
; To get this to compile (currently LLVM segfaults if it's not in this
; particular structure), when the landingpad is executing we test to make sure
; that the ID of the exception being thrown is indeed the one that we were
; expecting. If it's not, we resume the exception, and otherwise we return the
; pointer that we got
;
; Full disclosure: It's not clear to me what this `llvm.eh.typeid` stuff is
; doing *other* then just allowing LLVM to compile this file without
; segfaulting. I would expect the entire landing pad to just be:
;
;     %vals = landingpad ...
;     %ehptr = extractvalue { i8*, i32 } %vals, 0
;     ret i8* %ehptr
;
; but apparently LLVM chokes on this, so we do the more complicated thing to
; placate it.
catch:
    %vals = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
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

declare i32 @__C_specific_handler(...)
declare i32 @__rust_try_filter(i8*, i8*)
declare i32 @llvm.eh.typeid.for(i8*) readnone nounwind
