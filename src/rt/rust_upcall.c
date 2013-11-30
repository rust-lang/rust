// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*
  Upcalls

  These are runtime functions that the compiler knows about and generates
  calls to. They are called on the Rust stack and, in most cases, immediately
  switch to the C stack.
 */

#include "rust_globals.h"

//Unwinding ABI declarations.
typedef int _Unwind_Reason_Code;
typedef int _Unwind_Action;

struct _Unwind_Context;
struct _Unwind_Exception;

#if __USING_SJLJ_EXCEPTIONS__
#   define PERSONALITY_FUNC __gxx_personality_sj0
#else
#   ifdef __SEH__
#       define PERSONALITY_FUNC __gxx_personality_seh0
#   else
#       define PERSONALITY_FUNC __gxx_personality_v0
#   endif
#endif

_Unwind_Reason_Code
PERSONALITY_FUNC(int version,
                     _Unwind_Action actions,
                     uint64_t exception_class,
                     struct _Unwind_Exception *ue_header,
                     struct _Unwind_Context *context);

struct s_rust_personality_args {
    _Unwind_Reason_Code retval;
    int version;
    _Unwind_Action actions;
    uint64_t exception_class;
    struct _Unwind_Exception *ue_header;
    struct _Unwind_Context *context;
};

static void
upcall_s_rust_personality(struct s_rust_personality_args *args) {
    args->retval = PERSONALITY_FUNC(args->version,
                                    args->actions,
                                    args->exception_class,
                                    args->ue_header,
                                    args->context);
}

/**
   The exception handling personality function. It figures
   out what to do with each landing pad. Just a stack-switching
   wrapper around the C++ personality function.
*/
_Unwind_Reason_Code
upcall_rust_personality(int version,
                        _Unwind_Action actions,
                        uint64_t exception_class,
                        struct _Unwind_Exception *ue_header,
                        struct _Unwind_Context *context) {
    struct s_rust_personality_args args = {(_Unwind_Reason_Code)0,
                                           version, actions, exception_class,
                                           ue_header, context};
    upcall_s_rust_personality(&args);
    return args.retval;
}

// NOTE: remove after stage0
// Landing pads need to call this to insert the
// correct limit into TLS.
// NB: This must run on the Rust stack because it
// needs to acquire the value of the stack pointer
void
upcall_reset_stack_limit() {
}

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
