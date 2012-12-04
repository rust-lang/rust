// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Unwinding stuff missing on some architectures (Mac OS X).

#ifndef RUST_UNWIND_H
#define RUST_UNWIND_H

#ifdef __APPLE__
#include <libunwind.h>

typedef void _Unwind_Context;
typedef int _Unwind_Reason_Code;

#else

#include <unwind.h>

#endif

#if (defined __APPLE__) || (defined __clang__)

#ifndef __FreeBSD__

typedef int _Unwind_Action;
typedef void _Unwind_Exception;

#endif

#endif

#endif

