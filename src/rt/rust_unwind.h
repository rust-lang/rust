// Unwinding stuff missing on some architectures (Mac OS X).

#ifndef RUST_UNWIND_H
#define RUST_UNWIND_H

#ifdef __APPLE__
#include <libunwind.h>

typedef int _Unwind_Action;
typedef void _Unwind_Context;
typedef void _Unwind_Exception;
typedef int _Unwind_Reason_Code;

#else

#include <unwind.h>

#endif

#endif

