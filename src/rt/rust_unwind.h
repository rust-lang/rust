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

