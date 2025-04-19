/* This is meant to override Musl's src/include/features.h
 *
 * We use a separate file here to redefine some attributes that don't work on
 * all platforms that we would like to build on.
 */

#ifndef FEATURES_H
#define FEATURES_H

/* Get the required `#include "../../include/features.h"` since we can't use
 * the relative path. The C macros need double indirection to get a usable
 * string. */
#define _stringify_inner(s) #s
#define _stringify(s) _stringify_inner(s)
#include _stringify(ROOT_INCLUDE_FEATURES)

#if defined(__APPLE__)
#define weak __attribute__((__weak__))
#define hidden __attribute__((__visibility__("hidden")))

/* We _should_ be able to define this as:
 *     _Pragma(_stringify(weak musl_ ## new = musl_ ## old))
 * However, weak symbols aren't handled correctly [1]. So we manually write
 * wrappers, which are in `alias.c`.
 *
 * [1]: https://github.com/llvm/llvm-project/issues/111321
 */
#define weak_alias(old, new) /* nothing */

#else
#define weak __attribute__((__weak__))
#define hidden __attribute__((__visibility__("hidden")))
#define weak_alias(old, new) \
	extern __typeof(old) musl_ ## new \
	__attribute__((__weak__, __alias__(_stringify(musl_ ## old))))

#endif /* defined(__APPLE__) */

#endif
