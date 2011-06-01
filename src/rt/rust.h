#ifndef RUST_H
#define RUST_H

/*
 * Include this file after you've defined the ISO C9x stdint
 * types (size_t, uint8_t, uintptr_t, etc.)
 */

#ifdef __i386__
// 'cdecl' ABI only means anything on i386
#ifdef __WIN32__
#define CDECL __cdecl
#define FASTCALL __fastcall
#else
#define CDECL __attribute__((cdecl))
#define FASTCALL __attribute__((fastcall))
#endif
#else
#define CDECL
#define FASTCALL
#endif

/*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * c-basic-offset: 4
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 */

#endif /* RUST_H */
