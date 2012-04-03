
#ifndef RUST_UPCALL_H
#define RUST_UPCALL_H

// Upcalls used from C code on occasion:

extern "C" CDECL void upcall_shared_free(void* ptr);

#endif
