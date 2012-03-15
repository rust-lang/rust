#pragma once

// Upcalls used from C code on occasion:

extern "C" CDECL void upcall_shared_free(void* ptr);

