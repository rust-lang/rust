#pragma once

// Upcalls used from C code on occasion:

extern "C" CDECL void upcall_shared_free(void* ptr);
extern "C" CDECL void upcall_free_shared_type_desc(type_desc *td);

