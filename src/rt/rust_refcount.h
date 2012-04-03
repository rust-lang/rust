
#ifndef RUST_REFCOUNT_H
#define RUST_REFCOUNT_H

#include "sync/sync.h"

// Refcounting defines
typedef unsigned long ref_cnt_t;

#define RUST_REFCOUNTED(T) \
  RUST_REFCOUNTED_WITH_DTOR(T, delete (T*)this)

#define RUST_REFCOUNTED_WITH_DTOR(T, dtor)      \
  intptr_t ref_count;      \
  void ref() { ++ref_count; } \
  void deref() { if (--ref_count == 0) { dtor; } }

#define RUST_ATOMIC_REFCOUNT()                                             \
private:                                                                   \
   intptr_t ref_count;                                                     \
public:                                                                    \
   void ref() {                                                            \
       intptr_t old = sync::increment(ref_count);                          \
       assert(old > 0);                                                    \
   }                                                                       \
   void deref() { if(0 == sync::decrement(ref_count)) { delete_this(); } } \
   intptr_t get_ref_count() { return sync::read(ref_count); }

#endif
