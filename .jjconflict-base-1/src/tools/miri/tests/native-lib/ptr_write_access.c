#include <stddef.h>
#include <stdint.h>

// See comments in build_native_lib()
#define EXPORT __attribute__((visibility("default")))

/* Test: test_increment_int */

EXPORT void increment_int(int32_t *ptr) {
  *ptr += 1;
}

/* Test: test_init_int */

EXPORT void init_int(int32_t *ptr, int32_t val) {
  *ptr = val;
}

/* Test: test_init_array */

EXPORT void init_array(int32_t *array, size_t len, int32_t val) {
  for (size_t i = 0; i < len; i++) {
    array[i] = val;
  }
}

/* Test: test_init_static_inner */

typedef struct SyncPtr {
    int32_t *ptr;
} SyncPtr;

EXPORT void init_static_inner(const SyncPtr *s_ptr, int32_t val) {
  *(s_ptr->ptr) = val;
}

/* Tests: test_exposed, test_pass_dangling */

EXPORT void ignore_ptr(__attribute__((unused)) const int32_t *ptr) {
  return;
}

/* Test: test_expose_int */
EXPORT void expose_int(const int32_t *int_ptr, const int32_t **pptr) {
  *pptr = int_ptr;
}

/* Test: test_swap_ptr */

EXPORT void swap_ptr(const int32_t **pptr0, const int32_t **pptr1) {
  const int32_t *tmp = *pptr0;
  *pptr0 = *pptr1;
  *pptr1 = tmp;
}

/* Test: test_swap_ptr_tuple */

typedef struct Tuple {
    int32_t *ptr0;
    int32_t *ptr1;
} Tuple;

EXPORT void swap_ptr_tuple(Tuple *t_ptr) {
  int32_t *tmp = t_ptr->ptr0;
  t_ptr->ptr0 = t_ptr->ptr1;
  t_ptr->ptr1 = tmp;
}

/* Test: test_overwrite_dangling */

EXPORT void overwrite_ptr(const int32_t **pptr) {
  *pptr = NULL;
}

/* Test: test_swap_ptr_triple_dangling */

typedef struct Triple {
    int32_t *ptr0;
    int32_t *ptr1;
    int32_t *ptr2;
} Triple;

EXPORT void swap_ptr_triple_dangling(Triple *t_ptr) {
  int32_t *tmp = t_ptr->ptr0;
  t_ptr->ptr0 = t_ptr->ptr2;
  t_ptr->ptr2 = tmp;
}

EXPORT const int32_t *return_ptr(const int32_t *ptr) {
  return ptr;
}

/* Test: test_pass_ptr_as_int */

EXPORT void pass_ptr_as_int(uintptr_t ptr, int32_t set_to_val) {
  *(int32_t*)ptr = set_to_val;
}

/* Test: test_pass_ptr_via_previously_shared_mem */

int32_t** shared_place;

EXPORT void set_shared_mem(int32_t** ptr) {
  shared_place = ptr;
}

EXPORT void init_ptr_stored_in_shared_mem(int32_t val) {
  **shared_place = val;
}

/* Test: partial_init */

EXPORT void init_n(int32_t n, char* ptr) {
  for (int i=0; i<n; i++) {
    *(ptr+i) = 0;
  }
}
