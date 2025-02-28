#include <stddef.h>
#include <stdint.h>

// See comments in build_native_lib()
#define EXPORT __attribute__((visibility("default")))

/* Test: test_increment_int */

EXPORT void increment_int(int *ptr) {
  *ptr += 1;
}

/* Test: test_init_int */

EXPORT void init_int(int *ptr, int val) {
  *ptr = val;
}

/* Test: test_init_array */

EXPORT void init_array(int *array, size_t len, int val) {
  for (size_t i = 0; i < len; i++) {
    array[i] = val;
  }
}

/* Test: test_init_static_inner */

typedef struct SyncPtr {
    int *ptr;
} SyncPtr;

EXPORT void init_static_inner(const SyncPtr *s_ptr, int val) {
  *(s_ptr->ptr) = val;
}

/* Tests: test_exposed, test_pass_dangling */

EXPORT void ignore_ptr(__attribute__((unused)) const int *ptr) {
  return;
}

/* Test: test_expose_int */
EXPORT void expose_int(const int *int_ptr, const int **pptr) {
  *pptr = int_ptr;
}

/* Test: test_swap_ptr */

EXPORT void swap_ptr(const int **pptr0, const int **pptr1) {
  const int *tmp = *pptr0;
  *pptr0 = *pptr1;
  *pptr1 = tmp;
}

/* Test: test_swap_ptr_tuple */

typedef struct Tuple {
    int *ptr0;
    int *ptr1;
} Tuple;

EXPORT void swap_ptr_tuple(Tuple *t_ptr) {
  int *tmp = t_ptr->ptr0;
  t_ptr->ptr0 = t_ptr->ptr1;
  t_ptr->ptr1 = tmp;
}

/* Test: test_overwrite_dangling */

EXPORT void overwrite_ptr(const int **pptr) {
  *pptr = NULL;
}

/* Test: test_swap_ptr_triple_dangling */

typedef struct Triple {
    int *ptr0;
    int *ptr1;
    int *ptr2;
} Triple;

EXPORT void swap_ptr_triple_dangling(Triple *t_ptr) {
  int *tmp = t_ptr->ptr0;
  t_ptr->ptr0 = t_ptr->ptr2;
  t_ptr->ptr2 = tmp;
}

EXPORT const int *return_ptr(const int *ptr) {
  return ptr;
}

/* Test: test_pass_ptr_as_int */

EXPORT void pass_ptr_as_int(uintptr_t ptr, int set_to_val) {
  *(int*)ptr = set_to_val;
}

/* Test: test_pass_ptr_via_previously_shared_mem */

int** shared_place;

EXPORT void set_shared_mem(int** ptr) {
  shared_place = ptr;
}

EXPORT void init_ptr_stored_in_shared_mem(int val) {
  **shared_place = val;
}
