#include <stdio.h>
#include <stdint.h>

// See comments in build_native_lib()
#define EXPORT __attribute__((visibility("default")))

/* Test: test_access_pointer */

EXPORT void print_pointer(const int32_t *ptr) {
  printf("printing pointer dereference from C: %d\n", *ptr);
}

/* Test: test_access_simple */

typedef struct Simple {
  int32_t field;
} Simple;

EXPORT int32_t access_simple(const Simple *s_ptr) {
  return s_ptr->field;
}

/* Test: test_access_nested */

typedef struct Nested {
  int32_t value;
  struct Nested *next;
} Nested;

// Returns the innermost/last value of a Nested pointer chain.
EXPORT int32_t access_nested(const Nested *n_ptr) {
  // Edge case: `n_ptr == NULL` (i.e. first Nested is None).
  if (!n_ptr) { return 0; }

  while (n_ptr->next) {
    n_ptr = n_ptr->next;
  }

  return n_ptr->value;
}

/* Test: test_access_static */

typedef struct Static {
    int32_t value;
    struct Static *recurse;
} Static;

EXPORT int32_t access_static(const Static *s_ptr) {
  return s_ptr->recurse->recurse->value;
}

/* Test: unexposed_reachable_alloc */

EXPORT uintptr_t do_one_deref(const int32_t ***ptr) {
  return (uintptr_t)*ptr;
}
