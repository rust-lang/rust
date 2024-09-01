#include <stdio.h>

// See comments in build_native_lib()
#define EXPORT __attribute__((visibility("default")))

/* Test: test_pointer */

EXPORT void print_pointer(const int *ptr) {
  printf("printing pointer dereference from C: %d\n", *ptr);
}

/* Test: test_simple */

typedef struct Simple {
  int field;
} Simple;

EXPORT int access_simple(const Simple *s_ptr) {
  return s_ptr->field;
}

/* Test: test_nested */

typedef struct Nested {
  int value;
  struct Nested *next;
} Nested;

// Returns the innermost/last value of a Nested pointer chain.
EXPORT int access_nested(const Nested *n_ptr) {
  // Edge case: `n_ptr == NULL` (i.e. first Nested is None).
  if (!n_ptr) { return 0; }

  while (n_ptr->next) {
    n_ptr = n_ptr->next;
  }

  return n_ptr->value;
}

/* Test: test_static */

typedef struct Static {
    int value;
    struct Static *recurse;
} Static;

EXPORT int access_static(const Static *s_ptr) {
  return s_ptr->recurse->recurse->value;
}
