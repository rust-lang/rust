#include <stdint.h>

// See comments in build_native_lib()
#define EXPORT __attribute__((visibility("default")))

/* Test: fail/pass_struct_expose_only_range */

typedef struct HasPointer {
  uint8_t *ptr;
} HasPointer;

EXPORT uint8_t access_struct_ptr(const HasPointer s) {
  return *s.ptr;
}

/* Test: test_pass_struct */

typedef struct PassMe {
    int32_t value;
    int64_t other_value;
} PassMe;

EXPORT int64_t pass_struct(const PassMe pass_me) {
  return pass_me.value + pass_me.other_value;
}

/* Test: test_pass_struct_complex */

typedef struct Part1 {
    uint16_t high;
    uint16_t low;
} Part1;

typedef struct Part2 {
    uint32_t bits;
} Part2;

typedef struct ComplexStruct {
    Part1 part_1;
    Part2 part_2;
    uint32_t part_3;
} ComplexStruct;

EXPORT int32_t pass_struct_complex(const ComplexStruct complex, uint16_t high, uint16_t low, uint32_t bits) {
  if (complex.part_1.high == high && complex.part_1.low == low
      && complex.part_2.bits == bits
      && complex.part_3 == bits)
    return 0;
  else {
    return 1;
  }
}
