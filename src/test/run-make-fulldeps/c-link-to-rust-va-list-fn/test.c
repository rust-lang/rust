// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include <stdarg.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum {
    TAG_DOUBLE,
    TAG_LONG,
    TAG_LONGLONG,
    TAG_INT,
    TAG_BYTE,
    TAG_CSTR,
    TAG_SKIP,
} tag;

typedef struct {
    tag answer_type;
    union {
        double double_precision;
        long num_long;
        long long num_longlong;
        int num_int;
        int8_t byte;
        char* cstr;
        tag skip_ty;
    } answer_data;
} answer;

#define MK_DOUBLE(n) \
    { TAG_DOUBLE, { .double_precision = n } }
#define MK_LONG(n) \
    { TAG_LONG, { .num_long = n } }
#define MK_LONGLONG(n) \
    { TAG_LONGLONG, { .num_longlong = n } }
#define MK_INT(n) \
    { TAG_INT, { .num_int = n } }
#define MK_BYTE(b) \
    { TAG_BYTE, { .byte = b } }
#define MK_CSTR(s) \
    { TAG_CSTR, { .cstr = s } }
#define MK_SKIP(ty) \
    { TAG_SKIP, { .skip_ty = TAG_ ## ty } }

extern size_t check_rust(size_t argc, const answer* answers, va_list ap);
extern size_t check_rust_copy(size_t argc, const answer* answers, va_list ap);

size_t test_check_rust(size_t argc, const answer* answers, ...) {
    size_t ret = 0;
    va_list ap;
    va_start(ap, answers);
    ret = check_rust(argc, answers, ap);
    va_end(ap);
    return ret;
}

size_t test_check_rust_copy(size_t argc, const answer* answers, ...) {
    size_t ret = 0;
    va_list ap;
    va_start(ap, answers);
    ret = check_rust_copy(argc, answers, ap);
    va_end(ap);
    return ret;
}

int main(int argc, char* argv[]) {
    answer test_alignment0[] = {MK_LONGLONG(0x01LL), MK_INT(0x02), MK_LONGLONG(0x03LL)};
    assert(test_check_rust(3, test_alignment0, 0x01LL, 0x02, 0x03LL) == 0);

    answer test_alignment1[] = {MK_INT(-1), MK_BYTE('A'), MK_BYTE('4'), MK_BYTE(';'),
                                MK_INT(0x32), MK_INT(0x10000001), MK_CSTR("Valid!")};
    assert(test_check_rust(7, test_alignment1, -1, 'A', '4', ';', 0x32, 0x10000001,
                           "Valid!") == 0);

    answer basic_answers[] = {MK_DOUBLE(3.14), MK_LONG(12l), MK_BYTE('a'),
                              MK_DOUBLE(6.28), MK_CSTR("Hello"), MK_INT(42),
                              MK_CSTR("World")};
    assert(test_check_rust(7, basic_answers, 3.14, 12l, 'a', 6.28, "Hello",
                           42, "World") == 0);

    answer copy_answers[] = { MK_SKIP(DOUBLE), MK_SKIP(INT), MK_SKIP(BYTE), MK_SKIP(CSTR),
                              MK_CSTR("Correctly skipped and copied list") };
    assert(test_check_rust_copy(5, copy_answers, 6.28, 16, 'A', "Skip Me!",
                                "Correctly skipped and copied list") == 0);
    return 0;
}
