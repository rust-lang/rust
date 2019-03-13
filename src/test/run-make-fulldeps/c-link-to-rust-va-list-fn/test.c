#include <stdarg.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

extern size_t check_list_0(va_list ap);
extern size_t check_list_1(va_list ap);
extern size_t check_list_2(va_list ap);
extern size_t check_list_copy_0(va_list ap);
extern size_t check_varargs_0(int fixed, ...);
extern size_t check_varargs_1(int fixed, ...);
extern size_t check_varargs_2(int fixed, ...);

int test_rust(size_t (*fn)(va_list), ...) {
    size_t ret = 0;
    va_list ap;
    va_start(ap, fn);
    ret = fn(ap);
    va_end(ap);
    return ret;
}

int main(int argc, char* argv[]) {
    assert(test_rust(check_list_0, 0x01LL, 0x02, 0x03LL) == 0);

    assert(test_rust(check_list_1, -1, 'A', '4', ';', 0x32, 0x10000001, "Valid!") == 0);

    assert(test_rust(check_list_2, 3.14, 12l, 'a', 6.28, "Hello", 42, "World") == 0);

    assert(test_rust(check_list_copy_0, 6.28, 16, 'A', "Skip Me!", "Correct") == 0);

    assert(check_varargs_0(0, 42, "Hello, World!") == 0);

    assert(check_varargs_1(0, 3.14, 12l, 'A', 0x1LL) == 0);

    assert(check_varargs_2(0, "All", "of", "these", "are", "ignored", ".") == 0);

    return 0;
}
