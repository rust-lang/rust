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
extern size_t check_varargs_3(int fixed, ...);
extern size_t check_varargs_4(double fixed, ...);
extern size_t check_varargs_5(int fixed, ...);

extern size_t run_test_variadic();
extern size_t run_test_va_list_by_value();
extern size_t run_test_va_list_by_pointer();
extern size_t run_test_va_list_by_pointer_pointer();

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

    assert(check_varargs_3(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10) == 0);

    assert(check_varargs_4(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                           13.0) == 0);

    assert(check_varargs_5(0, 1.0, 1, 2.0, 2, 3.0, 3, 4.0, 4, 5, 5.0, 6, 6.0, 7, 7.0, 8, 8.0,
                           9, 9.0, 10, 10.0, 11, 11.0, 12, 12.0, 13, 13.0) == 0);

    assert(run_test_variadic() == 0);
    assert(run_test_va_list_by_value() == 0);
    assert(run_test_va_list_by_pointer() == 0);
    assert(run_test_va_list_by_pointer_pointer() == 0);

    return 0;
}

#define continue_if_else_end(cond) \
    do { if (!(cond)) { va_end(ap); return 0xff; } } while (0)

size_t test_variadic(int unused, ...) {
    va_list ap;
    va_start(ap, unused);

    continue_if_else_end(va_arg(ap, long long) == 1);
    continue_if_else_end(va_arg(ap, int) == 2);
    continue_if_else_end(va_arg(ap, long long) == 3);

    va_end(ap);

    return 0;
}

#define continue_if(cond) \
    do { if (!(cond)) { return 0xff; } } while (0)

size_t test_va_list_by_value(va_list ap) {
    continue_if(va_arg(ap, long long) == 1);
    continue_if(va_arg(ap, int) == 2);
    continue_if(va_arg(ap, long long) == 3);

    return 0;
}

size_t test_va_list_by_pointer(va_list *ap) {
    continue_if(va_arg(*ap, long long) == 1);
    continue_if(va_arg(*ap, int) == 2);
    continue_if(va_arg(*ap, long long) == 3);

    return 0;
}

size_t test_va_list_by_pointer_pointer(va_list **ap) {
    continue_if(va_arg(**ap, long long) == 1);
    continue_if(va_arg(**ap, int) == 2);
    continue_if(va_arg(**ap, long long) == 3);

    return 0;
}
