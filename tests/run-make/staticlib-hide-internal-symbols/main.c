#include <stdint.h>

extern int my_add(int a, int b);
extern uint64_t my_hash_lookup(uint64_t key);
extern int call_internal(void);
extern int my_safe_div(int a, int b);

int main() {
    if (my_add(10, 20) != 30)
        return 1;
    if (my_hash_lookup(5) != (uint64_t)5 * 2654435761ULL)
        return 1;
    if (call_internal() != 42)
        return 1;
    if (my_safe_div(100, 5) != 20)
        return 1;
    if (my_safe_div(100, 0) != -1)
        return 1;
    return 0;
}
