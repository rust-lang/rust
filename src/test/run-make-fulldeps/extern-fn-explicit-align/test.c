#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

struct TwoU64s
{
    uint64_t a;
    uint64_t b;
} __attribute__((aligned(16)));

struct BoolAndU32
{
    bool a;
    uint32_t b;
};

int32_t many_args(
    void *a,
    void *b,
    const char *c,
    uint64_t d,
    bool e,
    struct BoolAndU32 f,
    void *g,
    struct TwoU64s h,
    void *i,
    void *j,
    void *k,
    void *l,
    const char *m)
{
    assert(strcmp(m, "Hello world") == 0);
    return 0;
}
