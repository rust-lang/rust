#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

struct BoolAndU32
{
    bool a;
    uint32_t b;
};

#ifdef _MSC_VER
__declspec(align(16))
struct TwoU64s
{
    uint64_t a;
    uint64_t b;
};
#else
struct __attribute__((aligned(16))) TwoU64s
{
    uint64_t a;
    uint64_t b;
};
#endif

struct WrappedU64s
{
    struct TwoU64s a;
};

#ifdef _MSC_VER
__declspec(align(1))
struct LowerAlign
{
    uint64_t a;
    uint64_t b;
};
#else
struct __attribute__((aligned(1))) LowerAlign
{
    uint64_t a;
    uint64_t b;
};
#endif

#pragma pack(push, 1)
struct Packed
{
    uint64_t a;
    uint64_t b;
};
#pragma pack(pop)

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
    struct WrappedU64s j,
    void *k,
    struct LowerAlign l,
    void *m,
    struct Packed n,
    const char *o)
{
    assert(!a);
    assert(!b);
    assert(!c);
    assert(d == 42);
    assert(e);
    assert(f.a);
    assert(f.b == 1337);
    assert(!g);
    assert(h.a == 1);
    assert(h.b == 2);
    assert(!i);
    assert(j.a.a == 3);
    assert(j.a.b == 4);
    assert(!k);
    assert(l.a == 5);
    assert(l.b == 6);
    assert(!m);
    assert(n.a == 7);
    assert(n.b == 8);
    assert(strcmp(o, "Hello world") == 0);
    return 0;
}
