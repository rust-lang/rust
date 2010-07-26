
#include "rust_internal.h"

/* Native builtins. */
extern "C" CDECL rust_str*
str_alloc(rust_task *task, size_t n_bytes)
{
    rust_dom *dom = task->dom;
    size_t alloc = next_power_of_two(sizeof(rust_str) + n_bytes);
    void *mem = dom->malloc(alloc);
    if (!mem) {
        task->fail(2);
        return NULL;
    }
    rust_str *st = new (mem) rust_str(dom, alloc, 1, (uint8_t const *)"");
    return st;
}

extern "C" CDECL rust_str*
last_os_error(rust_task *task) {
    rust_dom *dom = task->dom;
    task->log(rust_log::TASK, "last_os_error()");

#if defined(__WIN32__)
    LPTSTR buf;
    DWORD err = GetLastError();
    DWORD res = FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                              FORMAT_MESSAGE_FROM_SYSTEM |
                              FORMAT_MESSAGE_IGNORE_INSERTS,
                              NULL, err,
                              MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                              (LPTSTR) &buf, 0, NULL);
    if (!res) {
        task->fail(1);
        return NULL;
    }
#elif defined(_GNU_SOURCE)
    char cbuf[1024];
    char *buf = strerror_r(errno, cbuf, sizeof(cbuf));
    if (!buf) {
        task->fail(1);
        return NULL;
    }
#else
    char buf[1024];
    int err = strerror_r(errno, buf, sizeof(buf));
    if (err) {
        task->fail(1);
        return NULL;
    }
#endif
    size_t fill = strlen(buf) + 1;
    size_t alloc = next_power_of_two(sizeof(rust_str) + fill);
    void *mem = dom->malloc(alloc);
    if (!mem) {
        task->fail(1);
        return NULL;
    }
    rust_str *st = new (mem) rust_str(dom, alloc, fill, (const uint8_t *)buf);

#ifdef __WIN32__
    LocalFree((HLOCAL)buf);
#endif
    return st;
}

extern "C" CDECL size_t
size_of(rust_task *task, type_desc *t) {
  return t->size;
}

extern "C" CDECL size_t
align_of(rust_task *task, type_desc *t) {
  return t->align;
}

extern "C" CDECL size_t
refcount(rust_task *task, type_desc *t, size_t *v) {
    // Passed-in value has refcount 1 too high
    // because it was ref'ed while making the call.
    return (*v) - 1;
}

extern "C" CDECL void
gc(rust_task *task) {
    task->gc(1);
}

extern "C" CDECL void
unsupervise(rust_task *task) {
    task->unsupervise();
}

extern "C" CDECL rust_vec*
vec_alloc(rust_task *task, type_desc *t, type_desc *elem_t, size_t n_elts)
{
    rust_dom *dom = task->dom;
    task->log(rust_log::MEM,
              "vec_alloc %" PRIdPTR " elements of size %" PRIdPTR,
              n_elts, elem_t->size);
    size_t fill = n_elts * elem_t->size;
    size_t alloc = next_power_of_two(sizeof(rust_vec) + fill);
    void *mem = task->malloc(alloc, t->is_stateful ? t : NULL);
    if (!mem) {
        task->fail(3);
        return NULL;
    }
    rust_vec *vec = new (mem) rust_vec(dom, alloc, 0, NULL);
    return vec;
}

extern "C" CDECL char const *
str_buf(rust_task *task, rust_str *s)
{
    return (char const *)&s->data[0];
}

extern "C" CDECL void *
vec_buf(rust_task *task, type_desc *ty, rust_vec *v)
{
    return (void *)&v->data[0];
}

extern "C" CDECL size_t
vec_len(rust_task *task, type_desc *ty, rust_vec *v)
{
    return v->fill / ty->size;
}

extern "C" CDECL void *
rand_new(rust_task *task)
{
    rust_dom *dom = task->dom;
    randctx *rctx = (randctx *) task->malloc(sizeof(randctx));
    if (!rctx) {
        task->fail(1);
        return NULL;
    }
    isaac_init(dom, rctx);
    return rctx;
}

extern "C" CDECL size_t
rand_next(rust_task *task, randctx *rctx)
{
    return rand(rctx);
}

extern "C" CDECL void
rand_free(rust_task *task, randctx *rctx)
{
    task->free(rctx);
}

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
