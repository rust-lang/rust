
#include "rust_internal.h"

/* Native builtins. */

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
    char cbuf[BUF_BYTES];
    char *buf = strerror_r(errno, cbuf, sizeof(cbuf));
    if (!buf) {
        task->fail(1);
        return NULL;
    }
#else
    char buf[BUF_BYTES];
    int err = strerror_r(errno, buf, sizeof(buf));
    if (err) {
        task->fail(1);
        return NULL;
    }
#endif
    size_t fill = strlen(buf) + 1;
    size_t alloc = next_power_of_two(sizeof(rust_str) + fill);
    void *mem = dom->malloc(alloc, memory_region::LOCAL);
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

extern "C" CDECL intptr_t
refcount(rust_task *task, type_desc *t, intptr_t *v) {

    if (*v == CONST_REFCOUNT)
        return CONST_REFCOUNT;

    // Passed-in value has refcount 1 too high
    // because it was ref'ed while making the call.
    return (*v) - 1;
}

extern "C" CDECL void
do_gc(rust_task *task) {
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
    task->log(rust_log::MEM | rust_log::STDLIB,
              "vec_alloc %" PRIdPTR " elements of size %" PRIdPTR,
              n_elts, elem_t->size);
    size_t fill = n_elts * elem_t->size;
    size_t alloc = next_power_of_two(sizeof(rust_vec) + fill);
    void *mem = task->malloc(alloc, t->is_stateful ? t : NULL);
    if (!mem) {
        task->fail(4);
        return NULL;
    }
    rust_vec *vec = new (mem) rust_vec(dom, alloc, 0, NULL);
    return vec;
}

extern "C" CDECL void *
vec_buf(rust_task *task, type_desc *ty, rust_vec *v, size_t offset)
{
    return (void *)&v->data[ty->size * offset];
}

extern "C" CDECL size_t
vec_len(rust_task *task, type_desc *ty, rust_vec *v)
{
    return v->fill / ty->size;
}

extern "C" CDECL void
vec_len_set(rust_task *task, type_desc *ty, rust_vec *v, size_t len)
{
    task->log(rust_log::STDLIB,
              "vec_len_set(0x%" PRIxPTR ", %" PRIdPTR ") on vec with "
              "alloc = %" PRIdPTR
              ", fill = %" PRIdPTR
              ", len = %" PRIdPTR
              ".  New fill is %" PRIdPTR,
              v, len, v->alloc, v->fill, v->fill / ty->size, len * ty->size);
    v->fill = len * ty->size;
}

extern "C" CDECL void
vec_print_debug_info(rust_task *task, type_desc *ty, rust_vec *v)
{
    task->log(rust_log::STDLIB,
              "vec_print_debug_info(0x%" PRIxPTR ")"
              " with tydesc 0x%" PRIxPTR
              " (size = %" PRIdPTR ", align = %" PRIdPTR ")"
              " alloc = %" PRIdPTR ", fill = %" PRIdPTR ", len = %" PRIdPTR
              " , data = ...",
              v,
              ty,
              ty->size,
              ty->align,
              v->alloc,
              v->fill,
              v->fill / ty->size);

    for (size_t i = 0; i < v->fill; ++i) {
        task->log(rust_log::STDLIB,
                  "  %" PRIdPTR ":    0x%" PRIxPTR,
                  i, v->data[i]);
    }
}

/* Helper for str_alloc and str_from_vec.  Returns NULL as failure. */
static rust_str *
str_alloc_with_data(rust_task *task,
                    size_t n_bytes,
                    size_t fill,
                    uint8_t const *d)
{
    rust_dom *dom = task->dom;
    size_t alloc = next_power_of_two(sizeof(rust_str) + n_bytes);
    void *mem = dom->malloc(alloc, memory_region::LOCAL);
    if (!mem)
        return NULL;
    rust_str *st = new (mem) rust_str(dom, alloc, fill, d);
    return st;
}

extern "C" CDECL rust_str*
str_alloc(rust_task *task, size_t n_bytes)
{
    rust_str *st = str_alloc_with_data(task,
                                       n_bytes + 1,  // +1 to fit at least ""
                                       1,
                                       (uint8_t const *)"");
    if (!st) {
        task->fail(2);
        return NULL;
    }
    return st;
}

extern "C" CDECL char const *
str_buf(rust_task *task, rust_str *s)
{
    return (char const *)&s->data[0];
}

extern "C" CDECL size_t
str_byte_len(rust_task *task, rust_str *s)
{
    return s->fill - 1;  // -1 for the '\0' terminator.
}

extern "C" CDECL rust_str *
str_from_vec(rust_task *task, rust_vec *v)
{
    rust_str *st =
        str_alloc_with_data(task,
                            v->fill + 1,  // +1 to fit at least '\0'
                            v->fill,
                            v->fill ? (uint8_t const *)v->data : NULL);
    if (!st) {
        task->fail(2);
        return NULL;
    }
    st->data[st->fill++] = '\0';
    return st;
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

extern "C" CDECL void upcall_sleep(rust_task *task, size_t time_in_us);

extern "C" CDECL void
task_sleep(rust_task *task, size_t time_in_us) {
    upcall_sleep(task, time_in_us);
}

/* Debug builtins for std.dbg. */

static void
debug_tydesc_helper(rust_task *task, type_desc *t)
{
    task->log(rust_log::STDLIB,
              "  size %" PRIdPTR ", align %" PRIdPTR
              ", stateful %" PRIdPTR ", first_param 0x%" PRIxPTR,
              t->size, t->align, t->is_stateful, t->first_param);
}

extern "C" CDECL void
debug_tydesc(rust_task *task, type_desc *t)
{
    task->log(rust_log::STDLIB, "debug_tydesc");
    debug_tydesc_helper(task, t);
}

extern "C" CDECL void
debug_opaque(rust_task *task, type_desc *t, uint8_t *front)
{
    task->log(rust_log::STDLIB, "debug_opaque");
    debug_tydesc_helper(task, t);
    // FIXME may want to actually account for alignment.  `front` may not
    // indeed be the front byte of the passed-in argument.
    for (uintptr_t i = 0; i < t->size; ++front, ++i) {
        task->log(rust_log::STDLIB,
                  "  byte %" PRIdPTR ": 0x%" PRIx8, i, *front);
    }
}

struct rust_box : rc_base<rust_box> {
    // FIXME `data` could be aligned differently from the actual box body data
    uint8_t data[];
};

extern "C" CDECL void
debug_box(rust_task *task, type_desc *t, rust_box *box)
{
    task->log(rust_log::STDLIB, "debug_box(0x%" PRIxPTR ")", box);
    debug_tydesc_helper(task, t);
    task->log(rust_log::STDLIB, "  refcount %" PRIdPTR,
              box->ref_count == CONST_REFCOUNT
              ? CONST_REFCOUNT
              : box->ref_count - 1);  // -1 because we ref'ed for this call
    for (uintptr_t i = 0; i < t->size; ++i) {
        task->log(rust_log::STDLIB,
                  "  byte %" PRIdPTR ": 0x%" PRIx8, i, box->data[i]);
    }
}

struct rust_tag {
    uintptr_t discriminant;
    uint8_t variant[];
};

extern "C" CDECL void
debug_tag(rust_task *task, type_desc *t, rust_tag *tag)
{
    task->log(rust_log::STDLIB, "debug_tag");
    debug_tydesc_helper(task, t);
    task->log(rust_log::STDLIB,
              "  discriminant %" PRIdPTR, tag->discriminant);

    for (uintptr_t i = 0; i < t->size - sizeof(tag->discriminant); ++i)
        task->log(rust_log::STDLIB,
                  "  byte %" PRIdPTR ": 0x%" PRIx8, i, tag->variant[i]);
}

struct rust_obj {
    uintptr_t *vtbl;
    rust_box *body;
};

extern "C" CDECL void
debug_obj(rust_task *task, type_desc *t, rust_obj *obj,
          size_t nmethods, size_t nbytes)
{
    task->log(rust_log::STDLIB,
              "debug_obj with %" PRIdPTR " methods", nmethods);
    debug_tydesc_helper(task, t);
    task->log(rust_log::STDLIB, "  vtbl at 0x%" PRIxPTR, obj->vtbl);
    task->log(rust_log::STDLIB, "  body at 0x%" PRIxPTR, obj->body);

    for (uintptr_t *p = obj->vtbl; p < obj->vtbl + nmethods; ++p)
        task->log(rust_log::STDLIB, "  vtbl word: 0x%" PRIxPTR, *p);

    for (uintptr_t i = 0; i < nbytes; ++i)
        task->log(rust_log::STDLIB,
                  "  body byte %" PRIdPTR ": 0x%" PRIxPTR,
                  i, obj->body->data[i]);
}

struct rust_fn {
    uintptr_t *thunk;
    rust_box *closure;
};

extern "C" CDECL void
debug_fn(rust_task *task, type_desc *t, rust_fn *fn)
{
    task->log(rust_log::STDLIB, "debug_fn");
    debug_tydesc_helper(task, t);
    task->log(rust_log::STDLIB, "  thunk at 0x%" PRIxPTR, fn->thunk);
    task->log(rust_log::STDLIB, "  closure at 0x%" PRIxPTR, fn->closure);
    if (fn->closure) {
        task->log(rust_log::STDLIB, "    refcount %" PRIdPTR,
                  fn->closure->ref_count);
    }
}

extern "C" CDECL void *
debug_ptrcast(rust_task *task,
              type_desc *from_ty,
              type_desc *to_ty,
              void *ptr)
{
    task->log(rust_log::STDLIB, "debug_ptrcast from");
    debug_tydesc_helper(task, from_ty);
    task->log(rust_log::STDLIB, "to");
    debug_tydesc_helper(task, to_ty);
    return ptr;
}

extern "C" CDECL void
debug_trap(rust_task *task, rust_str *s)
{
    task->log(rust_log::STDLIB, "trapping: %s", s->data);
    // FIXME: x86-ism.
    __asm__("int3");
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
