
#include "rust_internal.h"

#if !defined(__WIN32__)
#include <sys/time.h>
#endif

/* Native builtins. */

extern "C" CDECL rust_str*
last_os_error(rust_task *task) {
    rust_scheduler *sched = task->sched;
    LOG(task, task, "last_os_error()");

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
    void *mem = task->malloc(alloc);
    if (!mem) {
        task->fail(1);
        return NULL;
    }
    rust_str *st = new (mem) rust_str(sched, alloc, fill,
                                      (const uint8_t *)buf);

#ifdef __WIN32__
    LocalFree((HLOCAL)buf);
#endif
    return st;
}

extern "C" CDECL rust_str *
rust_getcwd(rust_task *task) {
    rust_scheduler *sched = task->sched;
    LOG(task, task, "rust_getcwd()");

    char cbuf[BUF_BYTES];

#if defined(__WIN32__)
    if (!_getcwd(cbuf, sizeof(cbuf))) {
#else
        if (!getcwd(cbuf, sizeof(cbuf))) {
#endif
        task->fail(1);
        return NULL;
    }

    size_t fill = strlen(cbuf) + 1;
    size_t alloc = next_power_of_two(sizeof(rust_str) + fill);
    void *mem = task->malloc(alloc);
    if (!mem) {
        task->fail(1);
        return NULL;
    }

    rust_str *st;
    st = new (mem) rust_str(sched, alloc, fill, (const uint8_t *)cbuf);

    return st;    
}

extern "C" CDECL
void squareroot(rust_task *task, double *input, double *output) {
    *output = sqrt(*input);
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
    rust_scheduler *sched = task->sched;
    LOG(task, mem, "vec_alloc %" PRIdPTR " elements of size %" PRIdPTR,
        n_elts, elem_t->size);
    size_t fill = n_elts * elem_t->size;
    size_t alloc = next_power_of_two(sizeof(rust_vec) + fill);
    void *mem = task->malloc(alloc, t->is_stateful ? t : NULL);
    if (!mem) {
        task->fail(4);
        return NULL;
    }
    rust_vec *vec = new (mem) rust_vec(sched, alloc, 0, NULL);
    return vec;
}

extern "C" CDECL rust_vec*
vec_alloc_mut(rust_task *task, type_desc *t, type_desc *elem_t, size_t n_elts)
{
    return vec_alloc(task, t, elem_t, n_elts);
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
    LOG(task, stdlib, "vec_len_set(0x%" PRIxPTR ", %" PRIdPTR ") on vec with "
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
    LOG(task, stdlib,
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
        LOG(task, stdlib, "  %" PRIdPTR ":    0x%" PRIxPTR, i, v->data[i]);
    }
}

/* Helper for str_alloc and str_from_vec.  Returns NULL as failure. */
static rust_vec*
vec_alloc_with_data(rust_task *task,
                    size_t n_elts,
                    size_t fill,
                    size_t elt_size,
                    void *d)
{
    rust_scheduler *sched = task->sched;
    size_t alloc = next_power_of_two(sizeof(rust_vec) + (n_elts * elt_size));
    void *mem = task->malloc(alloc);
    if (!mem) return NULL;
    return new (mem) rust_vec(sched, alloc, fill * elt_size, (uint8_t*)d);
}

extern "C" CDECL rust_vec*
vec_from_vbuf(rust_task *task, type_desc *ty, void *vbuf, size_t n_elts)
{
    return vec_alloc_with_data(task, n_elts, n_elts * ty->size, ty->size,
                               vbuf);
}

extern "C" CDECL rust_vec*
unsafe_vec_to_mut(rust_task *task, type_desc *ty, rust_vec *v)
{
    if (v->ref_count != CONST_REFCOUNT) {
        v->ref();
    }
    return v;
}

extern "C" CDECL rust_str*
str_alloc(rust_task *task, size_t n_bytes)
{
    rust_str *st = vec_alloc_with_data(task,
                                       n_bytes + 1,  // +1 to fit at least ""
                                       1, 1,
                                       (void*)"");
    if (!st) {
        task->fail(2);
        return NULL;
    }
    return st;
}

extern "C" CDECL rust_str*
str_push_byte(rust_task* task, rust_str* v, size_t byte)
{
    size_t fill = v->fill;
    size_t alloc = next_power_of_two(sizeof(rust_vec) + fill + 1);
    if (v->ref_count > 1 || v->alloc < alloc) {
        v = vec_alloc_with_data(task, fill + 1, fill, 1, (void*)&v->data[0]);
        if (!v) {
            task->fail(2);
            return NULL;
        }
    }
    else if (v->ref_count != CONST_REFCOUNT) {
        v->ref();
    }
    v->data[fill-1] = (char)byte;
    v->data[fill] = '\0';
    v->fill++;
    return v;
}

extern "C" CDECL rust_str*
str_slice(rust_task* task, rust_str* v, size_t begin, size_t end)
{
    size_t len = end - begin;
    rust_str *st =
        vec_alloc_with_data(task,
                            len + 1, // +1 to fit at least '\0'
                            len,
                            1,
                            len ? v->data + begin : NULL);
    if (!st) {
        task->fail(2);
        return NULL;
    }
    st->data[st->fill++] = '\0';
    return st;
}

extern "C" CDECL char const *
str_buf(rust_task *task, rust_str *s)
{
    return (char const *)&s->data[0];
}

extern "C" CDECL rust_vec*
str_vec(rust_task *task, rust_str *s)
{
    // FIXME: this should just upref s and return it, but we
    // accidentally made too much of the language and runtime know
    // and care about the difference between str and vec (trailing null);
    // eliminate these differences and then rewrite this back to just
    // the following:
    //
    // if (s->ref_count != CONST_REFCOUNT)
    //    s->ref();
    // return s;

    rust_str *v =
        vec_alloc_with_data(task,
                            s->fill - 1,
                            s->fill - 1,
                            1,
                            (s->fill - 1) ? (void*)s->data : NULL);
    if (!v) {
        task->fail(2);
        return NULL;
    }
    return v;
}

extern "C" CDECL size_t
str_byte_len(rust_task *task, rust_str *s)
{
    return s->fill - 1;  // -1 for the '\0' terminator.
}

extern "C" CDECL rust_str *
str_from_ivec(rust_task *task, rust_ivec *v)
{
    uintptr_t fill = v->fill ? v->fill : v->payload.ptr->fill;
    void *data = v->fill ? v->payload.data : v->payload.ptr->data;

    rust_str *st =
        vec_alloc_with_data(task,
                            fill + 1,   // +1 to fit at least '\0'
                            fill,
                            1,
                            fill ? data : NULL);
    if (!st) {
        task->fail(2);
        return NULL;
    }
    st->data[st->fill++] = '\0';
    return st;
}

extern "C" CDECL rust_str *
str_from_vec(rust_task *task, rust_vec *v)
{
    rust_str *st =
        vec_alloc_with_data(task,
                            v->fill + 1,  // +1 to fit at least '\0'
                            v->fill,
                            1,
                            v->fill ? (void*)v->data : NULL);
    if (!st) {
        task->fail(2);
        return NULL;
    }
    st->data[st->fill++] = '\0';
    return st;
}

extern "C" CDECL rust_str *
str_from_cstr(rust_task *task, char *sbuf)
{
    size_t len = strlen(sbuf) + 1;
    rust_str *st = vec_alloc_with_data(task, len, len, 1, sbuf);
    if (!st) {
        task->fail(2);
        return NULL;
    }
    return st;
}

extern "C" CDECL rust_str *
str_from_buf(rust_task *task, char *buf, unsigned int len) {
    rust_str *st = vec_alloc_with_data(task, len + 1, len, 1, buf);
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
    rust_scheduler *sched = task->sched;
    randctx *rctx = (randctx *) task->malloc(sizeof(randctx));
    if (!rctx) {
        task->fail(1);
        return NULL;
    }
    isaac_init(sched, rctx);
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

extern "C" CDECL void
task_yield(rust_task *task) {
    task->yield(1);
}

extern "C" CDECL void
task_join(rust_task *task, rust_task *join_task) {
    // If the other task is already dying, we don't have to wait for it.
    join_task->lock.lock();
    if (join_task->dead() == false) {
        join_task->tasks_waiting_to_join.push(task);
        task->block(join_task, "joining local task");
        join_task->lock.unlock();
        task->yield(2);
    }
    else {
        join_task->lock.unlock();
    }
}

/* Debug builtins for std.dbg. */

static void
debug_tydesc_helper(rust_task *task, type_desc *t)
{
    LOG(task, stdlib, "  size %" PRIdPTR ", align %" PRIdPTR
        ", stateful %" PRIdPTR ", first_param 0x%" PRIxPTR,
        t->size, t->align, t->is_stateful, t->first_param);
}

extern "C" CDECL void
debug_tydesc(rust_task *task, type_desc *t)
{
    LOG(task, stdlib, "debug_tydesc");
    debug_tydesc_helper(task, t);
}

extern "C" CDECL void
debug_opaque(rust_task *task, type_desc *t, uint8_t *front)
{
    LOG(task, stdlib, "debug_opaque");
    debug_tydesc_helper(task, t);
    // FIXME may want to actually account for alignment.  `front` may not
    // indeed be the front byte of the passed-in argument.
    for (uintptr_t i = 0; i < t->size; ++front, ++i) {
        LOG(task, stdlib, "  byte %" PRIdPTR ": 0x%" PRIx8, i, *front);
    }
}

struct rust_box {
    RUST_REFCOUNTED(rust_box)

    // FIXME `data` could be aligned differently from the actual box body data
    uint8_t data[];
};

extern "C" CDECL void
debug_box(rust_task *task, type_desc *t, rust_box *box)
{
    LOG(task, stdlib, "debug_box(0x%" PRIxPTR ")", box);
    debug_tydesc_helper(task, t);
    LOG(task, stdlib, "  refcount %" PRIdPTR,
        box->ref_count == CONST_REFCOUNT
        ? CONST_REFCOUNT
        : box->ref_count - 1);  // -1 because we ref'ed for this call
    for (uintptr_t i = 0; i < t->size; ++i) {
        LOG(task, stdlib, "  byte %" PRIdPTR ": 0x%" PRIx8, i, box->data[i]);
    }
}

struct rust_tag {
    uintptr_t discriminant;
    uint8_t variant[];
};

extern "C" CDECL void
debug_tag(rust_task *task, type_desc *t, rust_tag *tag)
{
    LOG(task, stdlib, "debug_tag");
    debug_tydesc_helper(task, t);
    LOG(task, stdlib, "  discriminant %" PRIdPTR, tag->discriminant);

    for (uintptr_t i = 0; i < t->size - sizeof(tag->discriminant); ++i)
        LOG(task, stdlib, "  byte %" PRIdPTR ": 0x%" PRIx8, i,
            tag->variant[i]);
}

struct rust_obj {
    uintptr_t *vtbl;
    rust_box *body;
};

extern "C" CDECL void
debug_obj(rust_task *task, type_desc *t, rust_obj *obj,
          size_t nmethods, size_t nbytes)
{
    LOG(task, stdlib, "debug_obj with %" PRIdPTR " methods", nmethods);
    debug_tydesc_helper(task, t);
    LOG(task, stdlib, "  vtbl at 0x%" PRIxPTR, obj->vtbl);
    LOG(task, stdlib, "  body at 0x%" PRIxPTR, obj->body);

    for (uintptr_t *p = obj->vtbl; p < obj->vtbl + nmethods; ++p)
        LOG(task, stdlib, "  vtbl word: 0x%" PRIxPTR, *p);

    for (uintptr_t i = 0; i < nbytes; ++i)
        LOG(task, stdlib, "  body byte %" PRIdPTR ": 0x%" PRIxPTR,
            i, obj->body->data[i]);
}

struct rust_fn {
    uintptr_t *thunk;
    rust_box *closure;
};

extern "C" CDECL void
debug_fn(rust_task *task, type_desc *t, rust_fn *fn)
{
    LOG(task, stdlib, "debug_fn");
    debug_tydesc_helper(task, t);
    LOG(task, stdlib, "  thunk at 0x%" PRIxPTR, fn->thunk);
    LOG(task, stdlib, "  closure at 0x%" PRIxPTR, fn->closure);
    if (fn->closure) {
        LOG(task, stdlib, "    refcount %" PRIdPTR, fn->closure->ref_count);
    }
}

extern "C" CDECL void *
debug_ptrcast(rust_task *task,
              type_desc *from_ty,
              type_desc *to_ty,
              void *ptr)
{
    LOG(task, stdlib, "debug_ptrcast from");
    debug_tydesc_helper(task, from_ty);
    LOG(task, stdlib, "to");
    debug_tydesc_helper(task, to_ty);
    return ptr;
}

extern "C" CDECL void
debug_trap(rust_task *task, rust_str *s)
{
    LOG(task, stdlib, "trapping: %s", s->data);
    // FIXME: x86-ism.
    __asm__("int3");
}

rust_str* c_str_to_rust(rust_task *task, char const *str) {
    size_t len = strlen(str) + 1;
    return vec_alloc_with_data(task, len, len, 1, (void*)str);
}

extern "C" CDECL rust_vec*
rust_list_files(rust_task *task, rust_str *path) {
    array_list<rust_str*> strings;
#if defined(__WIN32__)
    WIN32_FIND_DATA FindFileData;
    HANDLE hFind = FindFirstFile((char*)path->data, &FindFileData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            strings.push(c_str_to_rust(task, FindFileData.cFileName));
        } while (FindNextFile(hFind, &FindFileData));
        FindClose(hFind);
    }
#else
  DIR *dirp = opendir((char*)path->data);
  if (dirp) {
      struct dirent *dp;
      while ((dp = readdir(dirp)))
          strings.push(c_str_to_rust(task, dp->d_name));
      closedir(dirp);
  }
#endif
  return vec_alloc_with_data(task, strings.size(), strings.size(),
                             sizeof(rust_str*), strings.data());
}

extern "C" CDECL rust_box*
rust_list_files_ivec(rust_task *task, rust_str *path) {
    array_list<rust_str*> strings;
#if defined(__WIN32__)
    WIN32_FIND_DATA FindFileData;
    HANDLE hFind = FindFirstFile((char*)path->data, &FindFileData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            strings.push(c_str_to_rust(task, FindFileData.cFileName));
        } while (FindNextFile(hFind, &FindFileData));
        FindClose(hFind);
    }
#else
  DIR *dirp = opendir((char*)path->data);
  if (dirp) {
      struct dirent *dp;
      while ((dp = readdir(dirp)))
          strings.push(c_str_to_rust(task, dp->d_name));
      closedir(dirp);
  }
#endif
  rust_box *box = (rust_box *)task->malloc(sizeof(rust_box) +
                                           sizeof(rust_ivec));
  box->ref_count = 1;
  rust_ivec *iv = (rust_ivec *)&box->data;
  iv->fill = 0;

  size_t alloc_sz = sizeof(rust_str *) * strings.size();
  iv->alloc = alloc_sz;
  iv->payload.ptr = (rust_ivec_heap *)
      task->kernel->malloc(alloc_sz + sizeof(size_t));
  iv->payload.ptr->fill = alloc_sz;
  memcpy(&iv->payload.ptr->data, strings.data(), alloc_sz);
  return box;
}

#if defined(__WIN32__)
extern "C" CDECL rust_str *
rust_dirent_filename(rust_task *task, void* ent) {
    return NULL;
}
#else
extern "C" CDECL rust_str *
rust_dirent_filename(rust_task *task, dirent* ent) {
    return c_str_to_rust(task, ent->d_name);
}
#endif

extern "C" CDECL int
rust_file_is_dir(rust_task *task, rust_str *path) {
    struct stat buf;
    stat((char*)path->data, &buf);
    return S_ISDIR(buf.st_mode);
}

extern "C" CDECL FILE* rust_get_stdin() {return stdin;}
extern "C" CDECL FILE* rust_get_stdout() {return stdout;}

extern "C" CDECL int
rust_ptr_eq(rust_task *task, type_desc *t, rust_box *a, rust_box *b) {
    return a == b;
}

#if defined(__WIN32__)
extern "C" CDECL void
get_time(rust_task *task, uint32_t *sec, uint32_t *usec) {
    SYSTEMTIME systemTime;
    FILETIME fileTime;
    GetSystemTime(&systemTime);
    if (!SystemTimeToFileTime(&systemTime, &fileTime)) {
        task->fail(1);
        return;
    }

    // FIXME: This is probably completely wrong.
    *sec = fileTime.dwHighDateTime;
    *usec = fileTime.dwLowDateTime;
}
#else
extern "C" CDECL void
get_time(rust_task *task, uint32_t *sec, uint32_t *usec) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    *sec = tv.tv_sec;
    *usec = tv.tv_usec;
}
#endif

extern "C" CDECL void
nano_time(rust_task *task, uint64_t *ns) {
    timer t;
    *ns = t.time_ns();
}

/**
 * Preallocates the exact number of bytes in the given interior vector.
 */
extern "C" CDECL void
ivec_reserve(rust_task *task, type_desc *ty, rust_ivec *v, size_t n_elems)
{
    size_t new_alloc = n_elems * ty->size;
    if (new_alloc <= v->alloc)
        return;     // Already big enough.

    rust_ivec_heap *heap_part;
    if (v->fill || !v->payload.ptr) {
        // On stack; spill to heap.
        heap_part = (rust_ivec_heap *)task->malloc(new_alloc +
                                                   sizeof(size_t));
        heap_part->fill = v->fill;
        memcpy(&heap_part->data, v->payload.data, v->fill);

        v->fill = 0;
        v->payload.ptr = heap_part;
    } else {
        // On heap; resize.
        heap_part = (rust_ivec_heap *)
            task->realloc(v->payload.ptr,
                          new_alloc + sizeof(size_t));
        v->payload.ptr = heap_part;
    }

    v->alloc = new_alloc;
}

/**
 * Preallocates the exact number of bytes in the given interior vector.
 */
extern "C" CDECL void
ivec_reserve_shared(rust_task *task, type_desc *ty, rust_ivec *v,
                    size_t n_elems)
{
    size_t new_alloc = n_elems * ty->size;
    if (new_alloc <= v->alloc)
        return;     // Already big enough.

    rust_ivec_heap *heap_part;
    if (v->fill || !v->payload.ptr) {
        // On stack; spill to heap.
        heap_part = (rust_ivec_heap *)task->kernel->malloc(new_alloc +
                                                           sizeof(size_t));
        heap_part->fill = v->fill;
        memcpy(&heap_part->data, v->payload.data, v->fill);

        v->fill = 0;
        v->payload.ptr = heap_part;
    } else {
        // On heap; resize.
        heap_part = (rust_ivec_heap *)task->kernel->realloc(v->payload.ptr,
                                                new_alloc + sizeof(size_t));
        v->payload.ptr = heap_part;
    }

    v->alloc = new_alloc;
}

/**
 * Returns true if the given vector is on the heap and false if it's on the
 * stack.
 */
extern "C" CDECL bool
ivec_on_heap(rust_task *task, type_desc *ty, rust_ivec *v)
{
    return !v->fill && v->payload.ptr;
}

/**
 * Returns an unsafe pointer to the data part of an interior vector.
 */
extern "C" CDECL void *
ivec_to_ptr(rust_task *task, type_desc *ty, rust_ivec *v)
{
    return v->fill ? v->payload.data : v->payload.ptr->data;
}

static size_t
get_ivec_size(rust_ivec *v)
{
    if (v->fill)
        return v->fill;
    if (v->payload.ptr)
        return v->payload.ptr->fill;
    return 0;
}

/**
 * Copies elements in an unsafe buffer to the given interior vector. The
 * vector must have size zero.
 */
extern "C" CDECL void
ivec_copy_from_buf(rust_task *task, type_desc *ty, rust_ivec *v, void *ptr,
                   size_t count)
{
    size_t old_size = get_ivec_size(v);
    if (old_size) {
        task->fail(1);
        return;
    }

    ivec_reserve(task, ty, v, count);

    size_t new_size = count * ty->size;
    if (v->fill || !v->payload.ptr) {
        // On stack.
        memmove(v->payload.data, ptr, new_size);
        v->fill = new_size;
        return;
    }

    // On heap.
    memmove(v->payload.ptr->data, ptr, new_size);
    v->payload.ptr->fill = new_size;
}

/**
 * Copies elements in an unsafe buffer to the given interior vector. The
 * vector must have size zero.
 */
extern "C" CDECL void
ivec_copy_from_buf_shared(rust_task *task, type_desc *ty, rust_ivec *v,
                   void *ptr, size_t count)
{
    size_t old_size = get_ivec_size(v);
    if (old_size) {
        task->fail(1);
        return;
    }

    ivec_reserve_shared(task, ty, v, count);

    size_t new_size = count * ty->size;
    if (v->fill || !v->payload.ptr) {
        // On stack.
        memmove(v->payload.data, ptr, new_size);
        v->fill = new_size;
        return;
    }

    // On heap.
    memmove(v->payload.ptr->data, ptr, new_size);
    v->payload.ptr->fill = new_size;
}

extern "C" CDECL void
pin_task(rust_task *task) {
    task->pin();
}

extern "C" CDECL void
unpin_task(rust_task *task) {
    task->unpin();
}

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
