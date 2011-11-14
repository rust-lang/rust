/* Native builtins. */

#include "rust_internal.h"
#include "rust_scheduler.h"

#if !defined(__WIN32__)
#include <sys/time.h>
#endif

extern "C" CDECL rust_str*
last_os_error() {
    rust_task *task = rust_scheduler::get_task();

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
        task->fail();
        return NULL;
    }
#elif defined(_GNU_SOURCE)
    char cbuf[BUF_BYTES];
    char *buf = strerror_r(errno, cbuf, sizeof(cbuf));
    if (!buf) {
        task->fail();
        return NULL;
    }
#else
    char buf[BUF_BYTES];
    int err = strerror_r(errno, buf, sizeof(buf));
    if (err) {
        task->fail();
        return NULL;
    }
#endif

    rust_str * st = make_str(task->kernel, buf, strlen(buf),
                             "last_os_error");
#ifdef __WIN32__
    LocalFree((HLOCAL)buf);
#endif
    return st;
}

extern "C" CDECL rust_str *
rust_getcwd() {
    rust_task *task = rust_scheduler::get_task();
    LOG(task, task, "rust_getcwd()");

    char cbuf[BUF_BYTES];

#if defined(__WIN32__)
    if (!_getcwd(cbuf, sizeof(cbuf))) {
#else
        if (!getcwd(cbuf, sizeof(cbuf))) {
#endif
        task->fail();
        return NULL;
    }

    return make_str(task->kernel, cbuf, strlen(cbuf), "rust_str(getcwd");
}

// TODO: Allow calling native functions that return double results.
extern "C" CDECL
void squareroot(double *input, double *output) {
    *output = sqrt(*input);
}

extern "C" CDECL size_t
size_of(type_desc *t) {
  return t->size;
}

extern "C" CDECL size_t
align_of(type_desc *t) {
  return t->align;
}

extern "C" CDECL void
leak(void *thing) {
    // Do nothing. Call this with move-mode in order to say "Don't worry rust,
    // I'll take care of this."
}

extern "C" CDECL intptr_t
refcount(intptr_t *v) {
    // Passed-in value has refcount 1 too high
    // because it was ref'ed while making the call.
    return (*v) - 1;
}

extern "C" CDECL void
do_gc() {
    // TODO
}

extern "C" CDECL void
unsupervise() {
    rust_task *task = rust_scheduler::get_task();
    task->unsupervise();
}

extern "C" CDECL void
vec_reserve_shared(type_desc* ty, rust_vec** vp,
                   size_t n_elts) {
    rust_task *task = rust_scheduler::get_task();
    reserve_vec(task, vp, n_elts * ty->size);
}

/**
 * Copies elements in an unsafe buffer to the given interior vector. The
 * vector must have size zero.
 */
extern "C" CDECL rust_vec*
vec_from_buf_shared(type_desc *ty, void *ptr, size_t count) {
    rust_task *task = rust_scheduler::get_task();
    size_t fill = ty->size * count;
    rust_vec* v = (rust_vec*)task->kernel->malloc(fill + sizeof(rust_vec),
                                                    "vec_from_buf");
    v->fill = v->alloc = fill;
    memmove(&v->data[0], ptr, fill);
    return v;
}

extern "C" CDECL void
rust_str_push(rust_vec** sp, uint8_t byte) {
    rust_task *task = rust_scheduler::get_task();
    size_t fill = (*sp)->fill;
    reserve_vec(task, sp, fill + 1);
    (*sp)->data[fill-1] = byte;
    (*sp)->data[fill] = 0;
    (*sp)->fill = fill + 1;
}

extern "C" CDECL void *
rand_new() {
    rust_task *task = rust_scheduler::get_task();
    rust_scheduler *sched = task->sched;
    randctx *rctx = (randctx *) task->malloc(sizeof(randctx), "randctx");
    if (!rctx) {
        task->fail();
        return NULL;
    }
    isaac_init(sched, rctx);
    return rctx;
}

extern "C" CDECL size_t
rand_next(randctx *rctx) {
    return isaac_rand(rctx);
}

extern "C" CDECL void
rand_free(randctx *rctx) {
    rust_task *task = rust_scheduler::get_task();
    task->free(rctx);
}

/* Debug builtins for std::dbg. */

static void
debug_tydesc_helper(type_desc *t)
{
    rust_task *task = rust_scheduler::get_task();
    LOG(task, stdlib, "  size %" PRIdPTR ", align %" PRIdPTR
        ", first_param 0x%" PRIxPTR,
        t->size, t->align, t->first_param);
}

extern "C" CDECL void
debug_tydesc(type_desc *t) {
    rust_task *task = rust_scheduler::get_task();
    LOG(task, stdlib, "debug_tydesc");
    debug_tydesc_helper(t);
}

extern "C" CDECL void
debug_opaque(type_desc *t, uint8_t *front) {
    rust_task *task = rust_scheduler::get_task();
    LOG(task, stdlib, "debug_opaque");
    debug_tydesc_helper(t);
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
debug_box(type_desc *t, rust_box *box) {
    rust_task *task = rust_scheduler::get_task();
    LOG(task, stdlib, "debug_box(0x%" PRIxPTR ")", box);
    debug_tydesc_helper(t);
    LOG(task, stdlib, "  refcount %" PRIdPTR,
        box->ref_count - 1);  // -1 because we ref'ed for this call
    for (uintptr_t i = 0; i < t->size; ++i) {
        LOG(task, stdlib, "  byte %" PRIdPTR ": 0x%" PRIx8, i, box->data[i]);
    }
}

struct rust_tag {
    uintptr_t discriminant;
    uint8_t variant[];
};

extern "C" CDECL void
debug_tag(type_desc *t, rust_tag *tag) {
    rust_task *task = rust_scheduler::get_task();

    LOG(task, stdlib, "debug_tag");
    debug_tydesc_helper(t);
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
debug_obj(type_desc *t, rust_obj *obj, size_t nmethods, size_t nbytes) {
    rust_task *task = rust_scheduler::get_task();

    LOG(task, stdlib, "debug_obj with %" PRIdPTR " methods", nmethods);
    debug_tydesc_helper(t);
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
debug_fn(type_desc *t, rust_fn *fn) {
    rust_task *task = rust_scheduler::get_task();
    LOG(task, stdlib, "debug_fn");
    debug_tydesc_helper(t);
    LOG(task, stdlib, "  thunk at 0x%" PRIxPTR, fn->thunk);
    LOG(task, stdlib, "  closure at 0x%" PRIxPTR, fn->closure);
    if (fn->closure) {
        LOG(task, stdlib, "    refcount %" PRIdPTR, fn->closure->ref_count);
    }
}

extern "C" CDECL void *
debug_ptrcast(type_desc *from_ty,
              type_desc *to_ty,
              void *ptr) {
    rust_task *task = rust_scheduler::get_task();
    LOG(task, stdlib, "debug_ptrcast from");
    debug_tydesc_helper(from_ty);
    LOG(task, stdlib, "to");
    debug_tydesc_helper(to_ty);
    return ptr;
}

extern "C" CDECL rust_vec*
rust_list_files(rust_str *path) {
    rust_task *task = rust_scheduler::get_task();
    array_list<rust_str*> strings;
#if defined(__WIN32__)
    WIN32_FIND_DATA FindFileData;
    HANDLE hFind = FindFirstFile((char*)path->data, &FindFileData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            rust_str *str = make_str(task->kernel, FindFileData.cFileName,
                                     strlen(FindFileData.cFileName),
                                     "list_files_str");
            strings.push(str);
        } while (FindNextFile(hFind, &FindFileData));
        FindClose(hFind);
    }
#else
    DIR *dirp = opendir((char*)path->data);
  if (dirp) {
      struct dirent *dp;
      while ((dp = readdir(dirp))) {
          rust_vec *str = make_str(task->kernel, dp->d_name,
                                    strlen(dp->d_name),
                                    "list_files_str");
          strings.push(str);
      }
      closedir(dirp);
  }
#endif

  rust_vec *vec = (rust_vec *)
      task->kernel->malloc(vec_size<rust_vec*>(strings.size()),
                           "list_files_vec");
  size_t alloc_sz = sizeof(rust_vec*) * strings.size();
  vec->fill = vec->alloc = alloc_sz;
  memcpy(&vec->data[0], strings.data(), alloc_sz);
  return vec;
}

extern "C" CDECL int
rust_file_is_dir(char *path) {
    struct stat buf;
    if (stat(path, &buf)) {
        return 0;
    }
    return S_ISDIR(buf.st_mode);
}

extern "C" CDECL FILE* rust_get_stdin() {return stdin;}
extern "C" CDECL FILE* rust_get_stdout() {return stdout;}
extern "C" CDECL FILE* rust_get_stderr() {return stderr;}

extern "C" CDECL int
rust_ptr_eq(type_desc *t, rust_box *a, rust_box *b) {
    return a == b;
}

#if defined(__WIN32__)
extern "C" CDECL void
get_time(uint32_t *sec, uint32_t *usec) {
    rust_task *task = rust_scheduler::get_task();
    SYSTEMTIME systemTime;
    FILETIME fileTime;
    GetSystemTime(&systemTime);
    if (!SystemTimeToFileTime(&systemTime, &fileTime)) {
        task->fail();
        return;
    }

    // FIXME: This is probably completely wrong.
    *sec = fileTime.dwHighDateTime;
    *usec = fileTime.dwLowDateTime;
}
#else
extern "C" CDECL void
get_time(uint32_t *sec, uint32_t *usec) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    *sec = tv.tv_sec;
    *usec = tv.tv_usec;
}
#endif

extern "C" CDECL void
nano_time(uint64_t *ns) {
    timer t;
    *ns = t.time_ns();
}

extern "C" CDECL void
pin_task() {
    rust_task *task = rust_scheduler::get_task();
    task->pin();
}

extern "C" CDECL void
unpin_task() {
    rust_task *task = rust_scheduler::get_task();
    task->unpin();
}

extern "C" CDECL rust_task_id
get_task_id() {
    rust_task *task = rust_scheduler::get_task();
    return task->user.id;
}

extern "C" CDECL rust_task_id
new_task() {
    rust_task *task = rust_scheduler::get_task();
    return task->kernel->create_task(task, NULL);
}

extern "C" CDECL void
drop_task(rust_task *target) {
    if(target) {
        target->deref();
    }
}

extern "C" CDECL rust_task *
get_task_pointer(rust_task_id id) {
    rust_task *task = rust_scheduler::get_task();
    return task->kernel->get_task_by_id(id);
}

struct fn_env_pair {
    intptr_t f;
    intptr_t env;
};

// FIXME This is probably not needed at all anymore. Have to rearrange some
// argument passing to remove it.
void rust_spawn_wrapper(void* retptr, void* envptr,
                        void(*func)(void*, void*)) {
    func(retptr, envptr);
}

extern "C" CDECL void
start_task(rust_task_id id, fn_env_pair *f) {
    rust_task *task = rust_scheduler::get_task();
    rust_task *target = task->kernel->get_task_by_id(id);
    target->start((uintptr_t)rust_spawn_wrapper, f->f, f->env);
    target->deref();
}

extern "C" CDECL void
migrate_alloc(void *alloc, rust_task_id tid) {
    rust_task *task = rust_scheduler::get_task();
    if(!alloc) return;
    rust_task *target = task->kernel->get_task_by_id(tid);
    if(target) {
        const type_desc *tydesc = task->release_alloc(alloc);
        target->claim_alloc(alloc, tydesc);
        target->deref();
    }
    else {
        // We couldn't find the target. Maybe we should just free?
        task->fail();
    }
}

// defined in rust_task.cpp
extern size_t g_custom_min_stack_size;
extern "C" CDECL void
set_min_stack(uintptr_t stack_size) {
    g_custom_min_stack_size = stack_size;
}

extern "C" CDECL int
sched_threads() {
    rust_task *task = rust_scheduler::get_task();
    return task->kernel->num_threads;
}

extern "C" CDECL rust_port*
new_port(size_t unit_sz) {
    rust_task *task = rust_scheduler::get_task();
    LOG(task, comm, "new_port(task=0x%" PRIxPTR " (%s), unit_sz=%d)",
        (uintptr_t) task, task->name, unit_sz);
    // port starts with refcount == 1
    return new (task->kernel, "rust_port") rust_port(task, unit_sz);
}

extern "C" CDECL void
del_port(rust_port *port) {
    rust_task *task = rust_scheduler::get_task();
    LOG(task, comm, "del_port(0x%" PRIxPTR ")", (uintptr_t) port);
    scoped_lock with(task->lock);
    port->deref();
}

extern "C" CDECL size_t
rust_port_size(rust_port *port) {
    return port->size();
}

extern "C" CDECL rust_port_id
get_port_id(rust_port *port) {
    return port->id;
}

extern "C" CDECL uintptr_t
chan_id_send(type_desc *t, rust_task_id target_task_id,
             rust_port_id target_port_id, void *sptr) {
    // FIXME: make sure this is thread-safe
    bool sent = false;
    rust_task *task = rust_scheduler::get_task();
    rust_task *target_task = task->kernel->get_task_by_id(target_task_id);
    if(target_task) {
        rust_port *port = target_task->get_port_by_id(target_port_id);
        if(port) {
            port->send(sptr);
            scoped_lock with(target_task->lock);
            port->deref();
            sent = true;
        }
        target_task->deref();
    }
    return (uintptr_t)sent;
}

// This is called by an intrinsic on the Rust stack.
// Do not call on the C stack.
extern "C" CDECL void
rust_task_sleep(size_t time_in_us) {
    rust_task *task = rust_scheduler::get_task();
    task->yield(time_in_us);
}

// This is called by an intrinsic on the Rust stack.
// Do not call on the C stack.
extern "C" CDECL void
port_recv(uintptr_t *dptr, rust_port *port) {
    rust_task *task = rust_scheduler::get_task();
    {
        scoped_lock with(port->lock);

        LOG(task, comm, "port: 0x%" PRIxPTR ", dptr: 0x%" PRIxPTR
            ", size: 0x%" PRIxPTR,
            (uintptr_t) port, (uintptr_t) dptr, port->unit_sz);

        if (port->receive(dptr)) {
            return;
        }

        // No data was buffered on any incoming channel, so block this task on
        // the port. Remember the rendezvous location so that any sender task
        // can write to it before waking up this task.

        LOG(task, comm, "<=== waiting for rendezvous data ===");
        task->rendezvous_ptr = dptr;
        task->block(port, "waiting for rendezvous data");
    }
    task->yield(3);
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
