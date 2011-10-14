/* Native builtins. */

#include "rust_internal.h"
#include "rust_scheduler.h"

#if !defined(__WIN32__)
#include <sys/time.h>
#endif

extern "C" CDECL rust_str*
last_os_error(void *unused_task) {
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
rust_getcwd(void *unused_task) {
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
void squareroot(void *unused_task, double *input, double *output) {
    *output = sqrt(*input);
}

extern "C" CDECL size_t
size_of(void *unused_task, type_desc *t) {
  return t->size;
}

extern "C" CDECL size_t
align_of(void *unused_task, type_desc *t) {
  return t->align;
}

extern "C" CDECL void
leak(void *unused_task, type_desc *t, void *thing) {
    // Do nothing. Call this with move-mode in order to say "Don't worry rust,
    // I'll take care of this."
}

extern "C" CDECL intptr_t
refcount(void *unused_task, type_desc *t, intptr_t *v) {

    // Passed-in value has refcount 1 too high
    // because it was ref'ed while making the call.
    return (*v) - 1;
}

extern "C" CDECL void
do_gc(void *unused_task) {
    // TODO
}

extern "C" CDECL void
unsupervise(void *unused_task) {
    rust_task *task = rust_scheduler::get_task();
    task->unsupervise();
}

extern "C" CDECL void
vec_reserve_shared(void *unused_task, type_desc* ty, rust_vec** vp,
                   size_t n_elts) {
    rust_task *task = rust_scheduler::get_task();
    reserve_vec(task, vp, n_elts * ty->size);
}

/**
 * Copies elements in an unsafe buffer to the given interior vector. The
 * vector must have size zero.
 */
extern "C" CDECL rust_vec*
vec_from_buf_shared(void *unused_task, type_desc *ty,
                    void *ptr, size_t count) {
    rust_task *task = rust_scheduler::get_task();
    size_t fill = ty->size * count;
    rust_vec* v = (rust_vec*)task->kernel->malloc(fill + sizeof(rust_vec),
                                                    "vec_from_buf");
    v->fill = v->alloc = fill;
    memmove(&v->data[0], ptr, fill);
    return v;
}

extern "C" CDECL void
rust_str_push(void *unused_task, rust_vec** sp, uint8_t byte) {
    rust_task *task = rust_scheduler::get_task();
    size_t fill = (*sp)->fill;
    reserve_vec(task, sp, fill + 1);
    (*sp)->data[fill-1] = byte;
    (*sp)->data[fill] = 0;
    (*sp)->fill = fill + 1;
}

extern "C" CDECL void *
rand_new(void *unused_task)
{
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
rand_next(void *unused_task, randctx *rctx)
{
    return isaac_rand(rctx);
}

extern "C" CDECL void
rand_free(void *unused_task, randctx *rctx)
{
    rust_task *task = rust_scheduler::get_task();
    task->free(rctx);
}

extern "C" CDECL void
task_sleep(void *unused_task, size_t time_in_us) {
    rust_task *task = rust_scheduler::get_task();
    task->yield(time_in_us);
}

extern "C" CDECL void
task_yield(void *unused_task) {
    rust_task *task = rust_scheduler::get_task();
    task->yield(1);
}

extern "C" CDECL intptr_t
task_join(void *unused_task, rust_task_id tid) {
    rust_task *task = rust_scheduler::get_task();
    // If the other task is already dying, we don't have to wait for it.
    rust_task *join_task = task->kernel->get_task_by_id(tid);
    // FIXME: find task exit status and return that.
    if(!join_task) return 0;
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
    if (!join_task->failed) {
        join_task->deref();
        return 0;
    } else {
        join_task->deref();
        return -1;
    }
}

/* Debug builtins for std::dbg. */

static void
debug_tydesc_helper(void *unused_task, type_desc *t)
{
    rust_task *task = rust_scheduler::get_task();
    LOG(task, stdlib, "  size %" PRIdPTR ", align %" PRIdPTR
        ", first_param 0x%" PRIxPTR,
        t->size, t->align, t->first_param);
}

extern "C" CDECL void
debug_tydesc(void *unused_task, type_desc *t)
{
    rust_task *task = rust_scheduler::get_task();
    LOG(task, stdlib, "debug_tydesc");
    debug_tydesc_helper(task, t);
}

extern "C" CDECL void
debug_opaque(void *unused_task, type_desc *t, uint8_t *front)
{
    rust_task *task = rust_scheduler::get_task();
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
debug_box(void *unused_task, type_desc *t, rust_box *box)
{
    rust_task *task = rust_scheduler::get_task();
    LOG(task, stdlib, "debug_box(0x%" PRIxPTR ")", box);
    debug_tydesc_helper(task, t);
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
debug_tag(void *unused_task, type_desc *t, rust_tag *tag)
{
    rust_task *task = rust_scheduler::get_task();

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
debug_obj(void *unused_task, type_desc *t, rust_obj *obj,
          size_t nmethods, size_t nbytes)
{
    rust_task *task = rust_scheduler::get_task();

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
debug_fn(void *unused_task, type_desc *t, rust_fn *fn)
{
    rust_task *task = rust_scheduler::get_task();
    LOG(task, stdlib, "debug_fn");
    debug_tydesc_helper(task, t);
    LOG(task, stdlib, "  thunk at 0x%" PRIxPTR, fn->thunk);
    LOG(task, stdlib, "  closure at 0x%" PRIxPTR, fn->closure);
    if (fn->closure) {
        LOG(task, stdlib, "    refcount %" PRIdPTR, fn->closure->ref_count);
    }
}

extern "C" CDECL void *
debug_ptrcast(void *unused_task,
              type_desc *from_ty,
              type_desc *to_ty,
              void *ptr)
{
    rust_task *task = rust_scheduler::get_task();
    LOG(task, stdlib, "debug_ptrcast from");
    debug_tydesc_helper(task, from_ty);
    LOG(task, stdlib, "to");
    debug_tydesc_helper(task, to_ty);
    return ptr;
}

extern "C" CDECL rust_vec*
rust_list_files(void *unused_task, rust_vec **path) {
    rust_task *task = rust_scheduler::get_task();
    array_list<rust_str*> strings;
#if defined(__WIN32__)
    WIN32_FIND_DATA FindFileData;
    HANDLE hFind = FindFirstFile((char*)(*path)->data, &FindFileData);
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
    DIR *dirp = opendir((char*)(*path)->data);
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
rust_file_is_dir(void *unused_task, char *path) {
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
rust_ptr_eq(void *unused_task, type_desc *t, rust_box *a, rust_box *b) {
    return a == b;
}

#if defined(__WIN32__)
extern "C" CDECL void
get_time(void *unused_task, uint32_t *sec, uint32_t *usec) {
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
get_time(void *unused_task, uint32_t *sec, uint32_t *usec) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    *sec = tv.tv_sec;
    *usec = tv.tv_usec;
}
#endif

extern "C" CDECL void
nano_time(void *unused_task, uint64_t *ns) {
    timer t;
    *ns = t.time_ns();
}

extern "C" CDECL void
pin_task(void *unused_task) {
    rust_task *task = rust_scheduler::get_task();
    task->pin();
}

extern "C" CDECL void
unpin_task(void *unused_task) {
    rust_task *task = rust_scheduler::get_task();
    task->unpin();
}

extern "C" CDECL rust_task_id
get_task_id(void *unused_task) {
    rust_task *task = rust_scheduler::get_task();
    return task->user.id;
}

extern "C" CDECL rust_task_id
new_task(void *unused_task) {
    rust_task *task = rust_scheduler::get_task();
    return task->kernel->create_task(task, NULL);
}

extern "C" CDECL void
drop_task(void *unused_task, rust_task *target) {
    if(target) {
        target->deref();
    }
}

extern "C" CDECL rust_task *
get_task_pointer(void *unused_task, rust_task_id id) {
    rust_task *task = rust_scheduler::get_task();
    return task->kernel->get_task_by_id(id);
}

// FIXME: Transitional. Remove
extern "C" CDECL void **
get_task_trampoline(void *unused_task) {
    return NULL;
}

struct fn_env_pair {
    intptr_t f;
    intptr_t env;
};

// FIXME This is probably not needed at all anymore. Have to rearrange some
// argument passing to remove it.
void rust_spawn_wrapper(void* retptr, rust_task* taskptr, void* envptr,
                        void(*func)(void*, rust_task*, void*)) {
    func(retptr, taskptr, envptr);
}

extern "C" CDECL void
start_task(void *unused_task, rust_task_id id, fn_env_pair *f) {
    rust_task *task = rust_scheduler::get_task();
    rust_task *target = task->kernel->get_task_by_id(id);
    target->start((uintptr_t)rust_spawn_wrapper, f->f, f->env);
    target->deref();
}

extern "C" CDECL void
migrate_alloc(void *unused_task, void *alloc, rust_task_id tid) {
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
set_min_stack(void *unused_task, uintptr_t stack_size) {
    g_custom_min_stack_size = stack_size;
}

extern "C" CDECL int
sched_threads(void *unused_task) {
    rust_task *task = rust_scheduler::get_task();
    return task->kernel->num_threads;
}

extern "C" CDECL rust_port*
new_port(void *unused_task, size_t unit_sz) {
    rust_task *task = rust_scheduler::get_task();
    LOG(task, comm, "new_port(task=0x%" PRIxPTR " (%s), unit_sz=%d)",
        (uintptr_t) task, task->name, unit_sz);
    // take a reference on behalf of the port
    task->ref();
    return new (task->kernel, "rust_port") rust_port(task, unit_sz);
}

extern "C" CDECL void
del_port(void *unused_task, rust_port *port) {
    rust_task *task = rust_scheduler::get_task();
    LOG(task, comm, "del_port(0x%" PRIxPTR ")", (uintptr_t) port);
    I(task->sched, !port->ref_count);
    delete port;

    // FIXME: this should happen in the port.
    task->deref();
}

extern "C" CDECL rust_port_id
get_port_id(void *unused_task, rust_port *port) {
    return port->id;
}

extern "C" CDECL rust_chan*
new_chan(void *unused_task, rust_port *port) {
    rust_task *task = rust_scheduler::get_task();
    rust_scheduler *sched = task->sched;
    LOG(task, comm, "new_chan("
        "task=0x%" PRIxPTR " (%s), port=0x%" PRIxPTR ")",
        (uintptr_t) task, task->name, port);
    I(sched, port);
    return new (task->kernel, "rust_chan")
        rust_chan(task->kernel, port, port->unit_sz);
}

extern "C" CDECL
void del_chan(void *unused_task, rust_chan *chan) {
    rust_task *task = rust_scheduler::get_task();
    LOG(task, comm, "del_chan(0x%" PRIxPTR ")", (uintptr_t) chan);
    I(task->sched, false);
}

extern "C" CDECL
void take_chan(void *unused_task, rust_chan *chan) {
    chan->ref();
}

extern "C" CDECL
void drop_chan(void *unused_task, rust_chan *chan) {
    chan->deref();
}

extern "C" CDECL
void drop_port(void *, rust_port *port) {
    port->ref_count--;
}

extern "C" CDECL void
chan_send(void *unused_task, rust_chan *chan, void *sptr) {
    chan->send(sptr);
}

extern "C" CDECL void
chan_id_send(void *unused_task, type_desc *t, rust_task_id target_task_id,
             rust_port_id target_port_id, void *sptr) {
    // FIXME: make sure this is thread-safe
    rust_task *task = rust_scheduler::get_task();
    rust_task *target_task = task->kernel->get_task_by_id(target_task_id);
    if(target_task) {
        rust_port *port = target_task->get_port_by_id(target_port_id);
        if(port) {
            port->remote_chan->send(sptr);
        }
        target_task->deref();
        task->yield();
    }
}

extern "C" CDECL void
port_recv(void *unused_task, uintptr_t *dptr, rust_port *port) {
    rust_task *task = rust_scheduler::get_task();
    {
        scoped_lock with(port->lock);

        LOG(task, comm, "port: 0x%" PRIxPTR ", dptr: 0x%" PRIxPTR
            ", size: 0x%" PRIxPTR ", chan_no: %d",
            (uintptr_t) port, (uintptr_t) dptr, port->unit_sz,
            port->chans.length());

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
