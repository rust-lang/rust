#include "rust_gc.h"
#include "rust_internal.h"
#include "rust_upcall.h"

// Upcalls.

#ifdef __i386__
void
check_stack(rust_task *task) {
    void *esp;
    asm volatile("movl %%esp,%0" : "=r" (esp));
    if (esp < task->stk->data)
        task->kernel->fatal("Out of stack space, sorry");
}
#else
#warning "Stack checks are not supported on this architecture"
void
check_stack(rust_task *task) {
    // TODO
}
#endif

extern "C" void
upcall_grow_task(rust_task *task, size_t n_frame_bytes) {
    I(task->sched, false);
    LOG_UPCALL_ENTRY(task);
    task->grow(n_frame_bytes);
}

extern "C" CDECL
void upcall_log_int(rust_task *task, uint32_t level, int32_t i) {
    LOG_UPCALL_ENTRY(task);
    if (task->sched->log_lvl >= level)
        task->sched->log(task, level, "rust: %" PRId32 " (0x%" PRIx32 ")",
                       i, i);
}

extern "C" CDECL
void upcall_log_float(rust_task *task, uint32_t level, float f) {
    LOG_UPCALL_ENTRY(task);
    if (task->sched->log_lvl >= level)
        task->sched->log(task, level, "rust: %12.12f", f);
}

extern "C" CDECL
void upcall_log_double(rust_task *task, uint32_t level, double *f) {
    LOG_UPCALL_ENTRY(task);
    if (task->sched->log_lvl >= level)
        task->sched->log(task, level, "rust: %12.12f", *f);
}

extern "C" CDECL void
upcall_yield(rust_task *task) {
    LOG_UPCALL_ENTRY(task);
    LOG(task, comm, "upcall yield()");
    task->yield(1);
}

// Copy elements from one vector to another,
// dealing with reference counts
static inline void
copy_elements(rust_task *task, type_desc *elem_t,
              void *pdst, void *psrc, size_t n)
{
    char *dst = (char *)pdst, *src = (char *)psrc;
    memmove(dst, src, n);

    // increment the refcount of each element of the vector
    if (elem_t->take_glue) {
        glue_fn *take_glue = elem_t->take_glue;
        size_t elem_size = elem_t->size;
        const type_desc **tydescs = elem_t->first_param;
        for (char *p = dst; p < dst+n; p += elem_size) {
            take_glue(NULL, task, NULL, tydescs, p);
        }
    }
}

extern "C" CDECL void
upcall_sleep(rust_task *task, size_t time_in_us) {
    LOG_UPCALL_ENTRY(task);
    LOG(task, task, "elapsed %" PRIu64 " us",
              task->yield_timer.elapsed_us());
    LOG(task, task, "sleep %d us", time_in_us);
    task->yield(time_in_us);
}

extern "C" CDECL void
upcall_fail(rust_task *task,
            char const *expr,
            char const *file,
            size_t line) {
    LOG_UPCALL_ENTRY(task);
    LOG_ERR(task, upcall, "upcall fail '%s', %s:%" PRIdPTR, expr, file, line);
    task->fail();
}

/**
 * Called whenever a task's ref count drops to zero.
 */
extern "C" CDECL void
upcall_kill(rust_task *task, rust_task_id tid) {
    LOG_UPCALL_ENTRY(task);
    rust_task *target = task->kernel->get_task_by_id(tid);
    target->kill();
    target->deref();
}

/**
 * Called by the exit glue when the task terminates.
 */
extern "C" CDECL void
upcall_exit(rust_task *task) {
    LOG_UPCALL_ENTRY(task);
    task->die();
    task->notify_tasks_waiting_to_join();
    task->yield(1);
}

extern "C" CDECL uintptr_t
upcall_malloc(rust_task *task, size_t nbytes, type_desc *td) {
    LOG_UPCALL_ENTRY(task);

    LOG(task, mem,
        "upcall malloc(%" PRIdPTR ", 0x%" PRIxPTR ")"
        " with gc-chain head = 0x%" PRIxPTR,
        nbytes, td, task->gc_alloc_chain);

    gc::maybe_gc(task);

    // TODO: Maybe use dladdr here to find a more useful name for the
    // type_desc.

    void *p = task->malloc(nbytes, "tdesc", td);

    LOG(task, mem,
        "upcall malloc(%" PRIdPTR ", 0x%" PRIxPTR
        ") = 0x%" PRIxPTR
        " with gc-chain head = 0x%" PRIxPTR,
        nbytes, td, (uintptr_t)p, task->gc_alloc_chain);
    return (uintptr_t) p;
}

/**
 * Called whenever an object's ref count drops to zero.
 */
extern "C" CDECL void
upcall_free(rust_task *task, void* ptr, uintptr_t is_gc) {
    LOG_UPCALL_ENTRY(task);

    rust_scheduler *sched = task->sched;
    DLOG(sched, mem,
             "upcall free(0x%" PRIxPTR ", is_gc=%" PRIdPTR ")",
             (uintptr_t)ptr, is_gc);
    task->free(ptr, (bool) is_gc);
}

extern "C" CDECL uintptr_t
upcall_shared_malloc(rust_task *task, size_t nbytes, type_desc *td) {
    LOG_UPCALL_ENTRY(task);

    LOG(task, mem,
                   "upcall shared_malloc(%" PRIdPTR ", 0x%" PRIxPTR ")",
                   nbytes, td);
    void *p = task->kernel->malloc(nbytes, "shared malloc");
    LOG(task, mem,
                   "upcall shared_malloc(%" PRIdPTR ", 0x%" PRIxPTR
                   ") = 0x%" PRIxPTR,
                   nbytes, td, (uintptr_t)p);
    return (uintptr_t) p;
}

/**
 * Called whenever an object's ref count drops to zero.
 */
extern "C" CDECL void
upcall_shared_free(rust_task *task, void* ptr) {
    LOG_UPCALL_ENTRY(task);

    rust_scheduler *sched = task->sched;
    DLOG(sched, mem,
             "upcall shared_free(0x%" PRIxPTR")",
             (uintptr_t)ptr);
    task->kernel->free(ptr);
}

extern "C" CDECL type_desc *
upcall_get_type_desc(rust_task *task,
                     void *curr_crate, // ignored, legacy compat.
                     size_t size,
                     size_t align,
                     size_t n_descs,
                     type_desc const **descs,
                     uintptr_t n_obj_params) {
    check_stack(task);
    LOG_UPCALL_ENTRY(task);

    LOG(task, cache, "upcall get_type_desc with size=%" PRIdPTR
        ", align=%" PRIdPTR ", %" PRIdPTR " descs", size, align,
        n_descs);
    rust_crate_cache *cache = task->get_crate_cache();
    type_desc *td = cache->get_type_desc(size, align, n_descs, descs,
                                         n_obj_params);
    LOG(task, cache, "returning tydesc 0x%" PRIxPTR, td);
    return td;
}

extern "C" CDECL void
upcall_vec_grow(rust_task* task, rust_vec** vp, size_t new_sz) {
    LOG_UPCALL_ENTRY(task);
    reserve_vec(task, vp, new_sz);
    (*vp)->fill = new_sz;
}

extern "C" CDECL void
upcall_vec_push(rust_task* task, rust_vec** vp, type_desc* elt_ty,
                void* elt) {
    LOG_UPCALL_ENTRY(task);
    size_t new_sz = (*vp)->fill + elt_ty->size;
    reserve_vec(task, vp, new_sz);
    rust_vec* v = *vp;
    copy_elements(task, elt_ty, &v->data[0] + v->fill, elt, elt_ty->size);
    v->fill += elt_ty->size;
}

/**
 * Returns a token that can be used to deallocate all of the allocated space
 * space in the dynamic stack.
 */
extern "C" CDECL void *
upcall_dynastack_mark(rust_task *task) {
    return task->dynastack.mark();
}

/**
 * Allocates space in the dynamic stack and returns it.
 *
 * FIXME: Deprecated since dynamic stacks need to be self-describing for GC.
 */
extern "C" CDECL void *
upcall_dynastack_alloc(rust_task *task, size_t sz) {
    return sz ? task->dynastack.alloc(sz, NULL) : NULL;
}

/**
 * Allocates space associated with a type descriptor in the dynamic stack and
 * returns it.
 */
extern "C" CDECL void *
upcall_dynastack_alloc_2(rust_task *task, size_t sz, type_desc *ty) {
    return sz ? task->dynastack.alloc(sz, ty) : NULL;
}

/** Frees space in the dynamic stack. */
extern "C" CDECL void
upcall_dynastack_free(rust_task *task, void *ptr) {
    return task->dynastack.free(ptr);
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
