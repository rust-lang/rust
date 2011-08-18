#include "rust_internal.h"
#include "rust_upcall.h"

// Upcalls.

extern "C" CDECL char const *
str_buf(rust_task *task, rust_str *s);

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
upcall_log_str(rust_task *task, uint32_t level, rust_str *str) {
    LOG_UPCALL_ENTRY(task);
    if (task->sched->log_lvl >= level) {
        const char *c = str_buf(task, str);
        task->sched->log(task, level, "rust: %s", c);
    }
}

extern "C" CDECL void
upcall_log_istr(rust_task *task, uint32_t level, rust_ivec *str) {
    LOG_UPCALL_ENTRY(task);
    if (task->sched->log_lvl < level)
        return;
    const char *buf = (const char *)
        (str->fill ? str->payload.data : str->payload.ptr->data);
    task->sched->log(task, level, "rust: %s", buf);
}

extern "C" CDECL void
upcall_yield(rust_task *task) {
    LOG_UPCALL_ENTRY(task);
    LOG(task, comm, "upcall yield()");
    task->yield(1);
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
    task->die();
    task->fail();
    task->notify_tasks_waiting_to_join();
    task->yield(4);
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

extern "C" CDECL uintptr_t
upcall_mark(rust_task *task, void* ptr) {
    LOG_UPCALL_ENTRY(task);

    rust_scheduler *sched = task->sched;
    if (ptr) {
        gc_alloc *gcm = (gc_alloc*) (((char*)ptr) - sizeof(gc_alloc));
        uintptr_t marked = (uintptr_t) gcm->mark();
        DLOG(sched, gc, "upcall mark(0x%" PRIxPTR ") = %" PRIdPTR,
                 (uintptr_t)gcm, marked);
        return marked;
    }
    return 0;
}

rust_str *make_str(rust_task *task, char const *s, size_t fill) {
    size_t alloc = next_power_of_two(sizeof(rust_str) + fill);
    void *mem = task->malloc(alloc, "rust_str (make_str)");
    if (!mem) {
        task->fail();
        return NULL;
    }
    rust_str *st = new (mem) rust_str(alloc, fill,
                                      (uint8_t const *) s);
    LOG(task, mem,
        "upcall new_str('%s', %" PRIdPTR ") = 0x%" PRIxPTR,
        s, fill, st);
    return st;
}

extern "C" CDECL rust_str *
upcall_new_str(rust_task *task, char const *s, size_t fill) {
    LOG_UPCALL_ENTRY(task);
    return make_str(task, s, fill);
}

static rust_vec *
vec_grow(rust_task *task,
         rust_vec *v,
         size_t n_bytes,
         uintptr_t *need_copy,
         type_desc *td)
{
    rust_scheduler *sched = task->sched;
    LOG(task, mem,
        "vec_grow(0x%" PRIxPTR ", %" PRIdPTR
        "), rc=%" PRIdPTR " alloc=%" PRIdPTR ", fill=%" PRIdPTR
        ", need_copy=0x%" PRIxPTR,
        v, n_bytes, v->ref_count, v->alloc, v->fill, need_copy);

    *need_copy = 0;
    size_t alloc = next_power_of_two(sizeof(rust_vec) + v->fill + n_bytes);

    if (v->ref_count == 1) {

        // Fastest path: already large enough.
        if (v->alloc >= alloc) {
            LOG(task, mem, "no-growth path");
            return v;
        }

        // Second-fastest path: can at least realloc.
        LOG(task, mem, "realloc path");
        v = (rust_vec*) task->realloc(v, alloc, td->is_stateful);
        if (!v) {
            task->fail();
            return NULL;
        }
        v->alloc = alloc;

    } else {
        /**
         * Slowest path: make a new vec.
         *
         * 1. Allocate a new rust_vec with desired additional space.
         * 2. Down-ref the shared rust_vec, point to the new one instead.
         * 3. Copy existing elements into the new rust_vec.
         *
         * Step 3 is a bit tricky.  We don't know how to properly copy the
         * elements in the runtime (all we have are bits in a buffer; no
         * type information and no copy glue).  What we do instead is set the
         * need_copy outparam flag to indicate to our caller (vec-copy glue)
         * that we need the copies performed for us.
         */
        LOG(task, mem, "new vec path");
        void *mem = task->malloc(alloc, "rust_vec (vec_grow)", td);
        if (!mem) {
            task->fail();
            return NULL;
        }

        if (v->ref_count != CONST_REFCOUNT)
            v->deref();

        v = new (mem) rust_vec(alloc, 0, NULL);
        *need_copy = 1;
    }
    I(sched, sizeof(rust_vec) + v->fill <= v->alloc);
    return v;
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
    if (elem_t->copy_glue) {
        glue_fn *copy_glue = elem_t->copy_glue;
        size_t elem_size = elem_t->size;
        const type_desc **tydescs = elem_t->first_param;
        for (char *p = dst; p < dst+n; p += elem_size) {
            copy_glue(NULL, task, NULL, tydescs, p);
        }
    }
}

extern "C" CDECL void
upcall_vec_append(rust_task *task, type_desc *t, type_desc *elem_t,
                  rust_vec **dst_ptr, rust_vec *src, bool skip_null)
{
    LOG_UPCALL_ENTRY(task);
    rust_vec *dst = *dst_ptr;
    uintptr_t need_copy;
    size_t n_src_bytes = skip_null ? src->fill - 1 : src->fill;
    size_t n_dst_bytes = skip_null ? dst->fill - 1 : dst->fill;
    rust_vec *new_vec = vec_grow(task, dst, n_src_bytes, &need_copy, t);

    // If src and dst are the same (due to "v += v"), then dst getting
    // resized causes src to move as well.
    if (dst == src && !need_copy) {
        src = new_vec;
    }

    if (need_copy) {
        // Copy any dst elements in, omitting null if doing str.
        copy_elements(task, elem_t, &new_vec->data, &dst->data, n_dst_bytes);
    }

    // Copy any src elements in, carrying along null if doing str.
    void *new_end = (void *)((char *)new_vec->data + n_dst_bytes);
    copy_elements(task, elem_t, new_end, &src->data, src->fill);
    new_vec->fill = n_dst_bytes + src->fill;

    // Write new_vec back through the alias we were given.
    *dst_ptr = new_vec;
}


extern "C" CDECL type_desc *
upcall_get_type_desc(rust_task *task,
                     void *curr_crate, // ignored, legacy compat.
                     size_t size,
                     size_t align,
                     size_t n_descs,
                     type_desc const **descs) {
    check_stack(task);
    LOG_UPCALL_ENTRY(task);

    LOG(task, cache, "upcall get_type_desc with size=%" PRIdPTR
        ", align=%" PRIdPTR ", %" PRIdPTR " descs", size, align,
        n_descs);
    rust_crate_cache *cache = task->get_crate_cache();
    type_desc *td = cache->get_type_desc(size, align, n_descs, descs);
    LOG(task, cache, "returning tydesc 0x%" PRIxPTR, td);
    return td;
}

/**
 * Resizes an interior vector that has been spilled to the heap.
 */
extern "C" CDECL void
upcall_ivec_resize_shared(rust_task *task,
                          rust_ivec *v,
                          size_t newsz) {
    LOG_UPCALL_ENTRY(task);
    scoped_lock with(task->sched->lock);
    I(task->sched, !v->fill);

    size_t new_alloc = next_power_of_two(newsz);
    rust_ivec_heap *new_heap_part = (rust_ivec_heap *)
        task->kernel->realloc(v->payload.ptr, new_alloc + sizeof(size_t));

    new_heap_part->fill = newsz;
    v->alloc = new_alloc;
    v->payload.ptr = new_heap_part;
}

/**
 * Spills an interior vector to the heap.
 */
extern "C" CDECL void
upcall_ivec_spill_shared(rust_task *task,
                         rust_ivec *v,
                         size_t newsz) {
    LOG_UPCALL_ENTRY(task);
    scoped_lock with(task->sched->lock);
    size_t new_alloc = next_power_of_two(newsz);

    rust_ivec_heap *heap_part = (rust_ivec_heap *)
        task->kernel->malloc(new_alloc + sizeof(size_t),
                             "ivec spill shared");
    heap_part->fill = newsz;
    memcpy(&heap_part->data, v->payload.data, v->fill);

    v->fill = 0;
    v->alloc = new_alloc;
    v->payload.ptr = heap_part;
}

/**
 * Returns a token that can be used to deallocate all of the allocated space
 * space in the dynamic stack.
 */
extern "C" CDECL void *
upcall_dynastack_mark(rust_task *task) {
    return task->dynastack.alloc(0);
}

/** Allocates space in the dynamic stack and returns it. */
extern "C" CDECL void *
upcall_dynastack_alloc(rust_task *task, size_t sz) {
    return sz ? task->dynastack.alloc(sz) : NULL;
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
