#include "rust_internal.h"

// Upcalls.

#ifdef __GNUC__
#define LOG_UPCALL_ENTRY(task)                            \
    LOG(task, upcall,                                     \
        "> UPCALL %s - task: %s 0x%" PRIxPTR              \
        " retpc: x%" PRIxPTR                              \
        " ref_count: %d",                                 \
        __FUNCTION__,                                     \
        (task)->name, (task),                             \
        __builtin_return_address(0),                      \
        (task->ref_count));
#else
#define LOG_UPCALL_ENTRY(task)                            \
    LOG(task, upcall, "> UPCALL task: %s @x%" PRIxPTR,    \
        (task)->name, (task));
#endif

extern "C" CDECL char const *
str_buf(rust_task *task, rust_str *s);

extern "C" void
upcall_grow_task(rust_task *task, size_t n_frame_bytes) {
    I(task->dom, false);
    LOG_UPCALL_ENTRY(task);
    task->grow(n_frame_bytes);
}

extern "C" CDECL
void upcall_log_int(rust_task *task, uint32_t level, int32_t i) {
    LOG_UPCALL_ENTRY(task);
    if (task->dom->log_lvl >= level)
        task->dom->log(task, level, "rust: %" PRId32 " (0x%" PRIx32 ")",
                       i, i);
}

extern "C" CDECL
void upcall_log_float(rust_task *task, uint32_t level, float f) {
    LOG_UPCALL_ENTRY(task);
    if (task->dom->log_lvl >= level)
        task->dom->log(task, level, "rust: %12.12f", f);
}

extern "C" CDECL
void upcall_log_double(rust_task *task, uint32_t level, double *f) {
    LOG_UPCALL_ENTRY(task);
    if (task->dom->log_lvl >= level)
        task->dom->log(task, level, "rust: %12.12f", *f);
}

extern "C" CDECL void
upcall_log_str(rust_task *task, uint32_t level, rust_str *str) {
    LOG_UPCALL_ENTRY(task);
    if (task->dom->log_lvl >= level) {
        const char *c = str_buf(task, str);
        task->dom->log(task, level, "rust: %s", c);
    }
}

extern "C" CDECL void
upcall_trace_word(rust_task *task, uintptr_t i) {
    LOG_UPCALL_ENTRY(task);
    task->dom->log(task, 2, "trace: 0x%" PRIxPTR "", i, i, (char) i);
}

extern "C" CDECL void
upcall_trace_str(rust_task *task, char const *c) {
    LOG_UPCALL_ENTRY(task);
    task->dom->log(task, 2, "trace: %s", c);
}

extern "C" CDECL rust_port*
upcall_new_port(rust_task *task, size_t unit_sz) {
    LOG_UPCALL_ENTRY(task);
    rust_dom *dom = task->dom;
    scoped_lock with(dom->scheduler_lock);
    LOG(task, comm, "upcall_new_port(task=0x%" PRIxPTR " (%s), unit_sz=%d)",
        (uintptr_t) task, task->name, unit_sz);
    return new (dom) rust_port(task, unit_sz);
}

extern "C" CDECL void
upcall_del_port(rust_task *task, rust_port *port) {
    LOG_UPCALL_ENTRY(task);
    scoped_lock with(task->dom->scheduler_lock);
    LOG(task, comm, "upcall del_port(0x%" PRIxPTR ")", (uintptr_t) port);
    I(task->dom, !port->ref_count);
    delete port;
}

/**
 * Creates a new channel pointing to a given port.
 */
extern "C" CDECL rust_chan*
upcall_new_chan(rust_task *task, rust_port *port) {
    LOG_UPCALL_ENTRY(task);
    rust_dom *dom = task->dom;
    LOG(task, comm, "upcall_new_chan("
        "task=0x%" PRIxPTR " (%s), port=0x%" PRIxPTR ")",
        (uintptr_t) task, task->name, port);
    I(dom, port);
    return new (dom) rust_chan(task, port, port->unit_sz);
}

/**
 * Called whenever this channel needs to be flushed. This can happen due to a
 * flush statement, or automatically whenever a channel's ref count is
 * about to drop to zero.
 */
extern "C" CDECL void
upcall_flush_chan(rust_task *task, rust_chan *chan) {
    LOG_UPCALL_ENTRY(task);
    // Nop.
}

/**
 * Called whenever the channel's ref count drops to zero.
 *
 * Cannot Yield: If the task were to unwind, the dropped ref would still
 * appear to be live, causing modify-after-free errors.
 */
extern "C" CDECL
void upcall_del_chan(rust_task *task, rust_chan *chan) {
    LOG_UPCALL_ENTRY(task);
    scoped_lock with(task->dom->scheduler_lock);

    LOG(task, comm, "upcall del_chan(0x%" PRIxPTR ")", (uintptr_t) chan);

    A(task->dom, chan->ref_count == 0,
      "Channel's ref count should be zero.");

    if (chan->is_associated()) {
        if (chan->port->is_proxy()) {
            // Here is a good place to delete the port proxy we allocated
            // in upcall_clone_chan.
            rust_proxy<rust_port> *proxy = chan->port->as_proxy();
            chan->disassociate();
            delete proxy;
        } else {
            // We're trying to delete a channel that another task may be
            // reading from. We have two options:
            //
            // 1. We can flush the channel by blocking in upcall_flush_chan()
            //    and resuming only when the channel is flushed. The problem
            //    here is that we can get ourselves in a deadlock if the
            //    parent task tries to join us.
            //
            // 2. We can leave the channel in a "dormant" state by not freeing
            //    it and letting the receiver task delete it for us instead.
            if (chan->buffer.is_empty() == false) {
                return;
            }
            chan->disassociate();
        }
    }
    delete chan;
}

/**
 * Clones a channel and stores it in the spawnee's domain. Each spawned task
 * has its own copy of the channel.
 */
extern "C" CDECL rust_chan *
upcall_clone_chan(rust_task *task, maybe_proxy<rust_task> *target,
                  rust_chan *chan) {
    LOG_UPCALL_ENTRY(task);
    scoped_lock with(task->dom->scheduler_lock);
    size_t unit_sz = chan->buffer.unit_sz;
    maybe_proxy<rust_port> *port = chan->port;
    rust_task *target_task = NULL;
    if (target->is_proxy() == false) {
        port = chan->port;
        target_task = target->referent();
    } else {
        rust_handle<rust_port> *handle =
            task->dom->kernel->get_port_handle(port->as_referent());
        maybe_proxy<rust_port> *proxy = new rust_proxy<rust_port> (handle);
        LOG(task, mem, "new proxy: " PTR, proxy);
        port = proxy;
        target_task = target->as_proxy()->handle()->referent();
    }
    return new (target_task->dom) rust_chan(target_task, port, unit_sz);
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
    LOG(task, task, "elapsed %d",
              task->yield_timer.get_elapsed_time());
    LOG(task, task, "sleep %d us", time_in_us);
    task->yield(2, time_in_us);
}

/**
 * Buffers a chunk of data in the specified channel.
 *
 * sptr: pointer to a chunk of data to buffer
 */
extern "C" CDECL void
upcall_send(rust_task *task, rust_chan *chan, void *sptr) {
    LOG_UPCALL_ENTRY(task);
    scoped_lock with(task->dom->scheduler_lock);
    chan->send(sptr);
    LOG(task, comm, "=== sent data ===>");
}

extern "C" CDECL void
upcall_recv(rust_task *task, uintptr_t *dptr, rust_port *port) {
    {
        LOG_UPCALL_ENTRY(task);
        scoped_lock with(task->dom->scheduler_lock);

        LOG(task, comm, "port: 0x%" PRIxPTR ", dptr: 0x%" PRIxPTR
            ", size: 0x%" PRIxPTR ", chan_no: %d",
            (uintptr_t) port, (uintptr_t) dptr, port->unit_sz,
            port->chans.length());

        if (port->receive(dptr)) {
            return;
        }

        // No data was buffered on any incoming channel, so block this task
        // on the port. Remember the rendezvous location so that any sender
        // task can write to it before waking up this task.

        LOG(task, comm, "<=== waiting for rendezvous data ===");
        task->rendezvous_ptr = dptr;
        task->block(port, "waiting for rendezvous data");
    }
    task->yield(3);
}

extern "C" CDECL void
upcall_fail(rust_task *task,
            char const *expr,
            char const *file,
            size_t line) {
    LOG_UPCALL_ENTRY(task);
    LOG_ERR(task, upcall, "upcall fail '%s', %s:%" PRIdPTR, expr, file, line);
    task->fail(4);
}

/**
 * Called whenever a task's ref count drops to zero.
 */
extern "C" CDECL void
upcall_kill(rust_task *task, maybe_proxy<rust_task> *target) {
    LOG_UPCALL_ENTRY(task);
    scoped_lock with(task->dom->scheduler_lock);
    if (target->is_proxy()) {
        notify_message::
        send(notify_message::KILL, "kill", task->get_handle(),
             target->as_proxy()->handle());
        // The proxy ref_count dropped to zero, delete it here.
        delete target->as_proxy();
    } else {
        target->referent()->kill();
    }
}

/**
 * Called by the exit glue when the task terminates.
 */
extern "C" CDECL void
upcall_exit(rust_task *task) {
    {
        LOG_UPCALL_ENTRY(task);
        scoped_lock with(task->dom->scheduler_lock);
        LOG(task, task, "task ref_count: %d", task->ref_count);
        A(task->dom, task->ref_count >= 0,
          "Task ref_count should not be negative on exit!");
        task->die();
        task->notify_tasks_waiting_to_join();
    }
    task->yield(1);
}

extern "C" CDECL uintptr_t
upcall_malloc(rust_task *task, size_t nbytes, type_desc *td) {
    LOG_UPCALL_ENTRY(task);
    scoped_lock with(task->dom->scheduler_lock);

    LOG(task, mem,
                   "upcall malloc(%" PRIdPTR ", 0x%" PRIxPTR ")"
                   " with gc-chain head = 0x%" PRIxPTR,
                   nbytes, td, task->gc_alloc_chain);
    void *p = task->malloc(nbytes, td);
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
    scoped_lock with(task->dom->scheduler_lock);
    rust_dom *dom = task->dom;
    DLOG(dom, mem,
             "upcall free(0x%" PRIxPTR ", is_gc=%" PRIdPTR ")",
             (uintptr_t)ptr, is_gc);
    task->free(ptr, (bool) is_gc);
}

extern "C" CDECL uintptr_t
upcall_mark(rust_task *task, void* ptr) {
    LOG_UPCALL_ENTRY(task);
    scoped_lock with(task->dom->scheduler_lock);

    rust_dom *dom = task->dom;
    if (ptr) {
        gc_alloc *gcm = (gc_alloc*) (((char*)ptr) - sizeof(gc_alloc));
        uintptr_t marked = (uintptr_t) gcm->mark();
        DLOG(dom, gc, "upcall mark(0x%" PRIxPTR ") = %" PRIdPTR,
                 (uintptr_t)gcm, marked);
        return marked;
    }
    return 0;
}

rust_str *make_str(rust_task *task, char const *s, size_t fill) {
    rust_dom *dom = task->dom;
    size_t alloc = next_power_of_two(sizeof(rust_str) + fill);
    void *mem = task->malloc(alloc);
    if (!mem) {
        task->fail(3);
        return NULL;
    }
    rust_str *st = new (mem) rust_str(dom, alloc, fill, (uint8_t const *) s);
    LOG(task, mem,
        "upcall new_str('%s', %" PRIdPTR ") = 0x%" PRIxPTR,
        s, fill, st);
    return st;
}

extern "C" CDECL rust_str *
upcall_new_str(rust_task *task, char const *s, size_t fill) {
    LOG_UPCALL_ENTRY(task);
    scoped_lock with(task->dom->scheduler_lock);
    
    return make_str(task, s, fill);
}

extern "C" CDECL rust_str *
upcall_dup_str(rust_task *task, rust_str *str) {
    LOG_UPCALL_ENTRY(task);
    scoped_lock with(task->dom->scheduler_lock);

    return make_str(task, (char const *)str->data, str->fill);
}

extern "C" CDECL rust_vec *
upcall_new_vec(rust_task *task, size_t fill, type_desc *td) {
    LOG_UPCALL_ENTRY(task);
    scoped_lock with(task->dom->scheduler_lock);
    rust_dom *dom = task->dom;
    DLOG(dom, mem, "upcall new_vec(%" PRIdPTR ")", fill);
    size_t alloc = next_power_of_two(sizeof(rust_vec) + fill);
    void *mem = task->malloc(alloc, td);
    if (!mem) {
        task->fail(3);
        return NULL;
    }
    rust_vec *v = new (mem) rust_vec(dom, alloc, 0, NULL);
    LOG(task, mem,
              "upcall new_vec(%" PRIdPTR ") = 0x%" PRIxPTR, fill, v);
    return v;
}

static rust_vec *
vec_grow(rust_task *task,
         rust_vec *v,
         size_t n_bytes,
         uintptr_t *need_copy,
         type_desc *td)
{
    rust_dom *dom = task->dom;
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
            task->fail(4);
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
        void *mem = task->malloc(alloc, td);
        if (!mem) {
            task->fail(4);
            return NULL;
        }

        if (v->ref_count != CONST_REFCOUNT)
            v->deref();

        v = new (mem) rust_vec(dom, alloc, 0, NULL);
        *need_copy = 1;
    }
    I(dom, sizeof(rust_vec) + v->fill <= v->alloc);
    return v;
}

// Copy elements from one vector to another,
// dealing with reference counts
static inline void
copy_elements(rust_task *task, type_desc *elem_t,
              void *pdst, void *psrc, size_t n)
{
    char *dst = (char *)pdst, *src = (char *)psrc;

    // increment the refcount of each element of the vector
    if (elem_t->take_glue) {
        glue_fn *take_glue = elem_t->take_glue;
        size_t elem_size = elem_t->size;
        const type_desc **tydescs = elem_t->first_param;
        for (char *p = src; p < src+n; p += elem_size) {
            take_glue(NULL, task, NULL, tydescs, p);
        }
    }
    memmove(dst, src, n);
}

extern "C" CDECL void
upcall_vec_append(rust_task *task, type_desc *t, type_desc *elem_t,
                  rust_vec **dst_ptr, rust_vec *src, bool skip_null)
{
    LOG_UPCALL_ENTRY(task);
    scoped_lock with(task->dom->scheduler_lock);
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
    LOG_UPCALL_ENTRY(task);
    scoped_lock with(task->dom->scheduler_lock);
    LOG(task, cache, "upcall get_type_desc with size=%" PRIdPTR
        ", align=%" PRIdPTR ", %" PRIdPTR " descs", size, align,
        n_descs);
    rust_crate_cache *cache = task->get_crate_cache();
    type_desc *td = cache->get_type_desc(size, align, n_descs, descs);
    LOG(task, cache, "returning tydesc 0x%" PRIxPTR, td);
    return td;
}

extern "C" CDECL rust_task *
upcall_new_task(rust_task *spawner, rust_vec *name) {
    // name is a rust string structure.
    LOG_UPCALL_ENTRY(spawner);
    scoped_lock with(spawner->dom->scheduler_lock);
    rust_dom *dom = spawner->dom;
    rust_task *task = dom->create_task(spawner, (const char *)name->data);
    return task;
}

extern "C" CDECL rust_task *
upcall_start_task(rust_task *spawner,
                  rust_task *task,
                  uintptr_t spawnee_fn,
                  uintptr_t args,
                  size_t args_sz) {
    LOG_UPCALL_ENTRY(spawner);

    rust_dom *dom = spawner->dom;
    DLOG(dom, task,
         "upcall start_task(task %s @0x%" PRIxPTR
         ", spawnee 0x%" PRIxPTR ")", 
         task->name, task,
         spawnee_fn);

    // we used to be generating this tuple in rustc, but it's easier to do it
    // here.
    //
    // The args tuple is stack-allocated. We need to move it over to the new
    // stack.
    task->rust_sp -= args_sz;
    uintptr_t child_arg = (uintptr_t)task->rust_sp;

    memcpy((void*)task->rust_sp, (void*)args, args_sz);

    task->start(spawnee_fn, child_arg);
    return task;
}

/**
 * Resizes an interior vector that has been spilled to the heap.
 */
extern "C" CDECL void
upcall_ivec_resize(rust_task *task,
                   rust_ivec *v,
                   size_t newsz) {
    scoped_lock with(task->dom->scheduler_lock);
    I(task->dom, !v->fill);

    size_t new_alloc = next_power_of_two(newsz);
    rust_ivec_heap *new_heap_part = (rust_ivec_heap *)
        task->realloc(v->payload.ptr, new_alloc);

    new_heap_part->fill = newsz;
    v->alloc = new_alloc;
    v->payload.ptr = new_heap_part;
}

/**
 * Spills an interior vector to the heap.
 */
extern "C" CDECL void
upcall_ivec_spill(rust_task *task,
                  rust_ivec *v,
                  size_t newsz) {
    scoped_lock with(task->dom->scheduler_lock);
    size_t new_alloc = next_power_of_two(newsz);

    rust_ivec_heap *heap_part = (rust_ivec_heap *)
        task->malloc(new_alloc + sizeof(size_t));
    heap_part->fill = newsz;
    memcpy(&heap_part->data, v->payload.data, v->fill);

    v->fill = 0;
    v->alloc = new_alloc;
    v->payload.ptr = heap_part;
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
