#include "rust_internal.h"
#include "rust_chan.h"

/**
 * Create a new rust channel and associate it with the specified port.
 */
rust_chan::rust_chan(rust_kernel *kernel, maybe_proxy<rust_port> *port,
                     size_t unit_sz)
    : ref_count(1),
      kernel(kernel),
      port(port),
      buffer(kernel, unit_sz) {
    if (port) {
        associate(port);
    }
    // DLOG(task->sched, comm, "new rust_chan(task=0x%" PRIxPTR
    //     ", port=0x%" PRIxPTR ") -> chan=0x%" PRIxPTR,
    //     (uintptr_t) task, (uintptr_t) port, (uintptr_t) this);
}

rust_chan::~rust_chan() {
    // DLOG(kernel->sched, comm, "del rust_chan(task=0x%" PRIxPTR ")",
    //      (uintptr_t) this);

    // A(kernel->sched, is_associated() == false,
    //   "Channel must be disassociated before being freed.");
}

/**
 * Link this channel with the specified port.
 */
void rust_chan::associate(maybe_proxy<rust_port> *port) {
    this->port = port;
    if (port->is_proxy() == false) {
        scoped_lock with(port->referent()->lock);
        // DLOG(kernel->sched, task,
        //     "associating chan: 0x%" PRIxPTR " with port: 0x%" PRIxPTR,
        //     this, port);
        ++this->ref_count;
        this->task = port->referent()->task;
        this->task->ref();
        this->port->referent()->chans.push(this);
    }
}

bool rust_chan::is_associated() {
    return port != NULL;
}

/**
 * Unlink this channel from its associated port.
 */
void rust_chan::disassociate() {
    // A(kernel->sched, is_associated(),
    //   "Channel must be associated with a port.");

    if (port->is_proxy() == false) {
        scoped_lock with(port->referent()->lock);
        // DLOG(kernel->sched, task,
        //     "disassociating chan: 0x%" PRIxPTR " from port: 0x%" PRIxPTR,
        //     this, port->referent());
        --this->ref_count;
        task->deref();
        this->task = NULL;
        port->referent()->chans.swap_delete(this);
    }

    // Delete reference to the port.
    port = NULL;
}

/**
 * Attempt to send data to the associated port.
 */
void rust_chan::send(void *sptr) {
    // rust_scheduler *sched = kernel->sched;
    // I(sched, !port->is_proxy());

    rust_port *target_port = port->referent();
    // TODO: We can probably avoid this lock by using atomic operations in
    // circular_buffer.
    scoped_lock with(target_port->lock);

    buffer.enqueue(sptr);

    if (!is_associated()) {
        // W(sched, is_associated(),
        //   "rust_chan::transmit with no associated port.");
        return;
    }

    // A(sched, !buffer.is_empty(),
    //   "rust_chan::transmit with nothing to send.");

    if (port->is_proxy()) {
        data_message::send(buffer.peek(), buffer.unit_sz, "send data",
                           task->get_handle(), port->as_proxy()->handle());
        buffer.dequeue(NULL);
    } else {
        if (target_port->task->blocked_on(target_port)) {
            // DLOG(sched, comm, "dequeued in rendezvous_ptr");
            buffer.dequeue(target_port->task->rendezvous_ptr);
            target_port->task->rendezvous_ptr = 0;
            target_port->task->wakeup(target_port);
            return;
        }
    }

    return;
}

rust_chan *rust_chan::clone(rust_task *target) {
    size_t unit_sz = buffer.unit_sz;
    maybe_proxy<rust_port> *port = this->port;
    return new (target->kernel, "cloned chan")
        rust_chan(kernel, port, unit_sz);
}

/**
 * Cannot Yield: If the task were to unwind, the dropped ref would still
 * appear to be live, causing modify-after-free errors.
 */
void rust_chan::destroy() {
    // A(kernel->sched, ref_count == 0,
    //   "Channel's ref count should be zero.");

    if (is_associated()) {
        if (port->is_proxy()) {
            // Here is a good place to delete the port proxy we allocated
            // in upcall_clone_chan.
            rust_proxy<rust_port> *proxy = port->as_proxy();
            disassociate();
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
            // 2. We can leave the channel in a "dormnat" state by not freeing
            //    it and letting the receiver task delete it for us instead.
            if (buffer.is_empty() == false) {
                return;
            }
            disassociate();
        }
    }
    delete this;
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
