
#include "rust_port.h"
#include "rust_task.h"

rust_port::rust_port(rust_task *task, size_t unit_sz)
    : ref_count(1), kernel(task->kernel), task(task),
      unit_sz(unit_sz), buffer(kernel, unit_sz) {

    LOG(task, comm,
        "new rust_port(task=0x%" PRIxPTR ", unit_sz=%d) -> port=0x%"
        PRIxPTR, (uintptr_t)task, unit_sz, (uintptr_t)this);

    id = kernel->register_port(this);
}

rust_port::~rust_port() {
    LOG(task, comm, "~rust_port 0x%" PRIxPTR, (uintptr_t) this);
}

void rust_port::ref() {
    scoped_lock with(ref_lock);
    ref_count++;
}

void rust_port::deref() {
    scoped_lock with(ref_lock);
    ref_count--;
    if (!ref_count) {
        // The port owner is waiting for the port to be detached (if it
        // hasn't already been killed)
        scoped_lock with(task->lifecycle_lock);
        if (task->blocked_on(&detach_cond)) {
            task->wakeup_inner(&detach_cond);
        }
    }
}

void rust_port::begin_detach(uintptr_t *yield) {
    *yield = false;

    kernel->release_port_id(id);

    scoped_lock with(ref_lock);
    ref_count--;

    if (ref_count != 0) {
        task->block(&detach_cond, "waiting for port detach");
        *yield = true;
    }
}

void rust_port::end_detach() {
    // Just take the lock to make sure that the thread that signaled
    // the detach_cond isn't still holding it
    scoped_lock with(ref_lock);
    assert(ref_count == 0);
}

void rust_port::send(void *sptr) {
    bool did_rendezvous = false;
    {
        scoped_lock with(lock);

        buffer.enqueue(sptr);

        assert(!buffer.is_empty() &&
               "rust_chan::transmit with nothing to send.");

        {
            scoped_lock with(task->lifecycle_lock);
            if (task->blocked_on(this)) {
                KLOG(kernel, comm, "dequeued in rendezvous_ptr");
                buffer.dequeue(task->rendezvous_ptr);
                task->rendezvous_ptr = 0;
                task->wakeup_inner(this);
                did_rendezvous = true;
            }
        }
    }

    if (!did_rendezvous) {
        // If the task wasn't waiting specifically on this port,
        // it may be waiting on a group of ports

        rust_port_selector *port_selector = task->get_port_selector();
        // The port selector will check if the task is blocked, not us.
        port_selector->msg_sent_on(this);
    }
}

void rust_port::receive(void *dptr, uintptr_t *yield) {
    LOG(task, comm, "port: 0x%" PRIxPTR ", dptr: 0x%" PRIxPTR
        ", size: 0x%" PRIxPTR,
        (uintptr_t) this, (uintptr_t) dptr, unit_sz);

    scoped_lock with(lock);

    *yield = false;

    if (buffer.is_empty() == false) {
        buffer.dequeue(dptr);
        LOG(task, comm, "<=== read data ===");
        return;
    }
    memset(dptr, 0, buffer.unit_sz);

    // No data was buffered on any incoming channel, so block this task on
    // the port. Remember the rendezvous location so that any sender task
    // can write to it before waking up this task.

    LOG(task, comm, "<=== waiting for rendezvous data ===");
    task->rendezvous_ptr = (uintptr_t*) dptr;
    task->block(this, "waiting for rendezvous data");

    // Blocking the task might fail if the task has already been killed, but
    // in the event of both failure and success the task needs to yield. On
    // success, it yields and waits to be unblocked. On failure it yields and
    // is then fails the task.

    *yield = true;
}

size_t rust_port::size() {
    scoped_lock with(lock);
    return buffer.size();
}

void rust_port::log_state() {
    LOG(task, comm,
        "port size: %d",
        buffer.size());
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
