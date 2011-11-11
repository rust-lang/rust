#include "rust_internal.h"
#include "rust_port.h"
#include "rust_chan.h"


rust_port::rust_port(rust_task *task, size_t unit_sz)
    : ref_count(1), kernel(task->kernel), task(task),
      unit_sz(unit_sz) {

    LOG(task, comm,
        "new rust_port(task=0x%" PRIxPTR ", unit_sz=%d) -> port=0x%"
        PRIxPTR, (uintptr_t)task, unit_sz, (uintptr_t)this);

    id = task->register_port(this);
    remote_chan = new (task->kernel, "rust_chan")
        rust_chan(task->kernel, this, unit_sz);
    remote_chan->ref();
    remote_chan->port = this;
}

rust_port::~rust_port() {
    LOG(task, comm, "~rust_port 0x%" PRIxPTR, (uintptr_t) this);

    {
        scoped_lock with(lock);
        remote_chan->port = NULL;
        remote_chan->deref();
        remote_chan = NULL;
    }

    task->release_port(id);
}

bool rust_port::receive(void *dptr) {
    if (remote_chan->buffer.is_empty() == false) {
        remote_chan->buffer.dequeue(dptr);
        LOG(task, comm, "<=== read data ===");
        return true;
    }
    return false;
}

void rust_port::log_state() {
    LOG(task, comm,
        "\tchan: 0x%" PRIxPTR ", size: %d",
        remote_chan,
        remote_chan->buffer.size());
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
