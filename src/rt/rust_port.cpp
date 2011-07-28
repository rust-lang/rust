#include "rust_internal.h"
#include "rust_port.h"

rust_port::rust_port(rust_task *task, size_t unit_sz)
    : maybe_proxy<rust_port>(this), kernel(task->kernel), task(task),
      unit_sz(unit_sz), writers(task), chans(task) {

    LOG(task, comm,
        "new rust_port(task=0x%" PRIxPTR ", unit_sz=%d) -> port=0x%"
        PRIxPTR, (uintptr_t)task, unit_sz, (uintptr_t)this);

    // Allocate a remote channel, for remote channel data.
    remote_channel = new (kernel, "remote chan")
        rust_chan(kernel, this, unit_sz);
}

rust_port::~rust_port() {
    LOG(task, comm, "~rust_port 0x%" PRIxPTR, (uintptr_t) this);

    //    log_state();

    // Disassociate channels from this port.
    while (chans.is_empty() == false) {
        scoped_lock with(referent()->lock);
        rust_chan *chan = chans.peek();
        chan->disassociate();

        if (chan->ref_count == 0) {
            LOG(task, comm,
                "chan: 0x%" PRIxPTR " is dormant, freeing", chan);
            delete chan;
        }
    }

    delete remote_channel;
}

bool rust_port::receive(void *dptr) {
    for (uint32_t i = 0; i < chans.length(); i++) {
        rust_chan *chan = chans[i];
        if (chan->buffer.is_empty() == false) {
            chan->buffer.dequeue(dptr);
            LOG(task, comm, "<=== read data ===");
            return true;
        }
    }
    return false;
}

void rust_port::log_state() {
    LOG(task, comm,
              "rust_port: 0x%" PRIxPTR ", associated channel(s): %d",
              this, chans.length());
    for (uint32_t i = 0; i < chans.length(); i++) {
        rust_chan *chan = chans[i];
        LOG(task, comm,
            "\tchan: 0x%" PRIxPTR ", size: %d, remote: %s",
            chan,
            chan->buffer.size(),
            chan == remote_channel ? "yes" : "no");
    }
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
