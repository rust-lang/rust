#include "rust_internal.h"
#include "rust_port.h"

rust_port::rust_port(rust_task *task, size_t unit_sz) :
    maybe_proxy<rust_port>(this), task(task), unit_sz(unit_sz),
    writers(task->dom), chans(task->dom) {

    task->log(rust_log::MEM | rust_log::COMM,
              "new rust_port(task=0x%" PRIxPTR ", unit_sz=%d) -> port=0x%"
              PRIxPTR, (uintptr_t)task, unit_sz, (uintptr_t)this);

    // Allocate a remote channel, for remote channel data.
    remote_channel = new (task->dom) rust_chan(task, this);
}

rust_port::~rust_port() {
    task->log(rust_log::COMM | rust_log::MEM,
              "~rust_port 0x%" PRIxPTR, (uintptr_t) this);

    // Disassociate channels from this port.
    while (chans.is_empty() == false) {
        chans.pop()->disassociate();
    }

    // We're the only ones holding a reference to the remote channel, so
    // clean it up.
    delete remote_channel;
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
