#include "rust_internal.h"
#include "rust_chan.h"

rust_chan::rust_chan(rust_task *task, rust_port *port) :
    task(task), port(port), buffer(task->dom, port->unit_sz), token(this) {

    if (port) {
        port->chans.push(this);
        ref();
    }

    task->log(rust_log::MEM | rust_log::COMM,
              "new rust_chan(task=0x%" PRIxPTR
              ", port=0x%" PRIxPTR ") -> chan=0x%"
              PRIxPTR, (uintptr_t) task, (uintptr_t) port, (uintptr_t) this);
}

rust_chan::~rust_chan() {
    if (port) {
        if (token.pending())
            token.withdraw();
        port->chans.swap_delete(this);
    }
}

void rust_chan::disassociate() {
    I(task->dom, port);

    if (token.pending())
        token.withdraw();

    // Delete reference to the port/
    port = NULL;

    deref();
}

/**
 * Attempt to transmit channel data to the associated port.
 */
int rust_chan::transmit() {
    rust_dom *dom = task->dom;

    // TODO: Figure out how and why the port would become null.
    if (port == NULL) {
        dom->log(rust_log::COMM, "invalid port, transmission incomplete");
        return ERROR;
    }

    if (buffer.is_empty()) {
        dom->log(rust_log::COMM, "buffer is empty, transmission incomplete");
        return ERROR;
    }

    if(port->task->blocked_on(port)) {
        buffer.dequeue(port->task->rendezvous_ptr);
        port->task->wakeup(port);
    }

    return 0;

}
