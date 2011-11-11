#include "rust_internal.h"
#include "rust_chan.h"

/**
 * Create a new rust channel and associate it with the specified port.
 */
rust_chan::rust_chan(rust_kernel *kernel, rust_port *port,
                     size_t unit_sz)
    : ref_count(0),
      kernel(kernel),
      port(port),
      buffer(kernel, unit_sz) {
    KLOG(kernel, comm, "new rust_chan(task=0x%" PRIxPTR
        ", port=0x%" PRIxPTR ") -> chan=0x%" PRIxPTR,
        (uintptr_t) task, (uintptr_t) port, (uintptr_t) this);

    A(kernel, port != NULL, "Port must not be null");
    this->task = port->task;
    this->task->ref();
}

rust_chan::~rust_chan() {
    KLOG(kernel, comm, "del rust_chan(task=0x%" PRIxPTR ")",
         (uintptr_t) this);

    I(this->kernel, !is_associated());

    A(kernel, is_associated() == false,
      "Channel must be disassociated before being freed.");

    task->deref();
    task = NULL;
}

bool rust_chan::is_associated() {
    return port != NULL;
}

rust_chan *rust_chan::clone(rust_task *target) {
    return new (target->kernel, "cloned chan")
        rust_chan(kernel, port, buffer.unit_sz);
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
