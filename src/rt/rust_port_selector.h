#ifndef RUST_PORT_SELECTOR_H
#define RUST_PORT_SELECTOR_H

#include "rust_internal.h"

struct rust_task;
class rust_port;

class rust_port_selector : public rust_cond {
 private:
    rust_port **ports;
    size_t n_ports;
    lock_and_signal rendezvous_lock;

 public:
    rust_port_selector();

    void select(rust_task *task,
                rust_port **dptr,
                rust_port **ports,
                size_t n_ports,
                uintptr_t *yield);

    void msg_sent_on(rust_port *port);
};

#endif /* RUST_PORT_SELECTOR_H */
