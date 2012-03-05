#ifndef RUST_PORT_H
#define RUST_PORT_H

#include "rust_internal.h"

class port_detach_cond : public rust_cond { };

class rust_port : public kernel_owned<rust_port>, public rust_cond {
public:
    RUST_ATOMIC_REFCOUNT();

    rust_port_id id;

    rust_kernel *kernel;
    rust_task *task;
    size_t unit_sz;
    circular_buffer buffer;

    lock_and_signal lock;

private:
    // Protects blocking and signaling on detach_cond
    lock_and_signal detach_lock;
    port_detach_cond detach_cond;

public:
    rust_port(rust_task *task, size_t unit_sz);
    ~rust_port();
    void delete_this();

    void log_state();
    void send(void *sptr);
    void receive(void *dptr, uintptr_t *yield);
    size_t size();

    void begin_detach(uintptr_t *yield);
    void end_detach();
};

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

#endif /* RUST_PORT_H */
