#ifndef RUST_CHAN_H
#define RUST_CHAN_H

class rust_chan : public kernel_owned<rust_chan>,
                  public rust_cond {
    ~rust_chan();

public:
    RUST_ATOMIC_REFCOUNT();
    rust_chan(rust_kernel *kernel, rust_port *port,
              size_t unit_sz);

    rust_kernel *kernel;
    rust_task *task;
    rust_port *port;
    size_t idx;
    circular_buffer buffer;

    void associate(rust_port *port);
    void disassociate();
    bool is_associated();

    void send(void *sptr);

    rust_chan *clone(rust_task *target);
};

// Corresponds to the rust chan (currently _chan) type.
struct chan_handle {
    rust_task_id task;
    rust_port_id port;
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

#endif /* RUST_CHAN_H */
