#ifndef RUST_CHAN_H
#define RUST_CHAN_H

class rust_chan : public kernel_owned<rust_chan>,
                  public rust_cond {
public:
    RUST_REFCOUNTED_WITH_DTOR(rust_chan, destroy())
    rust_chan(rust_task *task, maybe_proxy<rust_port> *port, size_t unit_sz);

    ~rust_chan();

    rust_kernel *kernel;
    smart_ptr<rust_task> task;
    maybe_proxy<rust_port> *port;
    size_t idx;
    circular_buffer buffer;

    void associate(maybe_proxy<rust_port> *port);
    void disassociate();
    bool is_associated();

    void send(void *sptr);

    rust_chan *clone(maybe_proxy<rust_task> *target);

    // Called whenever the channel's ref count drops to zero.
    void destroy();
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
