
#ifndef RUST_CHAN_H
#define RUST_CHAN_H

class rust_chan : public rc_base<rust_chan>, public task_owned<rust_chan> {
public:
    rust_chan(rust_task *task, rust_port *port);
    ~rust_chan();

    rust_task *task;
    rust_port *port;
    circular_buffer buffer;
    size_t idx;           // Index into port->chans.

    // Token belonging to this chan, it will be placed into a port's
    // writers vector if we have something to send to the port.
    rust_token token;

    void disassociate();

    int transmit();
};

#endif /* RUST_CHAN_H */
