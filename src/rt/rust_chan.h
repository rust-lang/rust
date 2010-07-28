
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

#endif /* RUST_CHAN_H */
