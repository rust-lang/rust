
#include "rust_internal.h"
#include "rust_chan.h"

rust_chan::rust_chan(rust_task *task, rust_port *port) :
    task(task),
    port(port),
    buffer(task->dom, port->unit_sz),
    token(this)
{
    if (port)
        port->chans.push(this);
}

rust_chan::~rust_chan()
{
    if (port) {
        if (token.pending())
            token.withdraw();
        port->chans.swapdel(this);
    }
}

void
rust_chan::disassociate()
{
    I(task->dom, port);

    if (token.pending())
        token.withdraw();

    // Delete reference to the port/
    port = NULL;
}
