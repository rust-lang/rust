
#include "rust_internal.h"

template class ptr_vec<rust_token>;
template class ptr_vec<rust_alarm>;
template class ptr_vec<rust_chan>;

rust_alarm::rust_alarm(rust_task *receiver) :
    receiver(receiver)
{
}

// Ports.

rust_port::rust_port(rust_task *task, size_t unit_sz) :
    task(task),
    unit_sz(unit_sz),
    writers(task->dom),
    chans(task->dom)
{
    task->log(rust_log::MEM|rust_log::COMM,
              "new rust_port(task=0x%" PRIxPTR ", unit_sz=%d) -> port=0x%"
              PRIxPTR, (uintptr_t)task, unit_sz, (uintptr_t)this);
}

rust_port::~rust_port()
{
    task->log(rust_log::COMM|rust_log::MEM,
              "~rust_port 0x%" PRIxPTR,
              (uintptr_t)this);
    while (chans.length() > 0)
        chans.pop()->disassociate();
}


// Tokens.

rust_token::rust_token(rust_chan *chan) :
    chan(chan),
    idx(0),
    submitted(false)
{
}

rust_token::~rust_token()
{
}

bool
rust_token::pending() const
{
    return submitted;
}

void
rust_token::submit()
{
    rust_port *port = chan->port;
    rust_dom *dom = chan->task->dom;

    I(dom, port);
    I(dom, !submitted);

    port->writers.push(this);
    submitted = true;
}

void
rust_token::withdraw()
{
    rust_task *task = chan->task;
    rust_port *port = chan->port;
    rust_dom *dom = task->dom;

    I(dom, port);
    I(dom, submitted);

    if (task->blocked())
        task->wakeup(this); // must be blocked on us (or dead)
    port->writers.swap_delete(this);
    submitted = false;
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
