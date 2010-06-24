
#include "rust_internal.h"

template class ptr_vec<rust_token>;
template class ptr_vec<rust_alarm>;
template class ptr_vec<rust_chan>;

rust_alarm::rust_alarm(rust_task *receiver) :
    receiver(receiver)
{
}


// Circular buffers.

circ_buf::circ_buf(rust_dom *dom, size_t unit_sz) :
    dom(dom),
    alloc(INIT_CIRC_BUF_UNITS * unit_sz),
    unit_sz(unit_sz),
    next(0),
    unread(0),
    data((uint8_t *)dom->calloc(alloc))
{
    I(dom, unit_sz);
    dom->log(rust_log::MEM|rust_log::COMM,
             "new circ_buf(alloc=%d, unread=%d) -> circ_buf=0x%" PRIxPTR,
             alloc, unread, this);
    I(dom, data);
}

circ_buf::~circ_buf()
{
    dom->log(rust_log::MEM|rust_log::COMM,
             "~circ_buf 0x%" PRIxPTR,
             this);
    I(dom, data);
    // I(dom, unread == 0);
    dom->free(data);
}

void
circ_buf::transfer(void *dst)
{
    size_t i;
    uint8_t *d = (uint8_t *)dst;
    I(dom, dst);
    for (i = 0; i < unread; i += unit_sz)
        memcpy(&d[i], &data[next + i % alloc], unit_sz);
}

void
circ_buf::push(void *src)
{
    size_t i;
    void *tmp;

    I(dom, src);
    I(dom, unread <= alloc);

    /* Grow if necessary. */
    if (unread == alloc) {
        I(dom, alloc <= MAX_CIRC_BUF_SIZE);
        tmp = dom->malloc(alloc << 1);
        transfer(tmp);
        alloc <<= 1;
        dom->free(data);
        data = (uint8_t *)tmp;
    }

    dom->log(rust_log::MEM|rust_log::COMM,
             "circ buf push, unread=%d, alloc=%d, unit_sz=%d",
             unread, alloc, unit_sz);

    I(dom, unread < alloc);
    I(dom, unread + unit_sz <= alloc);

    i = (next + unread) % alloc;
    memcpy(&data[i], src, unit_sz);

    dom->log(rust_log::MEM|rust_log::COMM, "pushed data at index %d", i);
    unread += unit_sz;
}

void
circ_buf::shift(void *dst)
{
    size_t i;
    void *tmp;

    I(dom, dst);
    I(dom, unit_sz > 0);
    I(dom, unread >= unit_sz);
    I(dom, unread <= alloc);
    I(dom, data);
    i = next;
    memcpy(dst, &data[i], unit_sz);
    dom->log(rust_log::MEM|rust_log::COMM, "shifted data from index %d", i);
    unread -= unit_sz;
    next += unit_sz;
    I(dom, next <= alloc);
    if (next == alloc)
        next = 0;

    /* Shrink if necessary. */
    if (alloc >= INIT_CIRC_BUF_UNITS * unit_sz &&
        unread <= alloc / 4) {
        tmp = dom->malloc(alloc / 2);
        transfer(tmp);
        alloc >>= 1;
        dom->free(data);
        data = (uint8_t *)tmp;
    }
}


// Ports.

rust_port::rust_port(rust_task *task, size_t unit_sz) :
    task(task),
    unit_sz(unit_sz),
    writers(task->dom),
    chans(task->dom)
{
    rust_dom *dom = task->dom;
    dom->log(rust_log::MEM|rust_log::COMM,
             "new rust_port(task=0x%" PRIxPTR ", unit_sz=%d) -> port=0x%"
             PRIxPTR, (uintptr_t)task, unit_sz, (uintptr_t)this);
}

rust_port::~rust_port()
{
    rust_dom *dom = task->dom;
    dom->log(rust_log::COMM|rust_log::MEM,
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
    port->writers.swapdel(this);
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
