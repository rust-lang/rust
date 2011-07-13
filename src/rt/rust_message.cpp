#include "rust_internal.h"
#include "rust_message.h"

rust_message::
rust_message(memory_region *region, const char* label,
             rust_handle<rust_task> *source, rust_handle<rust_task> *target) :
             label(label), region(region), _source(source), _target(target) {
}

rust_message::~rust_message() {
}

void rust_message::process() {
}

void rust_message::kernel_process() {
}

notify_message::
notify_message(memory_region *region, notification_type type,
    const char* label, rust_handle<rust_task> *source,
    rust_handle<rust_task> *target) :
    rust_message(region, label, source, target), type(type) {
}

data_message::
data_message(memory_region *region, uint8_t *buffer, size_t buffer_sz,
             const char* label, rust_handle<rust_task> *source,
             rust_handle<rust_port> *port) :
             rust_message(region, label, source, NULL),
             _buffer_sz(buffer_sz), _port(port) {
    _buffer = (uint8_t *)malloc(buffer_sz);
    memcpy(_buffer, buffer, buffer_sz);
}

data_message::~data_message() {
    free (_buffer);
}

/**
 * Sends a message to the target task via a proxy. The message is allocated
 * in the target task domain along with a proxy which points back to the
 * source task.
 */
void notify_message::
send(notification_type type, const char* label,
     rust_handle<rust_task> *source, rust_handle<rust_task> *target) {
    memory_region *region = &target->message_queue->region;
    notify_message *message =
        new (region) notify_message(region, type, label, source, target);
    target->message_queue->enqueue(message);
}

void notify_message::process() {
    rust_task *task = _target->referent();
    switch (type) {
    case KILL:
        // task->ref_count--;
        task->kill();
        break;
    case JOIN: {
        if (task->dead() == false) {
            rust_proxy<rust_task> *proxy = new rust_proxy<rust_task>(_source);
            task->tasks_waiting_to_join.append(proxy);
        } else {
            send(WAKEUP, "wakeup", _target, _source);
        }
        break;
    }
    case WAKEUP:
        task->wakeup(_source);
        break;
    }
}

void notify_message::kernel_process() {
    switch(type) {
    case WAKEUP:
    case KILL:
        // Ignore.
        break;
    case JOIN:
        send(WAKEUP, "wakeup", _target, _source);
        break;
    }
}

void data_message::
send(uint8_t *buffer, size_t buffer_sz, const char* label,
     rust_handle<rust_task> *source, rust_handle<rust_port> *port) {

    memory_region *region = &port->message_queue->region;
    data_message *message =
        new (region) data_message(region, buffer, buffer_sz, label, source,
                                  port);
    LOG(source->referent(), comm, "==> sending \"%s\"" PTR " in queue " PTR,
        label, message, &port->message_queue);
    port->message_queue->enqueue(message);
}

void data_message::process() {
    _port->referent()->remote_channel->send(_buffer);
}

void data_message::kernel_process() {

}

rust_message_queue::rust_message_queue(rust_srv *srv, rust_kernel *kernel) 
    : region(srv, true),
      kernel(kernel),
      sched_handle(NULL) {
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
