#include "rust_internal.h"
#include "rust_message.h"

rust_message::
rust_message(const char* label, rust_task *source, rust_task *target) :
             dom(target->dom), label(label),
             _source(source),
             _target(target) {
}

rust_message::~rust_message() {
}

void rust_message::process() {
    I(dom, false);
}

rust_proxy<rust_task> *
rust_message::get_source_proxy() {
    return dom->get_task_proxy(_source);
}

notify_message::
notify_message(notification_type type, const char* label,
               rust_task *source,
               rust_task *target) :
               rust_message(label, source, target), type(type) {
}

/**
 * Sends a message to the target task via a proxy. The message is allocated
 * in the target task domain along with a proxy which points back to the
 * source task.
 */
void notify_message::
send(notification_type type, const char* label, rust_task *source,
     rust_proxy<rust_task> *target) {
    rust_task *target_task = target->delegate();
    rust_dom *target_domain = target_task->dom;
    notify_message *message = new (target_domain)
        notify_message(type, label, source, target_task);
    target_domain->send_message(message);
}

void notify_message::process() {
    rust_task *task = _target;
    switch (type) {
    case KILL:
        task->ref_count--;
        task->kill();
        break;
    case JOIN: {
        if (task->dead() == false) {
            task->tasks_waiting_to_join.append(get_source_proxy());
        } else {
            send(WAKEUP, "wakeup", task, get_source_proxy());
        }
        break;
    }
    case WAKEUP:
        task->wakeup(get_source_proxy()->delegate());
        break;
    }
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
