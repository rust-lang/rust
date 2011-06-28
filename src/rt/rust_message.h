#ifndef RUST_MESSAGE_H
#define RUST_MESSAGE_H

/**
 * Rust messages are used for inter-thread communication. They are enqueued
 * and allocated in the target domain.
 */

/**
 * Abstract base class for all message types.
 */
class rust_message : public region_owned<rust_message> {
public:
    const char* label;
    memory_region *region;
private:
protected:
    rust_handle<rust_task> *_source;
    rust_handle<rust_task> *_target;
public:
    rust_message(memory_region *region,
                 const char* label,
                 rust_handle<rust_task> *source,
                 rust_handle<rust_task> *target);

    virtual ~rust_message();

    /**
     * Processes the message in the target domain.
     */
    virtual void process();

    /**
     * Processes the message in the kernel.
     */
    virtual void kernel_process();
};

/**
 * Notify messages are simple argument-less messages.
 */
class notify_message : public rust_message {
public:
    enum notification_type {
        KILL, JOIN, WAKEUP
    };

    const notification_type type;

    notify_message(memory_region *region, notification_type type,
                   const char* label, rust_handle<rust_task> *source,
                   rust_handle<rust_task> *target);

    void process();
    void kernel_process();

    /**
     * This code executes in the sending domain's thread.
     */
    static void
    send(notification_type type, const char* label,
         rust_handle<rust_task> *source, rust_handle<rust_task> *target);
};

/**
 * Data messages carry a buffer.
 */
class data_message : public rust_message {
private:
    uint8_t *_buffer;
    size_t _buffer_sz;
    rust_handle<rust_port> *_port;

public:
    data_message(memory_region *region, uint8_t *buffer, size_t buffer_sz,
                 const char* label, rust_handle<rust_task> *source,
                 rust_handle<rust_port> *port);

    virtual ~data_message();
    void process();
    void kernel_process();

    /**
     * This code executes in the sending domain's thread.
     */
    static void
    send(uint8_t *buffer, size_t buffer_sz, const char* label,
         rust_handle<rust_task> *source, rust_handle<rust_port> *port);
};

class rust_message_queue : public lock_free_queue<rust_message*>,
                           public kernel_owned<rust_message_queue> {
public:
    memory_region region;
    rust_kernel *kernel;
    rust_handle<rust_scheduler> *sched_handle;
    int32_t list_index;
    rust_message_queue(rust_srv *srv, rust_kernel *kernel);

    void associate(rust_handle<rust_scheduler> *sched_handle) {
        this->sched_handle = sched_handle;
    }

    /**
     * The Rust domain relinquishes control to the Rust kernel.
     */
    void disassociate() {
        this->sched_handle = NULL;
    }

    /**
     * Checks if a Rust domain is responsible for draining the message queue.
     */
    bool is_associated() {
        return this->sched_handle != NULL;
    }

    void enqueue(rust_message* message) {
        lock_free_queue<rust_message*>::enqueue(message);
        kernel->notify_message_enqueued(this, message);
    }
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

#endif /* RUST_MESSAGE_H */
