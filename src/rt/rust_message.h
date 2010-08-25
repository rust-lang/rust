#ifndef RUST_MESSAGE_H
#define RUST_MESSAGE_H

/**
 * Rust messages are used for inter-thread communication. They are enqueued
 * and allocated in the target domain.
 */

/**
 * Abstract base class for all message types.
 */
class rust_message {
public:
    const char* label;
private:
    rust_dom *_dom;
    rust_task *_source;
protected:
    rust_task *_target;
public:
    rust_message(const char* label, rust_task *source, rust_task *target);
    virtual ~rust_message();

    /**
     * We can only access the source task through a proxy, so create one
     * on demand if we need it.
     */
    rust_proxy<rust_task> *get_source_proxy();

    /**
     * Processes the message in the target domain thread.
     */
    virtual void process();
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

    notify_message(notification_type type, const char* label,
                   rust_task *source, rust_task *target);

    void process();

    /**
     * This code executes in the sending domain's thread.
     */
    static void
    send(notification_type type, const char* label, rust_task *source,
         rust_proxy<rust_task> *target);
};

/**
 * Data messages carry a buffer.
 */
class data_message : public rust_message {
private:
    uint8_t *_buffer;
    size_t _buffer_sz;
    rust_port *_port;
public:

    data_message(uint8_t *buffer, size_t buffer_sz, const char* label,
                 rust_task *source, rust_task *target, rust_port *port);
    virtual ~data_message();
    void process();

    /**
     * This code executes in the sending domain's thread.
     */
    static void
    send(uint8_t *buffer, size_t buffer_sz, const char* label,
         rust_task *source, rust_proxy<rust_task> *target,
         rust_proxy<rust_port> *port);
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
