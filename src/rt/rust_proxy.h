#ifndef RUST_PROXY_H
#define RUST_PROXY_H

/**
 * A proxy object is a wrapper for remote objects. Proxy objects are domain
 * owned and provide a way distinguish between local and remote objects.
 */

template <typename T> struct rust_proxy;

/**
 * The base class of all objects that may delegate.
 */
template <typename T> struct
maybe_proxy : public rc_base<T>, public rust_cond {
protected:
    T *_referent;
public:
    maybe_proxy(T *referent) : _referent(referent) {
        // Nop.
    }

    T *referent() {
        return (T *)_referent;
    }

    bool is_proxy() {
        return _referent != this;
    }

    rust_proxy<T> *as_proxy() {
        return (rust_proxy<T> *) this;
    }

    T *as_referent() {
        return (T *) this;
    }
};

template <typename T> class rust_handle;

/**
 * A proxy object that delegates to another.
 */
template <typename T> struct
rust_proxy : public maybe_proxy<T> {
private:
    bool _strong;
    rust_handle<T> *_handle;
public:
    rust_proxy(rust_handle<T> *handle) :
        maybe_proxy<T> (NULL), _strong(FALSE), _handle(handle) {
        // Nop.
    }

    rust_proxy(T *referent) :
        maybe_proxy<T> (referent), _strong(FALSE), _handle(NULL) {
        // Nop.
    }

    rust_handle<T> *handle() {
        return _handle;
    }
};

class rust_message_queue;
struct rust_task;

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

#endif /* RUST_PROXY_H */
