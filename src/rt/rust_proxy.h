#ifndef RUST_PROXY_H
#define RUST_PROXY_H

/**
 * A proxy object is a wrapper around other Rust objects. One use of the proxy
 * object is to mitigate access between tasks in different thread domains.
 */

template <typename T> struct rust_proxy;
/**
 * The base class of all objects that may delegate.
 */
template <typename T> struct
maybe_proxy : public rc_base<T>, public rust_cond {
protected:
    T *_delegate;
public:
    maybe_proxy(T * delegate) : _delegate(delegate) {

    }
    T *delegate() {
        return _delegate;
    }
    bool is_proxy() {
        return _delegate != this;
    }
    rust_proxy<T> *as_proxy() {
        return (rust_proxy<T> *) this;
    }
    T *as_delegate() {
        I(_delegate->get_dom(), !is_proxy());
        return (T *) this;
    }
};

/**
 * A proxy object that delegates to another.
 */
template <typename T> struct
rust_proxy : public maybe_proxy<T>,
             public dom_owned<rust_proxy<T> > {
private:
    bool _strong;
public:
    rust_dom *dom;
    rust_proxy(rust_dom *dom, T *delegate, bool strong) :
        maybe_proxy<T> (delegate), _strong(strong), dom(dom) {
        this->dom->log(rust_log::COMM,
            "new proxy: 0x%" PRIxPTR " => 0x%" PRIxPTR, this, delegate);
        if (strong) {
            delegate->ref();
        }
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

#endif /* RUST_PROXY_H */
