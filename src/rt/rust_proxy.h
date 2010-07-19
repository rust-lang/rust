/**
 * A proxy object is a wrapper around other Rust objects. One use of the proxy
 * object is to mitigate access between tasks in different thread domains.
 */

#ifndef RUST_PROXY_H
#define RUST_PROXY_H

template <typename T> struct
rust_proxy_delegate : public rc_base<T> {
protected:
    T *_delegate;
public:
    rust_proxy_delegate(T * delegate) : _delegate(delegate) {
    }
    T *delegate() { return _delegate; }
};

template <typename T> struct
rust_proxy : public rust_proxy_delegate<T>,
             public dom_owned<rust_proxy<T> > {
public:
    rust_dom *dom;
    rust_proxy(rust_dom *dom, T *delegate) :
        rust_proxy_delegate<T> (delegate),
        dom(dom) {
        delegate->ref();
    }
};

#endif /* RUST_PROXY_H */
