/*
 *
 */

#ifndef CIRCULAR_BUFFER_H
#define CIRCULAR_BUFFER_H

class
circular_buffer : public dom_owned<circular_buffer> {
    static const size_t INITIAL_CIRCULAR_BUFFFER_SIZE_IN_UNITS = 8;
    static const size_t MAX_CIRCULAR_BUFFFER_SIZE = 1 << 24;

public:
    rust_dom *dom;
    circular_buffer(rust_dom *dom, size_t unit_sz);
    ~circular_buffer();
    void transfer(void *dst);
    void enqueue(void *src);
    void dequeue(void *dst);
    bool is_empty();

private:
    size_t _buffer_sz;
    size_t unit_sz;
    size_t _next;
    size_t _unread;
    uint8_t *_buffer;
};

#endif /* CIRCULAR_BUFFER_H */
