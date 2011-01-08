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
    // Size of the data unit in bytes.
    const size_t unit_sz;
    circular_buffer(rust_dom *dom, size_t unit_sz);
    ~circular_buffer();
    void transfer(void *dst);
    void enqueue(void *src);
    void dequeue(void *dst);
    uint8_t *peek();
    bool is_empty();
    size_t size();

private:
    // Initial size of the buffer in bytes.
    size_t _initial_sz;

    // Size of the buffer in bytes, should always be a power of two so that
    // modulo arithmetic (x % _buffer_sz) can optimized away with
    // (x & (_buffer_sz - 1)).
    size_t _buffer_sz;

    // Byte offset within the buffer where to read the next unit of data.
    size_t _next;

    // Number of bytes that have not been read from the buffer.
    size_t _unread;

    // The buffer itself.
    uint8_t *_buffer;
};

#endif /* CIRCULAR_BUFFER_H */
