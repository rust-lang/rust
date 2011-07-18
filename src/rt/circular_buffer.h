/*
 *
 */

#ifndef CIRCULAR_BUFFER_H
#define CIRCULAR_BUFFER_H

class
circular_buffer : public kernel_owned<circular_buffer> {
    static const size_t INITIAL_CIRCULAR_BUFFER_SIZE_IN_UNITS = 8;
    static const size_t MAX_CIRCULAR_BUFFER_SIZE = 1 << 24;

    rust_scheduler *sched;

public:
    rust_kernel *kernel;
    // Size of the data unit in bytes.
    const size_t unit_sz;
    circular_buffer(rust_kernel *kernel, size_t unit_sz);
    ~circular_buffer();
    void transfer(void *dst);
    void enqueue(void *src);
    void dequeue(void *dst);
    uint8_t *peek();
    bool is_empty();
    size_t size();

private:
    size_t initial_size();
    void grow();
    void shrink();

    // Size of the buffer in bytes.
    size_t _buffer_sz;

    // Byte offset within the buffer where to read the next unit of data.
    size_t _next;

    // Number of bytes that have not been read from the buffer.
    size_t _unread;

    // The buffer itself.
    uint8_t *_buffer;
};

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

#endif /* CIRCULAR_BUFFER_H */
