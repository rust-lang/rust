/*
 * A simple resizable circular buffer.
 */

#include "rust_internal.h"

circular_buffer::circular_buffer(rust_task *task, size_t unit_sz) :
    sched(task->sched),
    task(task),
    unit_sz(unit_sz),
    _buffer_sz(initial_size()),
    _next(0),
    _unread(0),
    _buffer((uint8_t *)task->malloc(_buffer_sz)) {

    A(sched, unit_sz, "Unit size must be larger than zero.");

    DLOG(sched, mem, "new circular_buffer(buffer_sz=%d, unread=%d)"
         "-> circular_buffer=0x%" PRIxPTR,
         _buffer_sz, _unread, this);

    A(sched, _buffer, "Failed to allocate buffer.");
}

circular_buffer::~circular_buffer() {
    DLOG(sched, mem, "~circular_buffer 0x%" PRIxPTR, this);
    I(sched, _buffer);
    W(sched, _unread == 0,
      "freeing circular_buffer with %d unread bytes", _unread);
    task->free(_buffer);
}

size_t
circular_buffer::initial_size() {
    I(sched, unit_sz > 0);
    return INITIAL_CIRCULAR_BUFFER_SIZE_IN_UNITS * unit_sz;
}

/**
 * Copies the unread data from this buffer to the "dst" address.
 */
void
circular_buffer::transfer(void *dst) {
    I(sched, dst);
    I(sched, _unread <= _buffer_sz);

    uint8_t *ptr = (uint8_t *) dst;

    // First copy from _next to either the end of the unread
    // items or the end of the buffer
    size_t head_sz;
    if (_next + _unread <= _buffer_sz) {
        head_sz = _unread;
    } else {
        head_sz = _buffer_sz - _next;
    }
    I(sched, _next + head_sz <= _buffer_sz);
    memcpy(ptr, _buffer + _next, head_sz);

    // Then copy any other items from the beginning of the buffer
    I(sched, _unread >= head_sz);
    size_t tail_sz = _unread - head_sz;
    I(sched, head_sz + tail_sz <= _buffer_sz);
    memcpy(ptr + head_sz, _buffer, tail_sz);
}

/**
 * Copies the data at the "src" address into this buffer. The buffer is
 * grown if it isn't large enough.
 */
void
circular_buffer::enqueue(void *src) {
    I(sched, src);
    I(sched, _unread <= _buffer_sz);
    I(sched, _buffer);

    // Grow if necessary.
    if (_unread == _buffer_sz) {
        grow();
    }

    DLOG(sched, mem, "circular_buffer enqueue "
         "unread: %d, next: %d, buffer_sz: %d, unit_sz: %d",
         _unread, _next, _buffer_sz, unit_sz);

    I(sched, _unread < _buffer_sz);
    I(sched, _unread + unit_sz <= _buffer_sz);

    // Copy data
    size_t dst_idx = _next + _unread;
    I(sched, dst_idx >= _buffer_sz || dst_idx + unit_sz <= _buffer_sz);
    if (dst_idx >= _buffer_sz) {
        dst_idx -= _buffer_sz;

        I(sched, _next >= unit_sz);
        I(sched, dst_idx <= _next - unit_sz);
    }

    I(sched, dst_idx + unit_sz <= _buffer_sz);
    memcpy(&_buffer[dst_idx], src, unit_sz);
    _unread += unit_sz;

    DLOG(sched, mem, "circular_buffer pushed data at index: %d", dst_idx);
}

/**
 * Copies data from this buffer to the "dst" address. The buffer is
 * shrunk if possible. If the "dst" address is NULL, then the message
 * is dequeued but is not copied.
 */
void
circular_buffer::dequeue(void *dst) {
    I(sched, unit_sz > 0);
    I(sched, _unread >= unit_sz);
    I(sched, _unread <= _buffer_sz);
    I(sched, _buffer);

    DLOG(sched, mem,
             "circular_buffer dequeue "
             "unread: %d, next: %d, buffer_sz: %d, unit_sz: %d",
             _unread, _next, _buffer_sz, unit_sz);

    I(sched, _next + unit_sz <= _buffer_sz);
    if (dst != NULL) {
        memcpy(dst, &_buffer[_next], unit_sz);
    }
    DLOG(sched, mem, "shifted data from index %d", _next);
    _unread -= unit_sz;
    _next += unit_sz;
    if (_next == _buffer_sz) {
        _next = 0;
    }

    // Shrink if possible.
    if (_buffer_sz > initial_size() && _unread <= _buffer_sz / 4) {
        shrink();
    }
}

void
circular_buffer::grow() {
    size_t new_buffer_sz = _buffer_sz * 2;
    I(sched, new_buffer_sz <= MAX_CIRCULAR_BUFFER_SIZE);
    DLOG(sched, mem, "circular_buffer is growing to %d bytes", new_buffer_sz);
    void *new_buffer = task->malloc(new_buffer_sz);
    transfer(new_buffer);
    task->free(_buffer);
    _buffer = (uint8_t *)new_buffer;
    _next = 0;
    _buffer_sz = new_buffer_sz;
}

void
circular_buffer::shrink() {
    size_t new_buffer_sz = _buffer_sz / 2;
    I(sched, initial_size() <= new_buffer_sz);
    DLOG(sched, mem, "circular_buffer is shrinking to %d bytes",
         new_buffer_sz);
    void *new_buffer = task->malloc(new_buffer_sz);
    transfer(new_buffer);
    task->free(_buffer);
    _buffer = (uint8_t *)new_buffer;
    _next = 0;
    _buffer_sz = new_buffer_sz;
}

uint8_t *
circular_buffer::peek() {
    return &_buffer[_next];
}

bool
circular_buffer::is_empty() {
    return _unread == 0;
}

size_t
circular_buffer::size() {
    return _unread;
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
