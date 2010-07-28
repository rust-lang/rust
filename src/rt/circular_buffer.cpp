/*
 * A simple resizable circular buffer.
 */

#include "rust_internal.h"

bool
is_power_of_two(size_t value) {
    if (value > 0) {
        return (value & (value - 1)) == 0;
    }
    return false;
}

circular_buffer::circular_buffer(rust_dom *dom, size_t unit_sz) :
    dom(dom),
    unit_sz(unit_sz),
    _buffer_sz(INITIAL_CIRCULAR_BUFFFER_SIZE_IN_UNITS * unit_sz),
    _next(0),
    _unread(0),
    _buffer((uint8_t *)dom->calloc(_buffer_sz)) {

    A(dom, unit_sz, "Unit size must be larger than zero.");

    dom->log(rust_log::MEM | rust_log::COMM,
             "new circular_buffer(buffer_sz=%d, unread=%d)"
             "-> circular_buffer=0x%" PRIxPTR,
             _buffer_sz, _unread, this);

    A(dom, _buffer, "Failed to allocate buffer.");
}

circular_buffer::~circular_buffer() {
    dom->log(rust_log::MEM | rust_log::COMM,
             "~circular_buffer 0x%" PRIxPTR,
             this);
    I(dom, _buffer);
    // I(dom, _unread == 0);
    dom->free(_buffer);
}

/**
 * Copies the unread data from this buffer to the "dst" address.
 */
void
circular_buffer::transfer(void *dst) {
    I(dom, dst);
    I(dom, is_power_of_two(_buffer_sz));
    uint8_t *ptr = (uint8_t *) dst;
    for (size_t i = 0; i < _unread; i += unit_sz) {
        memcpy(&ptr[i], &_buffer[(_next + i) & (_buffer_sz - 1)], unit_sz);
    }
}

/**
 * Copies the data at the "src" address into this buffer. The buffer is
 * grown if it isn't large enough.
 */
void
circular_buffer::enqueue(void *src) {
    I(dom, src);
    I(dom, _unread <= _buffer_sz);

    // Grow if necessary.
    if (_unread == _buffer_sz) {
        I(dom, _buffer_sz <= MAX_CIRCULAR_BUFFFER_SIZE);
        void *tmp = dom->malloc(_buffer_sz << 1);
        transfer(tmp);
        _buffer_sz <<= 1;
        dom->free(_buffer);
        _buffer = (uint8_t *)tmp;
    }

    dom->log(rust_log::MEM | rust_log::COMM,
             "circular_buffer enqueue "
             "unread: %d, buffer_sz: %d, unit_sz: %d",
             _unread, _buffer_sz, unit_sz);

    I(dom, is_power_of_two(_buffer_sz));
    I(dom, _unread < _buffer_sz);
    I(dom, _unread + unit_sz <= _buffer_sz);

    // Copy data
    size_t i = (_next + _unread) & (_buffer_sz - 1);
    memcpy(&_buffer[i], src, unit_sz);
    _unread += unit_sz;

    dom->log(rust_log::MEM | rust_log::COMM,
             "circular_buffer pushed data at index: %d", i);
}

/**
 * Copies data from this buffer to the "dst" address. The buffer is
 * shrunk if possible. If the "dst" address is NULL, then the message
 * is dequeued but is not copied.
 */
void
circular_buffer::dequeue(void *dst) {
    I(dom, unit_sz > 0);
    I(dom, _unread >= unit_sz);
    I(dom, _unread <= _buffer_sz);
    I(dom, _buffer);

    dom->log(rust_log::MEM | rust_log::COMM,
             "circular_buffer dequeue "
             "unread: %d, buffer_sz: %d, unit_sz: %d",
             _unread, _buffer_sz, unit_sz);

    if (dst != NULL) {
        memcpy(dst, &_buffer[_next], unit_sz);
    }
    dom->log(rust_log::MEM | rust_log::COMM,
             "shifted data from index %d", _next);
    _unread -= unit_sz;
    _next += unit_sz;
    I(dom, _next <= _buffer_sz);
    if (_next == _buffer_sz) {
        _next = 0;
    }

    // Shrink if possible.
    if (_buffer_sz >= INITIAL_CIRCULAR_BUFFFER_SIZE_IN_UNITS * unit_sz &&
        _unread <= _buffer_sz / 4) {
        dom->log(rust_log::MEM | rust_log::COMM,
                 "circular_buffer is shrinking to %d bytes", _buffer_sz / 2);
        void *tmp = dom->malloc(_buffer_sz / 2);
        transfer(tmp);
        _buffer_sz >>= 1;
        dom->free(_buffer);
        _buffer = (uint8_t *)tmp;
        _next = 0;
    }

}

uint8_t *
circular_buffer::peek() {
    return &_buffer[_next];
}

bool
circular_buffer::is_empty() {
    return _unread == 0;
}
