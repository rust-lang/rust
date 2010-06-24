#ifndef RUST_UTIL_H
#define RUST_UTIL_H

// Reference counted objects

template <typename T>
rc_base<T>::rc_base() :
    refcnt(1)
{
}

template <typename T>
rc_base<T>::~rc_base()
{
}

// Utility type: pointer-vector.

template <typename T>
ptr_vec<T>::ptr_vec(rust_dom *dom) :
    dom(dom),
    alloc(INIT_SIZE),
    fill(0),
    data(new (dom) T*[alloc])
{
    I(dom, data);
    dom->log(rust_log::MEM,
             "new ptr_vec(data=0x%" PRIxPTR ") -> 0x%" PRIxPTR,
             (uintptr_t)data, (uintptr_t)this);
}

template <typename T>
ptr_vec<T>::~ptr_vec()
{
    I(dom, data);
    dom->log(rust_log::MEM,
             "~ptr_vec 0x%" PRIxPTR ", data=0x%" PRIxPTR,
             (uintptr_t)this, (uintptr_t)data);
    I(dom, fill == 0);
    dom->free(data);
}

template <typename T> T *&
ptr_vec<T>::operator[](size_t offset) {
    I(dom, data[offset]->idx == offset);
    return data[offset];
}

template <typename T>
void
ptr_vec<T>::push(T *p)
{
    I(dom, data);
    I(dom, fill <= alloc);
    if (fill == alloc) {
        alloc *= 2;
        data = (T **)dom->realloc(data, alloc * sizeof(T*));
        I(dom, data);
    }
    I(dom, fill < alloc);
    p->idx = fill;
    data[fill++] = p;
}

template <typename T>
T *
ptr_vec<T>::pop()
{
    return data[--fill];
}

template <typename T>
void
ptr_vec<T>::trim(size_t sz)
{
    I(dom, data);
    if (sz <= (alloc / 4) &&
        (alloc / 2) >= INIT_SIZE) {
        alloc /= 2;
        I(dom, alloc >= fill);
        data = (T **)dom->realloc(data, alloc * sizeof(T*));
        I(dom, data);
    }
}

template <typename T>
void
ptr_vec<T>::swapdel(T *item)
{
    /* Swap the endpoint into i and decr fill. */
    I(dom, data);
    I(dom, fill > 0);
    I(dom, item->idx < fill);
    fill--;
    if (fill > 0) {
        T *subst = data[fill];
        size_t idx = item->idx;
        data[idx] = subst;
        subst->idx = idx;
    }
}

// Inline fn used regularly elsewhere.

static inline size_t
next_power_of_two(size_t s)
{
    size_t tmp = s - 1;
    tmp |= tmp >> 1;
    tmp |= tmp >> 2;
    tmp |= tmp >> 4;
    tmp |= tmp >> 8;
    tmp |= tmp >> 16;
#if SIZE_MAX == UINT64_MAX
    tmp |= tmp >> 32;
#endif
    return tmp + 1;
}

// Vectors (rust-user-code level).

struct
rust_vec : public rc_base<rust_vec>
{
    size_t alloc;
    size_t fill;
    uint8_t data[];
    rust_vec(rust_dom *dom, size_t alloc, size_t fill, uint8_t const *d) :
        alloc(alloc),
        fill(fill)
    {
        if (d || fill) {
            I(dom, d);
            I(dom, fill);
            memcpy(&data[0], d, fill);
        }
    }
    ~rust_vec() {}
};

// Rust types vec and str look identical from our perspective.
typedef rust_vec rust_str;

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

#endif
