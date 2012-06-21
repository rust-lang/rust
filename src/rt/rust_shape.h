// Functions that interpret the shape of a type to perform various low-level
// actions, such as copying, freeing, comparing, and so on.

#ifndef RUST_SHAPE_H
#define RUST_SHAPE_H

// Tell ISAAC to let go of max() and min() defines.
#undef max
#undef min

#include <iostream>

#include "rust_globals.h"
#include "rust_util.h"

// ISAAC pollutes our namespace.
#undef align

#define ARENA_SIZE          256

//#define DPRINT(fmt,...)     fprintf(stderr, fmt, ##__VA_ARGS__)
//#define DPRINTCX(cx)        shape::print::print_cx(cx)

#define DPRINT(fmt,...)
#define DPRINTCX(cx)


namespace shape {

typedef unsigned long tag_variant_t;
typedef unsigned long tag_align_t;

// Constants

const uint8_t SHAPE_U8 = 0u;
const uint8_t SHAPE_U16 = 1u;
const uint8_t SHAPE_U32 = 2u;
const uint8_t SHAPE_U64 = 3u;
const uint8_t SHAPE_I8 = 4u;
const uint8_t SHAPE_I16 = 5u;
const uint8_t SHAPE_I32 = 6u;
const uint8_t SHAPE_I64 = 7u;
const uint8_t SHAPE_F32 = 8u;
const uint8_t SHAPE_F64 = 9u;
const uint8_t SHAPE_BOX = 10u;
const uint8_t SHAPE_TAG = 12u;
const uint8_t SHAPE_STRUCT = 17u;
const uint8_t SHAPE_BOX_FN = 18u;
const uint8_t SHAPE_RES = 20u;
const uint8_t SHAPE_UNIQ = 22u;
const uint8_t SHAPE_UNIQ_FN = 25u;
const uint8_t SHAPE_STACK_FN = 26u;
const uint8_t SHAPE_BARE_FN = 27u;
const uint8_t SHAPE_TYDESC = 28u;
const uint8_t SHAPE_SEND_TYDESC = 29u;
const uint8_t SHAPE_RPTR = 31u;
const uint8_t SHAPE_FIXEDVEC = 32u;
const uint8_t SHAPE_SLICE = 33u;
const uint8_t SHAPE_UNBOXED_VEC = 34u;

#ifdef _LP64
const uint8_t SHAPE_PTR = SHAPE_U64;
#else
const uint8_t SHAPE_PTR = SHAPE_U32;
#endif


// Forward declarations

struct rust_obj;
struct size_align;
class ptr;


// Arenas; these functions must execute very quickly, so we use an arena
// instead of malloc or new.

class arena {
    uint8_t *ptr;
    uint8_t data[ARENA_SIZE];

public:
    arena() : ptr(data) {}

    template<typename T>
    inline T *alloc(size_t count = 1) {
        // FIXME: align (probably won't fix before #1498)
        size_t sz = count * sizeof(T);
        T *rv = (T *)ptr;
        ptr += sz;
        if (ptr > &data[ARENA_SIZE]) {
            fprintf(stderr, "Arena space exhausted, sorry\n");
            abort();
        }
        return rv;
    }
};


// Alignment inquiries
//
// We can't directly use __alignof__ everywhere because that returns the
// preferred alignment of the type, which is different from the ABI-mandated
// alignment of the type in some cases (e.g. doubles on x86). The latter is
// what actually gets used for struct elements.

template<typename T>
inline size_t
rust_alignof() {
#ifdef _MSC_VER
    return __alignof(T);
#else
    return __alignof__(T);
#endif
}

template<>
inline size_t
rust_alignof<double>() {
    return 4;
}

// Issue #2303
// On 32-bit x86 the alignment of 64-bit ints in structures is 4 bytes
// Which is different from the preferred 8-byte alignment reported
// by __alignof__ (at least on gcc).
#ifndef __WIN32__
#ifdef __i386__
template<>
inline size_t
rust_alignof<uint64_t>() {
    return 4;
}
#endif
#endif


// Utility classes

struct size_align {
    size_t size;
    size_t alignment;

    size_align(size_t in_size = 0, size_t in_align = 1) :
        size(in_size), alignment(in_align) {}

    bool is_set() const { return alignment != 0; }

    inline void set(size_t in_size, size_t in_align) {
        size = in_size;
        alignment = in_align;
    }

    inline void add(const size_align &other) {
        add(other.size, other.alignment);
    }

    inline void add(size_t extra_size, size_t extra_align) {
        size += extra_size;
        alignment = std::max(alignment, extra_align);
    }

    static inline size_align make(size_t in_size) {
        size_align sa;
        sa.size = sa.alignment = in_size;
        return sa;
    }

    static inline size_align make(size_t in_size, size_t in_align) {
        size_align sa;
        sa.size = in_size;
        sa.alignment = in_align;
        return sa;
    }
};

struct tag_info {
    uint16_t tag_id;                        // The tag ID.
    const uint8_t *info_ptr;                // Pointer to the info table.
    uint16_t variant_count;                 // Number of variants in the tag.
    const uint8_t *largest_variants_ptr;    // Ptr to largest variants table.
    size_align tag_sa;                      // Size and align of this tag.
};


// Utility functions

inline uint16_t
get_u16(const uint8_t *addr) {
    return *reinterpret_cast<const uint16_t *>(addr);
}

inline uint16_t
get_u16_bump(const uint8_t *&addr) {
    uint16_t result = get_u16(addr);
    addr += sizeof(uint16_t);
    return result;
}

template<typename T>
inline void
fmt_number(std::ostream &out, T n) {
    out << n;
}

// Override the character interpretation for these two.
template<>
inline void
fmt_number<uint8_t>(std::ostream &out, uint8_t n) {
    out << (int)n;
}
template<>
inline void
fmt_number<int8_t>(std::ostream &out, int8_t n) {
    out << (int)n;
}


// Contexts

// The base context, an abstract class. We use the curiously recurring
// template pattern here to avoid virtual dispatch.
template<typename T>
class ctxt {
public:
    const uint8_t *sp;                  // shape pointer
    const rust_shape_tables *tables;
    rust_task *task;
    bool align;

    ctxt(rust_task *in_task,
         bool in_align,
         const uint8_t *in_sp,
         const rust_shape_tables *in_tables)
    : sp(in_sp),
      tables(in_tables),
      task(in_task),
      align(in_align) {}

    template<typename U>
    ctxt(const ctxt<U> &other,
         const uint8_t *in_sp = NULL,
         const rust_shape_tables *in_tables = NULL)
    : sp(in_sp ? in_sp : other.sp),
      tables(in_tables ? in_tables : other.tables),
      task(other.task),
      align(other.align) {}

    void walk();
    void walk_reset();

    std::pair<const uint8_t *,const uint8_t *>
    get_variant_sp(tag_info &info, tag_variant_t variant_id);

    const char *
    get_variant_name(tag_info &info, tag_variant_t variant_id);

protected:
    inline uint8_t peek() { return *sp; }

    inline size_align get_size_align(const uint8_t *&addr);

private:
    void walk_vec0();
    void walk_unboxed_vec0();
    void walk_tag0();
    void walk_box0();
    void walk_uniq0();
    void walk_struct0();
    void walk_res0();
    void walk_rptr0();
    void walk_fixedvec0();
    void walk_slice0();
};


// Core Rust types

struct rust_fn {
    void (*code)(uint8_t *rv, rust_task *task, void *env, ...);
    void *env;
};

// Traversals

#define WALK_NUMBER(c_type) \
    static_cast<T *>(this)->template walk_number1<c_type>()
#define WALK_SIMPLE(method) static_cast<T *>(this)->method()

template<typename T>
void
ctxt<T>::walk() {
  char s = *sp++;
  switch (s) {
    case SHAPE_U8:       WALK_NUMBER(uint8_t);       break;
    case SHAPE_U16:      WALK_NUMBER(uint16_t);      break;
    case SHAPE_U32:      WALK_NUMBER(uint32_t);      break;
    case SHAPE_U64:      WALK_NUMBER(uint64_t);      break;
    case SHAPE_I8:       WALK_NUMBER(int8_t);        break;
    case SHAPE_I16:      WALK_NUMBER(int16_t);       break;
    case SHAPE_I32:      WALK_NUMBER(int32_t);       break;
    case SHAPE_I64:      WALK_NUMBER(int64_t);       break;
    case SHAPE_F32:      WALK_NUMBER(float);         break;
    case SHAPE_F64:      WALK_NUMBER(double);        break;
    case SHAPE_TAG:      walk_tag0();             break;
    case SHAPE_BOX:      walk_box0();             break;
    case SHAPE_STRUCT:   walk_struct0();          break;
    case SHAPE_RES:      walk_res0();             break;
    case SHAPE_UNIQ:     walk_uniq0();            break;
    case SHAPE_BOX_FN:
    case SHAPE_UNIQ_FN:
    case SHAPE_STACK_FN:
    case SHAPE_BARE_FN:  static_cast<T*>(this)->walk_fn1(s); break;
    case SHAPE_TYDESC:
    case SHAPE_SEND_TYDESC: static_cast<T*>(this)->walk_tydesc1(s); break;
    case SHAPE_RPTR:     walk_rptr0();            break;
    case SHAPE_FIXEDVEC: walk_fixedvec0();        break;
    case SHAPE_SLICE:    walk_slice0();           break;
    case SHAPE_UNBOXED_VEC: walk_unboxed_vec0();  break;
    default:             abort();
    }
}

template<typename T>
void
ctxt<T>::walk_reset() {
    const uint8_t *old_sp = sp;
    walk();
    sp = old_sp;
}

template<typename T>
size_align
ctxt<T>::get_size_align(const uint8_t *&addr) {
    size_align result;
    result.size = get_u16_bump(addr);
    result.alignment = *addr++;
    return result;
}

// Returns a pointer to the beginning and a pointer to the end of the shape of
// the tag variant with the given ID.
template<typename T>
std::pair<const uint8_t *,const uint8_t *>
ctxt<T>::get_variant_sp(tag_info &tinfo, tag_variant_t variant_id) {
    uint16_t variant_offset = get_u16(tinfo.info_ptr +
                                      variant_id * sizeof(uint16_t));
    const uint8_t *variant_ptr = tables->tags + variant_offset;
    uint16_t variant_len = get_u16_bump(variant_ptr);
    const uint8_t *variant_end = variant_ptr + variant_len;

    return std::make_pair(variant_ptr, variant_end);
}

template<typename T>
const char *
ctxt<T>::get_variant_name(tag_info &tinfo, tag_variant_t variant_id) {
    std::pair<const uint8_t *,const uint8_t *> variant_ptr_and_end =
      this->get_variant_sp(tinfo, variant_id);
    // skip over the length to get the null-terminated string:
    return (const char*)(variant_ptr_and_end.second + 2);
}

template<typename T>
void
ctxt<T>::walk_vec0() {
    bool is_pod = *sp++;

    uint16_t sp_size = get_u16_bump(sp);
    const uint8_t *end_sp = sp + sp_size;

    static_cast<T *>(this)->walk_vec1(is_pod);

    sp = end_sp;
}

template<typename T>
void
ctxt<T>::walk_unboxed_vec0() {
    bool is_pod = *sp++;

    uint16_t sp_size = get_u16_bump(sp);
    const uint8_t *end_sp = sp + sp_size;

    static_cast<T *>(this)->walk_unboxed_vec1(is_pod);

    sp = end_sp;
}

template<typename T>
void
ctxt<T>::walk_tag0() {
    tag_info tinfo;
    tinfo.tag_id = get_u16_bump(sp);

    // Determine the info pointer.
    uint16_t info_offset = get_u16(tables->tags +
                                   tinfo.tag_id * sizeof(uint16_t));
    tinfo.info_ptr = tables->tags + info_offset;

    tinfo.variant_count = get_u16_bump(tinfo.info_ptr);

    // Determine the largest-variants pointer.
    uint16_t largest_variants_offset = get_u16_bump(tinfo.info_ptr);
    tinfo.largest_variants_ptr = tables->tags + largest_variants_offset;

    // Determine the size and alignment.
    tinfo.tag_sa = get_size_align(tinfo.info_ptr);

    // Read in a dummy value; this used to be the number of parameters
    uint16_t number_of_params = get_u16_bump(sp);
    assert(number_of_params == 0 && "tag has type parameters on it");

    // Call to the implementation.
    static_cast<T *>(this)->walk_tag1(tinfo);
}

template<typename T>
void
ctxt<T>::walk_box0() {
    static_cast<T *>(this)->walk_box1();
}

template<typename T>
void
ctxt<T>::walk_uniq0() {
    uint16_t sp_size = get_u16_bump(sp);
    const uint8_t *end_sp = sp + sp_size;

    static_cast<T *>(this)->walk_uniq1();

    sp = end_sp;
}

template<typename T>
void
ctxt<T>::walk_rptr0() {
    uint16_t sp_size = get_u16_bump(sp);
    const uint8_t *end_sp = sp + sp_size;

    static_cast<T *>(this)->walk_rptr1();

    sp = end_sp;
}

template<typename T>
void
ctxt<T>::walk_fixedvec0() {
    uint16_t n_elts = get_u16_bump(sp);
    bool is_pod = *sp++;
    uint16_t sp_size = get_u16_bump(sp);
    const uint8_t *end_sp = sp + sp_size;

    static_cast<T *>(this)->walk_fixedvec1(n_elts, is_pod);

    sp = end_sp;
}

template<typename T>
void
ctxt<T>::walk_slice0() {
    bool is_pod = *sp++;
    bool is_str = *sp++;
    uint16_t sp_size = get_u16_bump(sp);
    const uint8_t *end_sp = sp + sp_size;

    static_cast<T *>(this)->walk_slice1(is_pod, is_str);

    sp = end_sp;
}


template<typename T>
void
ctxt<T>::walk_struct0() {
    uint16_t sp_size = get_u16_bump(sp);
    const uint8_t *end_sp = sp + sp_size;

    static_cast<T *>(this)->walk_struct1(end_sp);

    sp = end_sp;
}

template<typename T>
void
ctxt<T>::walk_res0() {
    uint16_t dtor_offset = get_u16_bump(sp);
    const rust_fn **resources =
        reinterpret_cast<const rust_fn **>(tables->resources);
    const rust_fn *dtor = resources[dtor_offset];

    // Read in a dummy value; this used to be the number of parameters
    uint16_t number_of_params = get_u16_bump(sp);
    assert(number_of_params == 0 && "resource has type parameters on it");

    uint16_t sp_size = get_u16_bump(sp);
    const uint8_t *end_sp = sp + sp_size;

    static_cast<T *>(this)->walk_res1(dtor, end_sp);

    sp = end_sp;
}

// A shape printer, useful for debugging

class print : public ctxt<print> {
public:
    template<typename T>
    print(const ctxt<T> &other,
          const uint8_t *in_sp = NULL,
          const rust_shape_tables *in_tables = NULL)
    : ctxt<print>(other, in_sp, in_tables) {}

    print(rust_task *in_task,
          bool in_align,
          const uint8_t *in_sp,
          const rust_shape_tables *in_tables)
    : ctxt<print>(in_task, in_align, in_sp, in_tables) {}

    void walk_tag1(tag_info &tinfo);
    void walk_struct1(const uint8_t *end_sp);
    void walk_res1(const rust_fn *dtor, const uint8_t *end_sp);

    void walk_vec1(bool is_pod) {
        DPRINT("vec<"); walk(); DPRINT(">");
    }
    void walk_unboxed_vec1(bool is_pod) {
        DPRINT("unboxed_vec<"); walk(); DPRINT(">");
    }
    void walk_uniq1() {
        DPRINT("~<"); walk(); DPRINT(">");
    }
    void walk_box1() {
        DPRINT("@<"); walk(); DPRINT(">");
    }
    void walk_rptr1() {
        DPRINT("&<"); walk(); DPRINT(">");
    }
    void walk_fixedvec1(uint16_t n_elts, bool is_pod) {
      DPRINT("fixedvec<%u, ", n_elts); walk(); DPRINT(">");
    }
    void walk_slice1(bool is_pod, bool is_str) {
      DPRINT("slice<"); walk(); DPRINT(">");
    }

    void walk_fn1(char kind) {
        switch(kind) {
          case SHAPE_BARE_FN:  DPRINT("fn");  break;
          case SHAPE_BOX_FN:   DPRINT("fn@"); break;
          case SHAPE_UNIQ_FN:  DPRINT("fn~"); break;
          case SHAPE_STACK_FN: DPRINT("fn&"); break;
          default: abort();
        }
    }
    void walk_iface1() { DPRINT("iface"); }

    void walk_tydesc1(char kind) {
        switch(kind) {
          case SHAPE_TYDESC: DPRINT("tydesc"); break;
          case SHAPE_SEND_TYDESC: DPRINT("send-tydesc"); break;
          default: abort();
        }
    }

    template<typename T>
    void walk_number1() {}

    template<typename T>
    static void print_cx(const T *cx) {
        print self(*cx);
        self.align = false;
        self.walk();
    }
};


//
// Size-of (which also computes alignment). Be warned: this is an expensive
// operation.
//
// TODO: Maybe dynamic_size_of() should call into this somehow?
//

class size_of : public ctxt<size_of> {
private:
    size_align sa;

public:
    size_of(const size_of &other,
            const uint8_t *in_sp = NULL,
            const rust_shape_tables *in_tables = NULL)
    : ctxt<size_of>(other, in_sp, in_tables) {}

    template<typename T>
    size_of(const ctxt<T> &other,
            const uint8_t *in_sp = NULL,
            const rust_shape_tables *in_tables = NULL)
    : ctxt<size_of>(other, in_sp, in_tables) {}

    void walk_tag1(tag_info &tinfo);
    void walk_struct1(const uint8_t *end_sp);

    void walk_uniq1()       { sa.set(sizeof(void *),   sizeof(void *)); }
    void walk_rptr1()       { sa.set(sizeof(void *),   sizeof(void *)); }
    void walk_slice1(bool,bool)
                            { sa.set(sizeof(void *)*2, sizeof(void *)); }
    void walk_box1()        { sa.set(sizeof(void *),   sizeof(void *)); }
    void walk_fn1(char)     { sa.set(sizeof(void *)*2, sizeof(void *)); }
    void walk_iface1()      { sa.set(sizeof(void *),   sizeof(void *)); }
    void walk_tydesc1(char) { sa.set(sizeof(void *),   sizeof(void *)); }
    void walk_closure1();

    void walk_vec1(bool is_pod) {
        sa.set(sizeof(void *), sizeof(void *));
    }

    void walk_unboxed_vec1(bool is_pod) {
        assert(false &&
               "trying to compute size of dynamically sized unboxed vector");
    }

    void walk_res1(const rust_fn *dtor, const uint8_t *end_sp) {
        abort();    // TODO
    }

    void walk_fixedvec1(uint16_t n_elts, bool is_pod) {
        size_of sub(*this);
        sub.walk();
        sa.set(sub.sa.size * n_elts, sub.sa.alignment);
    }

    template<typename T>
    void walk_number1()  { sa.set(sizeof(T), rust_alignof<T>()); }

    void compute_tag_size(tag_info &tinfo);

    template<typename T>
    static void compute_tag_size(const ctxt<T> &other_cx, tag_info &tinfo) {
        size_of cx(other_cx);
        cx.compute_tag_size(tinfo);
    }

    template<typename T>
    static size_align get(const ctxt<T> &other_cx, unsigned back_up = 0) {
        size_of cx(other_cx, other_cx.sp - back_up);
        cx.align = false;
        cx.walk();
        assert(cx.sa.alignment > 0);
        return cx.sa;
    }
};


// Pointer wrappers for data traversals

class ptr {
private:
    uint8_t *p;

public:
    template<typename T>
    struct data { typedef T t; };

    ptr() : p(NULL) {}
    explicit ptr(uint8_t *in_p) : p(in_p) {}
    explicit ptr(uintptr_t in_p) : p((uint8_t *)in_p) {}

    inline ptr operator+(const size_t amount) const {
        return make(p + amount);
    }
    inline ptr &operator+=(const size_t amount) { p += amount; return *this; }
    inline bool operator<(const ptr other) { return p < other.p; }
    inline ptr operator++() { ptr rv(*this); p++; return rv; }
    inline uint8_t operator*() { return *p; }

    template<typename T>
    inline operator T *() { return (T *)p; }

    inline operator bool() const { return p != NULL; }
    inline operator uintptr_t() const { return (uintptr_t)p; }

    inline const type_desc *box_body_td() const {
        rust_opaque_box *box = *reinterpret_cast<rust_opaque_box**>(p);
        assert(box->ref_count >= 1);
        return box->td;
    }

    inline const type_desc *uniq_body_td() const {
        rust_opaque_box *box = *reinterpret_cast<rust_opaque_box**>(p);
        return box->td;
    }

    inline ptr box_body() const {
        rust_opaque_box *box = *reinterpret_cast<rust_opaque_box**>(p);
        return make((uint8_t*)::box_body(box));
    }

    static inline ptr make(uint8_t *in_p) {
        ptr self(in_p);
        return self;
    }
};

template<typename T>
static inline T
bump_dp(ptr &dp) {
    T x = *((T *)dp);
    dp += sizeof(T);
    return x;
}

template<typename T>
static inline T
get_dp(ptr dp) {
    return *((T *)dp);
}


// Pointer pairs for structural comparison

template<typename T>
class data_pair {
public:
    T fst, snd;

    data_pair() {}
    data_pair(T &in_fst, T &in_snd) : fst(in_fst), snd(in_snd) {}

    inline void operator=(const T rhs) { fst = snd = rhs; }

    static data_pair<T> make(T &fst, T &snd) {
          data_pair<T> data(fst, snd);
        return data;
    }
};

class ptr_pair {
public:
    uint8_t *fst, *snd;

    template<typename T>
    struct data { typedef data_pair<T> t; };

    ptr_pair() : fst(NULL), snd(NULL) {}
    ptr_pair(uint8_t *in_fst, uint8_t *in_snd) : fst(in_fst), snd(in_snd) {}
    ptr_pair(data_pair<uint8_t *> &other) : fst(other.fst), snd(other.snd) {}

    inline void operator=(uint8_t *rhs) { fst = snd = rhs; }

    inline operator bool() const { return fst != NULL && snd != NULL; }

    inline ptr_pair operator+(size_t n) const {
        return make(fst + n, snd + n);
    }

    inline ptr_pair operator+=(size_t n) {
        fst += n; snd += n;
        return *this;
    }

    inline ptr_pair operator-(size_t n) const {
        return make(fst - n, snd - n);
    }

    inline bool operator<(const ptr_pair &other) const {
        return fst < other.fst && snd < other.snd;
    }

    static inline ptr_pair make(uint8_t *fst, uint8_t *snd) {
        ptr_pair self(fst, snd);
        return self;
    }

    static inline ptr_pair make(const data_pair<uint8_t *> &pair) {
        ptr_pair self(pair.fst, pair.snd);
        return self;
    }

    inline const type_desc *box_body_td() const {
        // Here we assume that the two ptrs are both boxes with
        // equivalent type descriptors.  This is safe because we only
        // use ptr_pair in the cmp glue, and we only use the cmp glue
        // when rust guarantees us that the boxes are of the same
        // type.  As box types are not opaque to Rust, it is in a
        // position to make this determination.
        rust_opaque_box *box_fst = *reinterpret_cast<rust_opaque_box**>(fst);
        assert(box_fst->ref_count >= 1);
        return box_fst->td;
    }

    inline const type_desc *uniq_body_td() const {
        rust_opaque_box *box_fst = *reinterpret_cast<rust_opaque_box**>(fst);
        return box_fst->td;
    }

    inline ptr_pair box_body() const {
        rust_opaque_box *box_fst = *reinterpret_cast<rust_opaque_box**>(fst);
        rust_opaque_box *box_snd = *reinterpret_cast<rust_opaque_box**>(snd);
        return make((uint8_t*)::box_body(box_fst),
                    (uint8_t*)::box_body(box_snd));
    }
};

// NB: This function does not align.
template<typename T>
inline data_pair<T>
bump_dp(ptr_pair &ptr) {
    data_pair<T> data(*reinterpret_cast<T *>(ptr.fst),
                      *reinterpret_cast<T *>(ptr.snd));
    ptr += sizeof(T);
    return data;
}

template<typename T>
inline data_pair<T>
get_dp(ptr_pair &ptr) {
    data_pair<T> data(*reinterpret_cast<T *>(ptr.fst),
                      *reinterpret_cast<T *>(ptr.snd));
    return data;
}

}   // end namespace shape


inline shape::ptr_pair
align_to(const shape::ptr_pair &pair, size_t n) {
    return shape::ptr_pair::make(align_to(pair.fst, n),
                                 align_to(pair.snd, n));
}


namespace shape {

// An abstract class (again using the curiously recurring template pattern)
// for methods that actually manipulate the data involved.

#define ALIGN_TO(alignment) \
    if (this->align) { \
        dp = align_to(dp, (alignment)); \
        if (this->end_dp && !(dp < this->end_dp)) \
            return; \
    }

#define DATA_SIMPLE(ty, call) \
    ALIGN_TO(rust_alignof<ty>()); \
    U end_dp = dp + sizeof(ty); \
    static_cast<T *>(this)->call; \
    dp = end_dp;

template<typename T,typename U>
class data : public ctxt< data<T,U> > {
public:
    U dp;

protected:
    U end_dp;

    void walk_box_contents1();
    void walk_uniq_contents1();
    void walk_rptr_contents1();
    void walk_fn_contents1();
    void walk_iface_contents1();
    void walk_variant1(tag_info &tinfo, tag_variant_t variant);

    static std::pair<uint8_t *,uint8_t *> get_vec_data_range(ptr dp);
    static std::pair<ptr_pair,ptr_pair> get_vec_data_range(ptr_pair &dp);

    static std::pair<uint8_t *,uint8_t *> get_unboxed_vec_data_range(ptr dp);
    static std::pair<ptr_pair,ptr_pair>
        get_unboxed_vec_data_range(ptr_pair &dp);
    static ptr get_unboxed_vec_end(ptr dp);
    static ptr_pair get_unboxed_vec_end(ptr_pair &dp);

    static std::pair<uint8_t *,uint8_t *> get_slice_data_range(bool is_str,
                                                               ptr dp);
    static std::pair<ptr_pair,ptr_pair> get_slice_data_range(bool is_str,
                                                             ptr_pair &dp);

    static std::pair<uint8_t *,uint8_t *>
        get_fixedvec_data_range(uint16_t n_elts, size_t elt_sz, ptr dp);
    static std::pair<ptr_pair,ptr_pair>
        get_fixedvec_data_range(uint16_t n_elts, size_t elt_sz, ptr_pair &dp);

public:
    data(rust_task *in_task,
         bool in_align,
         const uint8_t *in_sp,
         const rust_shape_tables *in_tables,
         U const &in_dp)
    : ctxt< data<T,U> >(in_task, in_align, in_sp, in_tables),
      dp(in_dp),
      end_dp() {}

    void walk_tag1(tag_info &tinfo);

    void walk_struct1(const uint8_t *end_sp) {
        // FIXME (probably won't fix before #1498): shouldn't we be aligning
        // to the first element here?
        static_cast<T *>(this)->walk_struct2(end_sp);
    }

    void walk_vec1(bool is_pod) {
        DATA_SIMPLE(void *, walk_vec2(is_pod));
    }

    void walk_unboxed_vec1(bool is_pod) {
        // align?
        U next_dp = get_unboxed_vec_end(dp);
        static_cast<T *>(this)->walk_unboxed_vec2(is_pod);
        dp = next_dp;
    }

    void walk_slice1(bool is_pod, bool is_str) {
        DATA_SIMPLE(void *, walk_slice2(is_pod, is_str));
    }

    void walk_fixedvec1(uint16_t n_elts, bool is_pod) {
        size_align sa = size_of::get(*this);
        ALIGN_TO(sa.alignment);
        U next_dp = dp + (n_elts * sa.size);
        static_cast<T *>(this)->walk_fixedvec2(n_elts, sa.size, is_pod);
        dp = next_dp;
    }

    void walk_box1() { DATA_SIMPLE(void *, walk_box2()); }

    void walk_uniq1() { DATA_SIMPLE(void *, walk_uniq2()); }

    void walk_rptr1() { DATA_SIMPLE(void *, walk_rptr2()); }

    void walk_fn1(char code) {
        ALIGN_TO(rust_alignof<void *>());
        U next_dp = dp + sizeof(void *) * 2;
        static_cast<T *>(this)->walk_fn2(code);
        dp = next_dp;
    }

    void walk_iface1() {
        ALIGN_TO(rust_alignof<void *>());
        U next_dp = dp + sizeof(void *);
        static_cast<T *>(this)->walk_iface2();
        dp = next_dp;
    }

    void walk_tydesc1(char kind) {
        ALIGN_TO(rust_alignof<void *>());
        U next_dp = dp + sizeof(void *);
        static_cast<T *>(this)->walk_tydesc2(kind);
        dp = next_dp;
    }

    void walk_res1(const rust_fn *dtor, const uint8_t *end_sp) {
        typename U::template data<uintptr_t>::t live = bump_dp<uintptr_t>(dp);
        // Delegate to the implementation.
        static_cast<T *>(this)->walk_res2(dtor, end_sp, live);
    }

    template<typename WN>
    void walk_number1() {
        //DATA_SIMPLE(W, walk_number2<W>());
        ALIGN_TO(rust_alignof<WN>());
        U end_dp = dp + sizeof(WN);
        T* t = static_cast<T *>(this);
        t->template walk_number2<WN>();
        dp = end_dp;
    }
};

template<typename T,typename U>
void
data<T,U>::walk_box_contents1() {
    const type_desc *body_td = dp.box_body_td();
    if (body_td) {
        U body_dp(dp.box_body());
        arena arena;
        T sub(*static_cast<T *>(this), body_td->shape,
              body_td->shape_tables, body_dp);
        sub.align = true;
        static_cast<T *>(this)->walk_box_contents2(sub);
    }
}

template<typename T,typename U>
void
data<T,U>::walk_uniq_contents1() {
    const type_desc *body_td = dp.uniq_body_td();
    if (body_td) {
        U body_dp(dp.box_body());
        arena arena;
        T sub(*static_cast<T *>(this), /*body_td->shape,*/ this->sp,
              body_td->shape_tables, body_dp);
        sub.align = true;
        static_cast<T *>(this)->walk_uniq_contents2(sub);
    }
}

template<typename T,typename U>
void
data<T,U>::walk_rptr_contents1() {
    typename U::template data<uint8_t *>::t box_ptr = bump_dp<uint8_t *>(dp);
    U data_ptr(box_ptr);
    T sub(*static_cast<T *>(this), data_ptr);
    static_cast<T *>(this)->walk_rptr_contents2(sub);
}

template<typename T,typename U>
void
data<T,U>::walk_variant1(tag_info &tinfo, tag_variant_t variant_id) {
    std::pair<const uint8_t *,const uint8_t *> variant_ptr_and_end =
      this->get_variant_sp(tinfo, variant_id);
    static_cast<T *>(this)->walk_variant2(tinfo, variant_id,
                                          variant_ptr_and_end);
}

template<typename T,typename U>
std::pair<uint8_t *,uint8_t *>
data<T,U>::get_vec_data_range(ptr dp) {
    rust_vec_box* ptr = bump_dp<rust_vec_box*>(dp);
    uint8_t* data = &ptr->body.data[0];
    return std::make_pair(data, data + ptr->body.fill);
}

template<typename T,typename U>
std::pair<ptr_pair,ptr_pair>
data<T,U>::get_vec_data_range(ptr_pair &dp) {
    std::pair<uint8_t *,uint8_t *> fst =
        get_vec_data_range(shape::ptr(dp.fst));
    std::pair<uint8_t *,uint8_t *> snd =
        get_vec_data_range(shape::ptr(dp.snd));
    ptr_pair start(fst.first, snd.first);
    ptr_pair end(fst.second, snd.second);
    return std::make_pair(start, end);
}

template<typename T,typename U>
std::pair<uint8_t *,uint8_t *>
data<T,U>::get_unboxed_vec_data_range(ptr dp) {
    rust_vec* ptr = (rust_vec*)dp;
    uint8_t* data = &ptr->data[0];
    return std::make_pair(data, data + ptr->fill);
}

template<typename T,typename U>
std::pair<ptr_pair,ptr_pair>
data<T,U>::get_unboxed_vec_data_range(ptr_pair &dp) {
    std::pair<uint8_t *,uint8_t *> fst =
        get_unboxed_vec_data_range(shape::ptr(dp.fst));
    std::pair<uint8_t *,uint8_t *> snd =
        get_unboxed_vec_data_range(shape::ptr(dp.snd));
    ptr_pair start(fst.first, snd.first);
    ptr_pair end(fst.second, snd.second);
    return std::make_pair(start, end);
}

template<typename T,typename U>
ptr data<T,U>::get_unboxed_vec_end(ptr dp) {
    rust_vec* ptr = (rust_vec*)dp;
    return dp + sizeof(rust_vec) + ptr->fill;
}

template<typename T,typename U>
ptr_pair data<T,U>::get_unboxed_vec_end(ptr_pair &dp) {
    return ptr_pair(get_unboxed_vec_end(ptr(dp.fst)),
                    get_unboxed_vec_end(ptr(dp.snd)));
}

template<typename T,typename U>
std::pair<uint8_t *,uint8_t *>
data<T,U>::get_slice_data_range(bool is_str, ptr dp) {
    uint8_t* ptr = bump_dp<uint8_t*>(dp);
    size_t len = bump_dp<size_t>(dp);
    if (is_str) len--;
    return std::make_pair(ptr, ptr + len);
}

template<typename T,typename U>
std::pair<ptr_pair,ptr_pair>
data<T,U>::get_slice_data_range(bool is_str, ptr_pair &dp) {
    std::pair<uint8_t *,uint8_t *> fst =
        get_slice_data_range(is_str, shape::ptr(dp.fst));
    std::pair<uint8_t *,uint8_t *> snd =
        get_slice_data_range(is_str, shape::ptr(dp.snd));
    ptr_pair start(fst.first, snd.first);
    ptr_pair end(fst.second, snd.second);
    return std::make_pair(start, end);
}

template<typename T,typename U>
std::pair<uint8_t *,uint8_t *>
data<T,U>::get_fixedvec_data_range(uint16_t n_elts, size_t elt_sz, ptr dp) {
    uint8_t* ptr = (uint8_t*)(dp);
    return std::make_pair(ptr, ptr + (((size_t)n_elts) * elt_sz));
}

template<typename T,typename U>
std::pair<ptr_pair,ptr_pair>
data<T,U>::get_fixedvec_data_range(uint16_t n_elts, size_t elt_sz,
                                   ptr_pair &dp) {
    std::pair<uint8_t *,uint8_t *> fst =
        get_fixedvec_data_range(n_elts, elt_sz, shape::ptr(dp.fst));
    std::pair<uint8_t *,uint8_t *> snd =
        get_fixedvec_data_range(n_elts, elt_sz, shape::ptr(dp.snd));
    ptr_pair start(fst.first, snd.first);
    ptr_pair end(fst.second, snd.second);
    return std::make_pair(start, end);
}


template<typename T,typename U>
void
data<T,U>::walk_tag1(tag_info &tinfo) {
    size_of::compute_tag_size(*this, tinfo);

    if (tinfo.variant_count > 1)
        ALIGN_TO(rust_alignof<tag_align_t>());

    U end_dp = dp + tinfo.tag_sa.size;

    typename U::template data<tag_variant_t>::t tag_variant;
    if (tinfo.variant_count > 1)
        tag_variant = bump_dp<tag_variant_t>(dp);
    else
        tag_variant = 0;

    static_cast<T *>(this)->walk_tag2(tinfo, tag_variant);

    dp = end_dp;
}

template<typename T,typename U>
void
  data<T,U>::walk_fn_contents1() {
    fn_env_pair pair = bump_dp<fn_env_pair>(dp);
    if (!pair.env)
        return;

    arena arena;
    const type_desc *closure_td = pair.env->td;
    ptr closure_dp((uintptr_t)box_body(pair.env));
    T sub(*static_cast<T *>(this), closure_td->shape,
          closure_td->shape_tables, closure_dp);
    sub.align = true;

    sub.walk();
}

template<typename T,typename U>
void
data<T,U>::walk_iface_contents1() {
    walk_box_contents1();
}

// Polymorphic logging, for convenience

class log : public data<log,ptr> {
    friend class data<log,ptr>;

private:
    std::ostream &out;
    const char *prefix;
    bool in_string;

    log(log &other,
        const uint8_t *in_sp,
        const rust_shape_tables *in_tables = NULL)
    : data<log,ptr>(other.task,
                    other.align,
                    in_sp,
                    in_tables ? in_tables : other.tables,
                    other.dp),
      out(other.out),
      prefix("") {}

    log(log &other,
        const uint8_t *in_sp,
        const rust_shape_tables *in_tables,
        ptr in_dp)
    : data<log,ptr>(other.task,
                    other.align,
                    in_sp,
                    in_tables,
                    in_dp),
      out(other.out),
      prefix("") {}

    log(log &other, ptr in_dp)
    : data<log,ptr>(other.task,
                    other.align,
                    other.sp,
                    other.tables,
                    in_dp),
      out(other.out),
      prefix("") {}

    void walk_vec2(bool is_pod) {
        if (!get_dp<void *>(dp))
            out << prefix << "(null)";
        else
            walk_vec2(is_pod, get_vec_data_range(dp));
    }

    void walk_unboxed_vec2(bool is_pod) {
        walk_vec2(is_pod, get_unboxed_vec_data_range(dp));
    }

    void walk_slice2(bool is_pod, bool is_str) {
        walk_vec2(is_pod, get_slice_data_range(is_str, dp));
        out << "/&";
    }

    void walk_fixedvec2(uint16_t n_elts, size_t elt_sz, bool is_pod) {
        walk_vec2(is_pod, get_fixedvec_data_range(n_elts, elt_sz, dp));
        out << "/" << n_elts;
    }

    void walk_tag2(tag_info &tinfo, tag_variant_t tag_variant) {
        // out << prefix << "tag" << tag_variant;
        out << prefix << get_variant_name(tinfo, tag_variant);
        data<log,ptr>::walk_variant1(tinfo, tag_variant);
    }

    void walk_box2() {
        out << prefix << "@";
        prefix = "";
        data<log,ptr>::walk_box_contents1();
    }

    void walk_uniq2() {
        out << prefix << "~";
        prefix = "";
        data<log,ptr>::walk_uniq_contents1();
    }

    void walk_rptr2() {
        out << prefix << "&";
        prefix = "";
        data<log,ptr>::walk_rptr_contents1();
    }

    void walk_fn2(char kind) {
        out << prefix << "fn";
        prefix = "";
        data<log,ptr>::walk_fn_contents1();
    }

    void walk_iface2() {
        out << prefix << "iface(";
        prefix = "";
        data<log,ptr>::walk_iface_contents1();
        out << prefix << ")";
    }

    void walk_tydesc2(char kind) {
        out << prefix << "tydesc";
    }

    void walk_subcontext2(log &sub) { sub.walk(); }

    void walk_box_contents2(log &sub) {
        out << prefix;
        rust_opaque_box *box_ptr = *(rust_opaque_box **) dp;
        if (!box_ptr) {
            out << "(null)";
        } else {
            sub.align = true;
            sub.walk();
        }
    }

    void walk_uniq_contents2(log &sub) {
        out << prefix;
        sub.align = true;
        sub.walk();
    }

    void walk_rptr_contents2(log &sub) {
        out << prefix;
        sub.align = true;
        sub.walk();
    }

    void walk_struct2(const uint8_t *end_sp);
    void walk_vec2(bool is_pod, const std::pair<ptr,ptr> &data);
    void walk_slice2(bool is_pod, const std::pair<ptr,ptr> &data);
    void walk_variant2(tag_info &tinfo,
                       tag_variant_t variant_id,
                       const std::pair<const uint8_t *,const uint8_t *>
                       variant_ptr_and_end);
    void walk_string2(const std::pair<ptr,ptr> &data);
    void walk_res2(const rust_fn *dtor, const uint8_t *end_sp, bool live);

    template<typename T>
    inline void walk_number2() {
        out << prefix;
        fmt_number(out, get_dp<T>(dp));
    }

public:
    log(rust_task *in_task,
        bool in_align,
        const uint8_t *in_sp,
        const rust_shape_tables *in_tables,
        uint8_t *in_data,
        std::ostream &in_out)
        : data<log,ptr>(in_task, in_align, in_sp, in_tables,
                        ptr(in_data)),
      out(in_out),
      prefix("") {}
};

}   // end namespace shape

#endif

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
