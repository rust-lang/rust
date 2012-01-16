// Functions that interpret the shape of a type to perform various low-level
// actions, such as copying, freeing, comparing, and so on.

#ifndef RUST_SHAPE_H
#define RUST_SHAPE_H

// Tell ISAAC to let go of max() and min() defines.
#undef max
#undef min

#include <iostream>
#include "rust_internal.h"
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
typedef unsigned long ref_cnt_t;

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
const uint8_t SHAPE_VEC = 11u;
const uint8_t SHAPE_TAG = 12u;
const uint8_t SHAPE_BOX = 13u;
const uint8_t SHAPE_STRUCT = 17u;
const uint8_t SHAPE_BOX_FN = 18u;
const uint8_t SHAPE_OBJ = 19u;
const uint8_t SHAPE_RES = 20u;
const uint8_t SHAPE_VAR = 21u;
const uint8_t SHAPE_UNIQ = 22u;
const uint8_t SHAPE_IFACE = 24u;
const uint8_t SHAPE_UNIQ_FN = 25u;
const uint8_t SHAPE_STACK_FN = 26u;
const uint8_t SHAPE_BARE_FN = 27u;
const uint8_t SHAPE_TYDESC = 28u;
const uint8_t SHAPE_SEND_TYDESC = 29u;

#ifdef _LP64
const uint8_t SHAPE_PTR = SHAPE_U64;
#else
const uint8_t SHAPE_PTR = SHAPE_U32;
#endif


// Forward declarations

struct rust_obj;
struct size_align;
class ptr;
class type_param;


// Arenas; these functions must execute very quickly, so we use an arena
// instead of malloc or new.

class arena {
    uint8_t *ptr;
    uint8_t data[ARENA_SIZE];

public:
    arena() : ptr(data) {}

    template<typename T>
    inline T *alloc(size_t count = 1) {
        // FIXME: align
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
alignof() {
#ifdef _MSC_VER
    return __alignof(T);
#else
    return __alignof__(T);
#endif
}

template<>
inline size_t
alignof<double>() {
    return 4;
}


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
    uint16_t n_params;                      // Number of type parameters.
    const type_param *params;               // Array of type parameters.
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
    const type_param *params;           // shapes of type parameters
    const rust_shape_tables *tables;
    rust_task *task;
    bool align;

    ctxt(rust_task *in_task,
         bool in_align,
         const uint8_t *in_sp,
         const type_param *in_params,
         const rust_shape_tables *in_tables)
    : sp(in_sp),
      params(in_params),
      tables(in_tables),
      task(in_task),
      align(in_align) {}

    template<typename U>
    ctxt(const ctxt<U> &other,
         const uint8_t *in_sp = NULL,
         const type_param *in_params = NULL,
         const rust_shape_tables *in_tables = NULL)
    : sp(in_sp ? in_sp : other.sp),
      params(in_params ? in_params : other.params),
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
    void walk_tag0();
    void walk_box0();
    void walk_uniq0();
    void walk_struct0();
    void walk_res0();
    void walk_var0();
};


// Core Rust types

struct rust_fn {
    void (*code)(uint8_t *rv, rust_task *task, void *env, ...);
    void *env;
};

struct rust_closure {
    type_desc *tydesc;
    uint32_t target_0;
    uint32_t target_1;
    uint32_t bindings[0];

    uint8_t *get_bindings() const { return (uint8_t *)bindings; }
};

struct rust_obj_box {
    type_desc *tydesc;

    uint8_t *get_bindings() const { return (uint8_t *)this; }
};

struct rust_vtable {
    CDECL void (*dtor)(void *rv, rust_task *task, rust_obj obj);
};

struct rust_obj {
    rust_vtable *vtable;
    void *box;
};


// Type parameters

class type_param {
private:
    static type_param *make(const type_desc **tydescs, unsigned n_tydescs,
                            arena &arena);

public:
    const uint8_t *shape;
    const rust_shape_tables *tables;
    const type_param *params;   // subparameters

    // Creates type parameters from an object shape description.
    static type_param *from_obj_shape(const uint8_t *sp, ptr dp,
                                      arena &arena);

    template<typename T>
    inline void set(ctxt<T> *cx) {
        shape = cx->sp;
        tables = cx->tables;
        params = cx->params;
    }

    // Creates type parameters from a type descriptor.
    static inline type_param *from_tydesc(const type_desc *tydesc,
                                          arena &arena) {
        // In order to find the type parameters of objects and functions, we
        // have to actually have the data pointer, since we don't statically
        // know from the type of an object or function which type parameters
        // it closes over.
        assert(!tydesc->n_obj_params && "Type-parametric objects "
               "must go through from_tydesc_and_data() instead!");

        return make(tydesc->first_param, tydesc->n_params, arena);
    }

    static type_param *from_tydesc_and_data(const type_desc *tydesc,
                                            uint8_t *dp, arena &arena) {
        if (tydesc->n_obj_params) {
            uintptr_t n_obj_params = tydesc->n_obj_params;
            const type_desc **first_param;
            // Object closure.
            DPRINT("n_obj_params OBJ %lu, tydesc %p, starting at %p\n",
                   (unsigned long)n_obj_params, tydesc,
                   dp + sizeof(uintptr_t) * 2);
            first_param = (const type_desc **)(dp + sizeof(uintptr_t) * 2);
            return make(first_param, n_obj_params, arena);
        }

        return make(tydesc->first_param, tydesc->n_params, arena);
    }
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
    case SHAPE_VEC:      walk_vec0();             break;
    case SHAPE_TAG:      walk_tag0();             break;
    case SHAPE_BOX:      walk_box0();             break;
    case SHAPE_STRUCT:   walk_struct0();          break;
    case SHAPE_OBJ:      WALK_SIMPLE(walk_obj1);      break;
    case SHAPE_RES:      walk_res0();             break;
    case SHAPE_VAR:      walk_var0();             break;
    case SHAPE_UNIQ:     walk_uniq0();            break;
    case SHAPE_IFACE:    WALK_SIMPLE(walk_iface1);    break;
    case SHAPE_BOX_FN:
    case SHAPE_UNIQ_FN:
    case SHAPE_STACK_FN:
    case SHAPE_BARE_FN:  static_cast<T*>(this)->walk_fn1(s); break;
    case SHAPE_SEND_TYDESC:
    case SHAPE_TYDESC:   static_cast<T*>(this)->walk_tydesc1(s); break;
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

    static_cast<T *>(this)->walk_vec1(is_pod, sp_size);

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

    // Determine the number of parameters.
    tinfo.n_params = get_u16_bump(sp);

    // Read in the tag type parameters.
    type_param params[tinfo.n_params];
    for (uint16_t i = 0; i < tinfo.n_params; i++) {
        uint16_t len = get_u16_bump(sp);
        params[i].set(this);
        sp += len;
    }

    tinfo.params = params;

    // Call to the implementation.
    static_cast<T *>(this)->walk_tag1(tinfo);
}

template<typename T>
void
ctxt<T>::walk_box0() {
    uint16_t sp_size = get_u16_bump(sp);
    const uint8_t *end_sp = sp + sp_size;

    static_cast<T *>(this)->walk_box1();

    sp = end_sp;
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

    uint16_t n_ty_params = get_u16_bump(sp);

    // Read in the tag type parameters.
    type_param params[n_ty_params];
    for (uint16_t i = 0; i < n_ty_params; i++) {
        uint16_t ty_param_len = get_u16_bump(sp);
        const uint8_t *next_sp = sp + ty_param_len;
        params[i].set(this);
        sp = next_sp;
    }

    uint16_t sp_size = get_u16_bump(sp);
    const uint8_t *end_sp = sp + sp_size;

    static_cast<T *>(this)->walk_res1(dtor, n_ty_params, params, end_sp);

    sp = end_sp;
}

template<typename T>
void
ctxt<T>::walk_var0() {
    uint8_t param = *sp++;
    static_cast<T *>(this)->walk_var1(param);
}

// A shape printer, useful for debugging

class print : public ctxt<print> {
public:
    template<typename T>
    print(const ctxt<T> &other,
          const uint8_t *in_sp = NULL,
          const type_param *in_params = NULL,
          const rust_shape_tables *in_tables = NULL)
    : ctxt<print>(other, in_sp, in_params, in_tables) {}

    print(rust_task *in_task,
          bool in_align,
          const uint8_t *in_sp,
          const type_param *in_params,
          const rust_shape_tables *in_tables)
    : ctxt<print>(in_task, in_align, in_sp, in_params, in_tables) {}

    void walk_tag1(tag_info &tinfo);
    void walk_struct1(const uint8_t *end_sp);
    void walk_res1(const rust_fn *dtor, unsigned n_params,
                   const type_param *params, const uint8_t *end_sp);
    void walk_var1(uint8_t param);

    void walk_vec1(bool is_pod, uint16_t sp_size) {
        DPRINT("vec<"); walk(); DPRINT(">");
    }
    void walk_uniq1() {
        DPRINT("~<"); walk(); DPRINT(">");
    }
    void walk_box1() {
        DPRINT("@<"); walk(); DPRINT(">");
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
    void walk_obj1() { DPRINT("obj"); }
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
            const type_param *in_params = NULL,
            const rust_shape_tables *in_tables = NULL)
    : ctxt<size_of>(other, in_sp, in_params, in_tables) {}

    template<typename T>
    size_of(const ctxt<T> &other,
            const uint8_t *in_sp = NULL,
            const type_param *in_params = NULL,
            const rust_shape_tables *in_tables = NULL)
    : ctxt<size_of>(other, in_sp, in_params, in_tables) {}

    void walk_tag1(tag_info &tinfo);
    void walk_struct1(const uint8_t *end_sp);

    void walk_uniq1()       { sa.set(sizeof(void *),   sizeof(void *)); }
    void walk_box1()        { sa.set(sizeof(void *),   sizeof(void *)); }
    void walk_fn1(char)     { sa.set(sizeof(void *)*2, sizeof(void *)); }
    void walk_obj1()        { sa.set(sizeof(void *)*2, sizeof(void *)); }
    void walk_iface1()      { sa.set(sizeof(void *),   sizeof(void *)); }
    void walk_tydesc1(char) { sa.set(sizeof(void *),   sizeof(void *)); }
    void walk_closure1();

    void walk_vec1(bool is_pod, uint16_t sp_size) {
        sa.set(sizeof(void *), sizeof(void *));
    }

    void walk_var1(uint8_t param_index) {
        const type_param *param = &params[param_index];
        size_of sub(*this, param->shape, param->params, param->tables);
        sub.walk();
        sa = sub.sa;
    }

    void walk_res1(const rust_fn *dtor, unsigned n_params,
                   const type_param *params, const uint8_t *end_sp) {
        abort();    // TODO
    }

    template<typename T>
    void walk_number1()  { sa.set(sizeof(T), alignof<T>()); }

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
    ptr(uint8_t *in_p) : p(in_p) {}
    ptr(uintptr_t in_p) : p((uint8_t *)in_p) {}

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
    ALIGN_TO(alignof<ty>()); \
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
    void walk_fn_contents1(ptr &dp, bool null_td);
    void walk_obj_contents1(ptr &dp);
    void walk_iface_contents1(ptr &dp);
    void walk_variant1(tag_info &tinfo, tag_variant_t variant);

    static std::pair<uint8_t *,uint8_t *> get_vec_data_range(ptr dp);
    static std::pair<ptr_pair,ptr_pair> get_vec_data_range(ptr_pair &dp);

public:
    data(rust_task *in_task,
         bool in_align,
         const uint8_t *in_sp,
         const type_param *in_params,
         const rust_shape_tables *in_tables,
         U const &in_dp)
    : ctxt< data<T,U> >(in_task, in_align, in_sp, in_params, in_tables),
      dp(in_dp),
      end_dp() {}

    void walk_tag1(tag_info &tinfo);

    void walk_struct1(const uint8_t *end_sp) {
        static_cast<T *>(this)->walk_struct2(end_sp);
    }

    void walk_vec1(bool is_pod, uint16_t sp_size) {
        DATA_SIMPLE(void *, walk_vec2(is_pod, sp_size));
    }

    void walk_box1() { DATA_SIMPLE(void *, walk_box2()); }

    void walk_uniq1() { DATA_SIMPLE(void *, walk_uniq2()); }

    void walk_fn1(char code) {
        ALIGN_TO(alignof<void *>());
        U next_dp = dp + sizeof(void *) * 2;
        static_cast<T *>(this)->walk_fn2(code);
        dp = next_dp;
    }

    void walk_obj1() {
        ALIGN_TO(alignof<void *>());
        U next_dp = dp + sizeof(void *) * 2;
        static_cast<T *>(this)->walk_obj2();
        dp = next_dp;
    }

    void walk_iface1() {
        ALIGN_TO(alignof<void *>());
        U next_dp = dp + sizeof(void *);
        static_cast<T *>(this)->walk_iface2();
        dp = next_dp;
    }

    void walk_tydesc1(char kind) {
        ALIGN_TO(alignof<void *>());
        U next_dp = dp + sizeof(void *);
        static_cast<T *>(this)->walk_tydesc2(kind);
        dp = next_dp;
    }

    void walk_res1(const rust_fn *dtor, unsigned n_params,
                   const type_param *params, const uint8_t *end_sp) {
        typename U::template data<uintptr_t>::t live = bump_dp<uintptr_t>(dp);
        // Delegate to the implementation.
        static_cast<T *>(this)->walk_res2(dtor, n_params, params, end_sp,
                                         live);
    }

    void walk_var1(uint8_t param_index) {
        const type_param *param = &this->params[param_index];
        T sub(*static_cast<T *>(this), param->shape, param->params,
              param->tables);
        static_cast<T *>(this)->walk_subcontext2(sub);
        dp = sub.dp;
    }

    template<typename WN>
    void walk_number1() { 
        //DATA_SIMPLE(W, walk_number2<W>());
        ALIGN_TO(alignof<WN>());
        U end_dp = dp + sizeof(WN);
        T* t = static_cast<T *>(this);
        t->template walk_number2<WN>();
        dp = end_dp;
    }
};

template<typename T,typename U>
void
data<T,U>::walk_box_contents1() {
    typename U::template data<uint8_t *>::t box_ptr = bump_dp<uint8_t *>(dp);
    U ref_count_dp(box_ptr);
    T sub(*static_cast<T *>(this), ref_count_dp + sizeof(ref_cnt_t));
    static_cast<T *>(this)->walk_box_contents2(sub, ref_count_dp);
}

template<typename T,typename U>
void
data<T,U>::walk_uniq_contents1() {
    typename U::template data<uint8_t *>::t box_ptr = bump_dp<uint8_t *>(dp);
    U data_ptr(box_ptr);
    T sub(*static_cast<T *>(this), data_ptr);
    static_cast<T *>(this)->walk_uniq_contents2(sub);
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
    rust_vec* ptr = bump_dp<rust_vec*>(dp);
    uint8_t* data = &ptr->data[0];
    return std::make_pair(data, data + ptr->fill);
}

template<typename T,typename U>
std::pair<ptr_pair,ptr_pair>
data<T,U>::get_vec_data_range(ptr_pair &dp) {
    std::pair<uint8_t *,uint8_t *> fst = get_vec_data_range(dp.fst);
    std::pair<uint8_t *,uint8_t *> snd = get_vec_data_range(dp.snd);
    ptr_pair start(fst.first, snd.first);
    ptr_pair end(fst.second, snd.second);
    return std::make_pair(start, end);
}

template<typename T,typename U>
void
data<T,U>::walk_tag1(tag_info &tinfo) {
    size_of::compute_tag_size(*this, tinfo);

    if (tinfo.variant_count > 1)
        ALIGN_TO(alignof<tag_align_t>());

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
data<T,U>::walk_fn_contents1(ptr &dp, bool null_td) {
    fn_env_pair pair = bump_dp<fn_env_pair>(dp);
    if (!pair.env)
        return;

    arena arena;
    const type_desc *closure_td = pair.env->td;
    type_param *params =
      type_param::from_tydesc(closure_td, arena);
    ptr closure_dp((uintptr_t)pair.env);
    T sub(*static_cast<T *>(this), closure_td->shape, params,
          closure_td->shape_tables, closure_dp);
    sub.align = true;

    if (null_td) {
        // if null_td flag is true, null out the type descr from
        // the data structure while we walk.  This is used in cycle
        // collector when we are sweeping up data.  The idea is that
        // we are using the information in the embedded type desc to
        // walk the contents, so we do not want to free it during that
        // walk.  This is not *strictly* necessary today because
        // type_param::from_tydesc() actually pulls out the "shape"
        // string and other information and copies it into a new
        // location that is unaffected by the free.  But it seems
        // safer, particularly as this pulling out of information will
        // not cope with nested, derived type descriptors.
        pair.env->td = NULL;
    }

    sub.walk();

    if (null_td) {
        pair.env->td = closure_td;
    }
}

template<typename T,typename U>
void
data<T,U>::walk_obj_contents1(ptr &dp) {
    dp += sizeof(void *);   // Skip over the vtable.

    uint8_t *box_ptr = bump_dp<uint8_t *>(dp);
    type_desc *subtydesc =
        *reinterpret_cast<type_desc **>(box_ptr + sizeof(void *));
    ptr obj_closure_dp(box_ptr + sizeof(void *));
    if (!box_ptr)   // Null check.
        return;

    arena arena;
    type_param *params = type_param::from_obj_shape(subtydesc->shape,
                                                    obj_closure_dp, arena);
    T sub(*static_cast<T *>(this), subtydesc->shape, params,
          subtydesc->shape_tables, obj_closure_dp);
    sub.align = true;
    sub.walk();
}

template<typename T,typename U>
void
data<T,U>::walk_iface_contents1(ptr &dp) {
    uint8_t *box_ptr = bump_dp<uint8_t *>(dp);
    if (!box_ptr) return;
    U ref_count_dp(box_ptr);
    uint8_t *body_ptr = box_ptr + sizeof(void*);
    type_desc *valtydesc =
        *reinterpret_cast<type_desc **>(body_ptr);
    ptr value_dp(body_ptr + sizeof(void*) * 2);
    // FIXME The 5 is a hard-coded way to skip over a struct shape
    // header and the first two (number-typed) fields. This is too
    // fragile, but I didn't see a good way to properly encode it.
    T sub(*static_cast<T *>(this), valtydesc->shape + 5, NULL, NULL,
          value_dp);
    sub.align = true;
    static_cast<T *>(this)->walk_box_contents2(sub, ref_count_dp);
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
        const type_param *in_params,
        const rust_shape_tables *in_tables = NULL)
    : data<log,ptr>(other.task,
                    other.align,
                    in_sp,
                    in_params,
                    in_tables ? in_tables : other.tables,
                    other.dp),
      out(other.out),
      prefix("") {}

    log(log &other,
        const uint8_t *in_sp,
        const type_param *in_params,
        const rust_shape_tables *in_tables,
        ptr in_dp)
    : data<log,ptr>(other.task,
                    other.align,
                    in_sp,
                    in_params,
                    in_tables,
                    in_dp),
      out(other.out),
      prefix("") {}

    log(log &other, ptr in_dp)
    : data<log,ptr>(other.task,
                    other.align,
                    other.sp,
                    other.params,
                    other.tables,
                    in_dp),
      out(other.out),
      prefix("") {}

    void walk_vec2(bool is_pod, uint16_t sp_size) {
        if (!get_dp<void *>(dp))
            out << prefix << "(null)";
        else
            walk_vec2(is_pod, get_vec_data_range(dp));
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

    void walk_fn2(char kind) {
        out << prefix << "fn";
        prefix = "";
        data<log,ptr>::walk_fn_contents1(dp, false);
    }

    void walk_obj2() {
        out << prefix << "obj";
        prefix = "";
        data<log,ptr>::walk_obj_contents1(dp);
    }

    void walk_iface2() {
        out << prefix << "iface(";
        prefix = "";
        data<log,ptr>::walk_iface_contents1(dp);
        out << prefix << ")";
    }

    void walk_tydesc2(char kind) {
        out << prefix << "tydesc";
    }

    void walk_subcontext2(log &sub) { sub.walk(); }

    void walk_box_contents2(log &sub, ptr &ref_count_dp) {
        out << prefix;
        if (!ref_count_dp) {
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

    void walk_struct2(const uint8_t *end_sp);
    void walk_vec2(bool is_pod, const std::pair<ptr,ptr> &data);
    void walk_variant2(tag_info &tinfo,
                       tag_variant_t variant_id,
                       const std::pair<const uint8_t *,const uint8_t *>
                       variant_ptr_and_end);
    void walk_string2(const std::pair<ptr,ptr> &data);
    void walk_res2(const rust_fn *dtor, unsigned n_params,
                   const type_param *params, const uint8_t *end_sp, bool live);

    template<typename T>
    inline void walk_number2() {
        out << prefix;
        fmt_number(out, get_dp<T>(dp));
    }

public:
    log(rust_task *in_task,
        bool in_align,
        const uint8_t *in_sp,
        const type_param *in_params,
        const rust_shape_tables *in_tables,
        uint8_t *in_data,
        std::ostream &in_out)
    : data<log,ptr>(in_task, in_align, in_sp, in_params, in_tables, in_data),
      out(in_out),
      prefix("") {}
};

}   // end namespace shape

#endif

