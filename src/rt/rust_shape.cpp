// Functions that interpret the shape of a type to perform various low-level
// actions, such as copying, freeing, comparing, and so on.

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include "rust_internal.h"
#include "rust_shape.h"

namespace shape {

using namespace shape;

// Forward declarations

struct rust_obj;
struct size_align;
struct type_param;


// Constants

const uint8_t CMP_EQ = 0u;
const uint8_t CMP_LT = 1u;
const uint8_t CMP_LE = 2u;

}   // end namespace shape


namespace shape {

using namespace shape;

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

// A shape printer, useful for debugging

void
print::walk_tag(bool align, tag_info &tinfo) {
    DPRINT("tag%u", tinfo.tag_id);
    if (!tinfo.n_params)
        return;

    DPRINT("<");

    bool first = true;
    for (uint16_t i = 0; i < tinfo.n_params; i++) {
        if (!first)
            DPRINT(",");
        first = false;

        ctxt<print> sub(*this, tinfo.params[i].shape);
        sub.walk(align);
    }

    DPRINT(">");
}

void
print::walk_struct(bool align, const uint8_t *end_sp) {
    DPRINT("(");

    bool first = true;
    while (sp != end_sp) {
        if (!first)
            DPRINT(",");
        first = false;

        walk(align);
    }

    DPRINT(")");
}

void
print::walk_res(bool align, const rust_fn *dtor, uint16_t n_ty_params,
                const uint8_t *ty_params_sp) {
    DPRINT("res@%p", dtor);
    if (!n_ty_params)
        return;

    DPRINT("<");

    bool first = true;
    for (uint16_t i = 0; i < n_ty_params; i++) {
        if (!first)
            DPRINT(",");
        first = false;
        get_u16_bump(sp);   // Skip over the size.
        walk(align);
    }

    DPRINT(">");
}

void
print::walk_var(bool align, uint8_t param_index) {
    DPRINT("%c=", 'T' + param_index);

    const type_param *param = &params[param_index];
    print sub(*this, param->shape, param->params, param->tables);
    sub.walk(align);
}

template<>
void print::walk_number<uint8_t>(bool align)    { DPRINT("u8"); }
template<>
void print::walk_number<uint16_t>(bool align)   { DPRINT("u16"); }
template<>
void print::walk_number<uint32_t>(bool align)   { DPRINT("u32"); }
template<>
void print::walk_number<uint64_t>(bool align)   { DPRINT("u64"); }
template<>
void print::walk_number<int8_t>(bool align)     { DPRINT("i8"); }
template<>
void print::walk_number<int16_t>(bool align)    { DPRINT("i16"); }
template<>
void print::walk_number<int32_t>(bool align)    { DPRINT("i32"); }
template<>
void print::walk_number<int64_t>(bool align)    { DPRINT("i64"); }
template<>
void print::walk_number<float>(bool align)      { DPRINT("f32"); }
template<>
void print::walk_number<double>(bool align)     { DPRINT("f64"); }


void
size_of::compute_tag_size(tag_info &tinfo) {
    // If the precalculated size and alignment are good, use them.
    if (tinfo.tag_sa.is_set())
        return;

    uint16_t n_largest_variants = get_u16_bump(tinfo.largest_variants_ptr);
    tinfo.tag_sa.set(0, 0);
    for (uint16_t i = 0; i < n_largest_variants; i++) {
        uint16_t variant_id = get_u16_bump(tinfo.largest_variants_ptr);
        std::pair<const uint8_t *,const uint8_t *> variant_ptr_and_end =
            get_variant_sp(tinfo, variant_id);
        const uint8_t *variant_ptr = variant_ptr_and_end.first;
        const uint8_t *variant_end = variant_ptr_and_end.second;

        size_of sub(*this, variant_ptr, tinfo.params, NULL);

        // Compute the size of this variant.
        size_align variant_sa;
        bool first = true;
        while (sub.sp != variant_end) {
            if (!first)
                variant_sa.size = align_to(variant_sa.size, sub.sa.alignment);
            sub.walk(!first);
            first = false;

            variant_sa.add(sub.sa.size, sub.sa.alignment);
        }

        if (tinfo.tag_sa.size < variant_sa.size)
            tinfo.tag_sa = variant_sa;
    }

    if (tinfo.variant_count == 1) {
        if (!tinfo.tag_sa.size)
            tinfo.tag_sa.set(1, 1);
    } else {
        // Add in space for the tag.
        tinfo.tag_sa.add(sizeof(uint32_t), ALIGNOF(uint32_t));
    }
}

void
size_of::walk_tag(bool align, tag_info &tinfo) {
    compute_tag_size(*this, tinfo);
    sa = tinfo.tag_sa;
}

void
size_of::walk_struct(bool align, const uint8_t *end_sp) {
    size_align struct_sa(0, 1);

    bool first = true;
    while (sp != end_sp) {
        if (!first)
            struct_sa.size = align_to(struct_sa.size, sa.alignment);
        walk(!first);
        first = false;

        struct_sa.add(sa);
    }

    sa = struct_sa;
}

void
size_of::walk_ivec(bool align, bool is_pod, size_align &elem_sa) {
    if (!elem_sa.is_set())
        walk(align);    // Determine the size the slow way.
    else
        sa = elem_sa;   // Use the size hint.

    sa.set(sizeof(rust_ivec) - sizeof(uintptr_t) + sa.size * 4,
           max(sa.alignment, sizeof(uintptr_t)));
}


// Copy constructors

#if 0

class copy : public data<copy,uint8_t *> {
    // TODO
};

#endif


// Structural comparison glue.

class cmp : public data<cmp,ptr_pair> {
    friend class data<cmp,ptr_pair>;

private:
    void walk_vec(bool align, bool is_pod,
                  const std::pair<ptr_pair,ptr_pair> &data_range);

    inline void walk_subcontext(bool align, cmp &sub) {
        sub.walk(align);
        result = sub.result;
    }

    inline void walk_box_contents(bool align, cmp &sub,
                                  ptr_pair &ref_count_dp) {
        sub.walk(true);
        result = sub.result;
    }

    inline void cmp_two_pointers(bool align) {
        if (align) dp = align_to(dp, ALIGNOF(uint8_t *) * 2);
        data_pair<uint8_t *> fst = bump_dp<uint8_t *>(dp);
        data_pair<uint8_t *> snd = bump_dp<uint8_t *>(dp);
        cmp_number(fst);
        if (!result)
            cmp_number(snd);
    }

    inline void cmp_pointer(bool align) {
        if (align) dp = align_to(dp, ALIGNOF(uint8_t *));
        cmp_number(bump_dp<uint8_t *>(dp));
    }

    template<typename T>
    void cmp_number(const data_pair<T> &nums) {
        result = (nums.fst < nums.snd) ? -1 : (nums.fst == nums.snd) ? 0 : 1;
    }

public:
    int result;

    cmp(rust_task *in_task,
        const uint8_t *in_sp,
        const type_param *in_params,
        const rust_shape_tables *in_tables,
        uint8_t *in_data_0,
        uint8_t *in_data_1)
    : data<cmp,ptr_pair>(in_task, in_sp, in_params, in_tables,
                         ptr_pair::make(in_data_0, in_data_1)),
      result(0) {}

    cmp(const cmp &other,
        const uint8_t *in_sp = NULL,
        const type_param *in_params = NULL,
        const rust_shape_tables *in_tables = NULL)
    : data<cmp,ptr_pair>(other.task,
                         in_sp ? in_sp : other.sp,
                         in_params ? in_params : other.params,
                         in_tables ? in_tables : other.tables,
                         other.dp),
      result(0) {}

    cmp(const cmp &other, const ptr_pair &in_dp)
    : data<cmp,ptr_pair>(other.task, other.sp, other.params, other.tables,
                         in_dp),
      result(0) {}

    void walk_evec(bool align, bool is_pod, uint16_t sp_size) {
        walk_vec(align, is_pod, get_evec_data_range(dp));
    }

    void walk_ivec(bool align, bool is_pod, size_align &elem_sa) {
        walk_vec(align, is_pod, get_ivec_data_range(dp));
    }

    void walk_box(bool align) {
        data<cmp,ptr_pair>::walk_box_contents(align);
    }

    void walk_fn(bool align) { return cmp_two_pointers(align); }
    void walk_obj(bool align) { return cmp_two_pointers(align); }
    void walk_port(bool align) { return cmp_pointer(align); }
    void walk_chan(bool align) { return cmp_pointer(align); }
    void walk_task(bool align) { return cmp_pointer(align); }

    void walk_tag(bool align, tag_info &tinfo,
                  const data_pair<uint32_t> &tag_variants);
    void walk_struct(bool align, const uint8_t *end_sp);
    void walk_res(bool align, const rust_fn *dtor, uint16_t n_ty_params,
                  const uint8_t *ty_params_sp);
    void walk_variant(bool align, tag_info &tinfo, uint32_t variant_id,
                      const std::pair<const uint8_t *,const uint8_t *>
                      variant_ptr_and_end);

    template<typename T>
    void walk_number() { cmp_number(get_dp<T>(dp)); }
};

template<>
void cmp::cmp_number<int32_t>(const data_pair<int32_t> &nums) {
    result = (nums.fst < nums.snd) ? -1 : (nums.fst == nums.snd) ? 0 : 1;
}

void
cmp::walk_vec(bool align, bool is_pod,
              const std::pair<ptr_pair,ptr_pair> &data_range) {
    cmp sub(*this, data_range.first);
    ptr_pair data_end = data_range.second;
    while (!result && sub.dp < data_end) {
        sub.walk_reset(align);
        result = sub.result;
        align = true;
    }

    if (!result) {
        // If we hit the end, the result comes down to length comparison.
        int len_fst = data_range.second.fst - data_range.first.fst;
        int len_snd = data_range.second.snd - data_range.first.snd;
        cmp_number(data_pair<int>::make(len_fst, len_snd));
    }
}

void
cmp::walk_tag(bool align, tag_info &tinfo,
              const data_pair<uint32_t> &tag_variants) {
    cmp_number(tag_variants);
    if (result != 0)
        return;
    data<cmp,ptr_pair>::walk_variant(align, tinfo, tag_variants.fst);
}

void
cmp::walk_struct(bool align, const uint8_t *end_sp) {
    while (!result && this->sp != end_sp) {
        this->walk(align);
        align = true;
    }
}

void
cmp::walk_res(bool align, const rust_fn *dtor, uint16_t n_ty_params,
              const uint8_t *ty_params_sp) {
    abort();    // TODO
}

void
cmp::walk_variant(bool align, tag_info &tinfo, uint32_t variant_id,
                  const std::pair<const uint8_t *,const uint8_t *>
                  variant_ptr_and_end) {
    cmp sub(*this, variant_ptr_and_end.first, tinfo.params);

    const uint8_t *variant_end = variant_ptr_and_end.second;
    while (!result && sub.sp < variant_end) {
        sub.walk(align);
        result = sub.result;
        align = true;
    }
}


// Polymorphic logging, for convenience

void
log::walk_string(const std::pair<ptr,ptr> &data) {
    out << "\"" << std::hex;

    ptr subdp = data.first;
    while (subdp < data.second) {
        char ch = *subdp;
        if (isprint(ch))
            out << ch;
        else if (ch)
            out << "\\x" << std::setw(2) << std::setfill('0') << (int)ch;
        ++subdp;
    }

    out << "\"" << std::dec;
}

void
log::walk_struct(bool align, const uint8_t *end_sp) {
    out << "(";

    bool first = true;
    while (sp != end_sp) {
        if (!first)
            out << ", ";
        walk(align);
        align = true, first = false;
    }

    out << ")";
}

void
log::walk_vec(bool align, bool is_pod, const std::pair<ptr,ptr> &data) {
    if (peek() == SHAPE_U8) {
        sp++;   // It's a string. We handle this ourselves.
        walk_string(data);
        return;
    }

    out << "[";

    log sub(*this, data.first);

    bool first = true;
    while (sub.dp < data.second) {
        if (!first)
            out << ", ";

        sub.walk_reset(align);

        align = true;
        first = false;
    }

    out << "]";
}

void
log::walk_variant(bool align, tag_info &tinfo, uint32_t variant_id,
                  const std::pair<const uint8_t *,const uint8_t *>
                  variant_ptr_and_end) {
    log sub(*this, variant_ptr_and_end.first, tinfo.params);
    const uint8_t *variant_end = variant_ptr_and_end.second;

    bool first = true;
    while (sub.sp < variant_end) {
        out << (first ? "(" : ", ");

        sub.walk(align);

        align = true;
        first = false;
    }

    if (!first)
        out << ")";
}

} // end namespace shape

extern "C" void
upcall_cmp_type(int8_t *result, rust_task *task, type_desc *tydesc,
                const type_desc **subtydescs, uint8_t *data_0,
                uint8_t *data_1, uint8_t cmp_type) {
    shape::arena arena;
    shape::type_param *params = shape::type_param::make(tydesc, arena);
    shape::cmp cmp(task, tydesc->shape, params, tydesc->shape_tables, data_0,
                   data_1);
    cmp.walk(true);

    switch (cmp_type) {
    case shape::CMP_EQ: *result = cmp.result == 0;  break;
    case shape::CMP_LT: *result = cmp.result < 0;   break;
    case shape::CMP_LE: *result = cmp.result <= 0;  break;
    }
}

extern "C" void
upcall_log_type(rust_task *task, type_desc *tydesc, uint8_t *data,
                uint32_t level) {
    if (task->sched->log_lvl < level)
        return;     // TODO: Don't evaluate at all?

    shape::arena arena;
    shape::type_param *params = shape::type_param::make(tydesc, arena);

    std::stringstream ss;
    shape::log log(task, tydesc->shape, params, tydesc->shape_tables, data,
                   ss);

    log.walk(true);

    task->sched->log(task, level, "%s", ss.str().c_str());
}

