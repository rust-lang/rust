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

// Constants

const uint8_t CMP_EQ = 0u;
const uint8_t CMP_LT = 1u;
const uint8_t CMP_LE = 2u;


// Type parameters

type_param *
type_param::make(const type_desc **tydescs, unsigned n_tydescs,
                 arena &arena) {
    if (!n_tydescs)
        return NULL;

    type_param *ptrs = arena.alloc<type_param>(n_tydescs);
    for (uint32_t i = 0; i < n_tydescs; i++) {
        const type_desc *subtydesc = tydescs[i];
        ptrs[i].shape = subtydesc->shape;
        ptrs[i].tables = subtydesc->shape_tables;

        // FIXME: Doesn't handle a type-parametric object closing over a
        // type-parametric object type properly.
        ptrs[i].params = from_tydesc(subtydesc, arena);
    }
    return ptrs;
}

// Constructs type parameters from a function shape. This is a bit messy,
// because it requires that the function shape have a specific format.
type_param *
type_param::from_fn_shape(const uint8_t *sp, ptr dp, arena &arena) {
    const type_desc *tydesc = bump_dp<const type_desc *>(dp);
    const type_desc **descs = (const type_desc **)(dp + tydesc->size);
    unsigned n_tydescs = tydesc->n_obj_params & 0x7fffffff;
    return make(descs, n_tydescs, arena);
}

// Constructs type parameters from an object shape. This is also a bit messy,
// because it requires that the object shape have a specific format.
type_param *
type_param::from_obj_shape(const uint8_t *sp, ptr dp, arena &arena) {
    uint8_t shape = *sp++; assert(shape == SHAPE_STRUCT);
    get_u16_bump(sp);   // Skip over the size.
    shape = *sp++; assert(shape == SHAPE_PTR);
    shape = *sp++; assert(shape == SHAPE_STRUCT);

    unsigned n_tydescs = get_u16_bump(sp);

    // Type descriptors start right after the reference count.
    const type_desc **descs = (const type_desc **)(dp + sizeof(uintptr_t));

    return make(descs, n_tydescs, arena);
}


// A shape printer, useful for debugging

void
print::walk_tag(tag_info &tinfo) {
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
        sub.walk();
    }

    DPRINT(">");
}

void
print::walk_struct(const uint8_t *end_sp) {
    DPRINT("(");

    bool first = true;
    while (sp != end_sp) {
        if (!first)
            DPRINT(",");
        first = false;

        walk();
    }

    DPRINT(")");
}

void
print::walk_res(const rust_fn *dtor, unsigned n_params,
                const type_param *params, const uint8_t *end_sp) {
    DPRINT("res@%p", dtor);

    // Print type parameters.
    if (n_params) {
        DPRINT("<");

        bool first = true;
        for (uint16_t i = 0; i < n_params; i++) {
            if (!first)
                DPRINT(",");
            first = false;

            ctxt<print> sub(*this, params[i].shape);
            sub.walk();
        }

        DPRINT(">");
    }

    // Print arguments.

    if (sp == end_sp)
        return;

    DPRINT("(");

    bool first = true;
    while (sp != end_sp) {
        if (!first)
            DPRINT(",");
        first = false;

        walk();
    }

    DPRINT(")");
}

void
print::walk_var(uint8_t param_index) {
    DPRINT("%c=", 'T' + param_index);

    const type_param *param = &params[param_index];
    print sub(*this, param->shape, param->params, param->tables);
    sub.walk();
}

template<>
void print::walk_number<uint8_t>()      { DPRINT("u8"); }
template<>
void print::walk_number<uint16_t>()     { DPRINT("u16"); }
template<>
void print::walk_number<uint32_t>()     { DPRINT("u32"); }
template<>
void print::walk_number<uint64_t>()     { DPRINT("u64"); }
template<>
void print::walk_number<int8_t>()       { DPRINT("i8"); }
template<>
void print::walk_number<int16_t>()      { DPRINT("i16"); }
template<>
void print::walk_number<int32_t>()      { DPRINT("i32"); }
template<>
void print::walk_number<int64_t>()      { DPRINT("i64"); }
template<>
void print::walk_number<float>()        { DPRINT("f32"); }
template<>
void print::walk_number<double>()       { DPRINT("f64"); }


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
        sub.align = false;

        // Compute the size of this variant.
        size_align variant_sa;
        bool first = true;
        while (sub.sp != variant_end) {
            if (!first)
                variant_sa.size = align_to(variant_sa.size, sub.sa.alignment);
            sub.walk();
            sub.align = true, first = false;

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
        tinfo.tag_sa.add(sizeof(uint32_t), alignof<uint32_t>());
    }
}

void
size_of::walk_tag(tag_info &tinfo) {
    compute_tag_size(*this, tinfo);
    sa = tinfo.tag_sa;
}

void
size_of::walk_struct(const uint8_t *end_sp) {
    size_align struct_sa(0, 1);

    bool first = true;
    while (sp != end_sp) {
        if (!first)
            struct_sa.size = align_to(struct_sa.size, sa.alignment);
        walk();
        align = true, first = false;

        struct_sa.add(sa);
    }

    sa = struct_sa;
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
    void walk_vec(bool is_pod,
                  const std::pair<ptr_pair,ptr_pair> &data_range);

    inline void walk_subcontext(cmp &sub) {
        sub.walk();
        result = sub.result;
    }

    inline void walk_box_contents(cmp &sub, ptr_pair &ref_count_dp) {
        sub.align = true;
        sub.walk();
        result = sub.result;
    }

    inline void walk_uniq_contents(cmp &sub) {
        sub.align = true;
        sub.walk();
        result = sub.result;
    }

    inline void cmp_two_pointers() {
        ALIGN_TO(alignof<void *>() * 2);
        data_pair<uint8_t *> fst = bump_dp<uint8_t *>(dp);
        data_pair<uint8_t *> snd = bump_dp<uint8_t *>(dp);
        cmp_number(fst);
        if (!result)
            cmp_number(snd);
    }

    inline void cmp_pointer() {
        ALIGN_TO(alignof<void *>());
        cmp_number(bump_dp<uint8_t *>(dp));
    }

    template<typename T>
    void cmp_number(const data_pair<T> &nums) {
        result = (nums.fst < nums.snd) ? -1 : (nums.fst == nums.snd) ? 0 : 1;
    }

public:
    int result;

    cmp(rust_task *in_task,
        bool in_align,
        const uint8_t *in_sp,
        const type_param *in_params,
        const rust_shape_tables *in_tables,
        uint8_t *in_data_0,
        uint8_t *in_data_1)
    : data<cmp,ptr_pair>(in_task, in_align, in_sp, in_params, in_tables,
                         ptr_pair::make(in_data_0, in_data_1)),
      result(0) {}

    cmp(const cmp &other,
        const uint8_t *in_sp = NULL,
        const type_param *in_params = NULL,
        const rust_shape_tables *in_tables = NULL)
    : data<cmp,ptr_pair>(other.task,
                         other.align,
                         in_sp ? in_sp : other.sp,
                         in_params ? in_params : other.params,
                         in_tables ? in_tables : other.tables,
                         other.dp),
      result(0) {}

    cmp(const cmp &other, const ptr_pair &in_dp)
    : data<cmp,ptr_pair>(other.task,
                         other.align,
                         other.sp,
                         other.params,
                         other.tables,
                         in_dp),
      result(0) {}

    void walk_vec(bool is_pod, uint16_t sp_size) {
        walk_vec(is_pod, get_vec_data_range(dp));
    }

    void walk_box() {
        data<cmp,ptr_pair>::walk_box_contents();
    }

    void walk_uniq() {
        data<cmp,ptr_pair>::walk_uniq_contents();
    }

    void walk_fn()  { return cmp_two_pointers(); }
    void walk_obj() { return cmp_two_pointers(); }

    void walk_tag(tag_info &tinfo, const data_pair<uint32_t> &tag_variants);
    void walk_struct(const uint8_t *end_sp);
    void walk_res(const rust_fn *dtor, uint16_t n_ty_params,
                  const type_param *ty_params_sp, const uint8_t *end_sp,
                  const data_pair<uintptr_t> &live);
    void walk_variant(tag_info &tinfo, uint32_t variant_id,
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
cmp::walk_vec(bool is_pod, const std::pair<ptr_pair,ptr_pair> &data_range) {
    cmp sub(*this, data_range.first);
    ptr_pair data_end = sub.end_dp = data_range.second;
    while (!result && sub.dp < data_end) {
        sub.walk_reset();
        result = sub.result;
        sub.align = true;
    }

    if (!result) {
        // If we hit the end, the result comes down to length comparison.
        int len_fst = data_range.second.fst - data_range.first.fst;
        int len_snd = data_range.second.snd - data_range.first.snd;
        cmp_number(data_pair<int>::make(len_fst, len_snd));
    }
}

void
cmp::walk_tag(tag_info &tinfo, const data_pair<uint32_t> &tag_variants) {
    cmp_number(tag_variants);
    if (result != 0)
        return;
    data<cmp,ptr_pair>::walk_variant(tinfo, tag_variants.fst);
}

void
cmp::walk_struct(const uint8_t *end_sp) {
    while (!result && this->sp != end_sp) {
        this->walk();
        align = true;
    }
}

void
cmp::walk_res(const rust_fn *dtor, uint16_t n_ty_params,
              const type_param *ty_params_sp, const uint8_t *end_sp,
              const data_pair<uintptr_t> &live) {
    abort();    // TODO
}

void
cmp::walk_variant(tag_info &tinfo, uint32_t variant_id,
                  const std::pair<const uint8_t *,const uint8_t *>
                  variant_ptr_and_end) {
    cmp sub(*this, variant_ptr_and_end.first, tinfo.params);

    const uint8_t *variant_end = variant_ptr_and_end.second;
    while (!result && sub.sp < variant_end) {
        sub.walk();
        result = sub.result;
        sub.align = true;
    }
}


// Polymorphic logging, for convenience

void
log::walk_string(const std::pair<ptr,ptr> &data) {
    out << prefix << "\"" << std::hex;

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
log::walk_struct(const uint8_t *end_sp) {
    out << prefix << "(";
    prefix = "";

    bool first = true;
    while (sp != end_sp) {
        if (!first)
            out << ", ";
        walk();
        align = true, first = false;
    }

    out << ")";
}

void
log::walk_vec(bool is_pod, const std::pair<ptr,ptr> &data) {
    if (peek() == SHAPE_U8) {
        sp++;   // It's a string. We handle this ourselves.
        walk_string(data);
        return;
    }

    out << prefix << "[";

    log sub(*this, data.first);
    sub.end_dp = data.second;

    while (sub.dp < data.second) {
        sub.walk_reset();
        sub.align = true;
        sub.prefix = ", ";
    }

    out << "]";
}

void
log::walk_variant(tag_info &tinfo, uint32_t variant_id,
                  const std::pair<const uint8_t *,const uint8_t *>
                  variant_ptr_and_end) {
    log sub(*this, variant_ptr_and_end.first, tinfo.params);
    const uint8_t *variant_end = variant_ptr_and_end.second;

    bool first = true;
    while (sub.sp < variant_end) {
        out << (first ? "(" : ", ");
        sub.walk();
        sub.align = true, first = false;
    }

    if (!first)
        out << ")";
}

void
log::walk_res(const rust_fn *dtor, unsigned n_params,
              const type_param *params, const uint8_t *end_sp, bool live) {
    out << prefix << "res";

    if (this->sp == end_sp)
        return;

    out << "(";

    bool first = true;
    while (sp != end_sp) {
        if (!first)
            out << ", ";
        walk();
        align = true, first = false;
    }

    out << ")";
}

} // end namespace shape

extern "C" void
upcall_cmp_type(int8_t *result, rust_task *task, const type_desc *tydesc,
                const type_desc **subtydescs, uint8_t *data_0,
                uint8_t *data_1, uint8_t cmp_type) {
    shape::arena arena;

    // FIXME: This may well be broken when comparing two closures or objects
    // that close over different sets of type parameters.
    shape::type_param *params =
        shape::type_param::from_tydesc_and_data(tydesc, data_0, arena);

    shape::cmp cmp(task, true, tydesc->shape, params, tydesc->shape_tables,
                   data_0, data_1);
    cmp.walk();

    switch (cmp_type) {
    case shape::CMP_EQ: *result = cmp.result == 0;  break;
    case shape::CMP_LT: *result = cmp.result < 0;   break;
    case shape::CMP_LE: *result = cmp.result <= 0;  break;
    }
}

extern "C" void
upcall_log_type(const type_desc *tydesc, uint8_t *data, uint32_t level) {
    rust_task *task = rust_scheduler::get_task();
    if (task->sched->log_lvl < level)
        return;     // TODO: Don't evaluate at all?

    shape::arena arena;
    shape::type_param *params =
        shape::type_param::from_tydesc_and_data(tydesc, data, arena);

    std::stringstream ss;
    shape::log log(task, true, tydesc->shape, params, tydesc->shape_tables,
                   data, ss);

    log.walk();

    task->sched->log(task, level, "%s", ss.str().c_str());
}

