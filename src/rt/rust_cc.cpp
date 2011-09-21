// Rust cycle collector. Temporary, but will probably stick around for some
// time until LLVM's GC infrastructure is more mature.

#include "rust_gc.h"
#include "rust_internal.h"
#include "rust_shape.h"
#include "rust_task.h"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>
#include <stdint.h>

#undef DPRINT
#define DPRINT(fmt,...)     fprintf(stderr, fmt, ##__VA_ARGS__)

namespace cc {

// Internal reference count computation

typedef std::map<void *,uintptr_t> irc_map;

class irc : public shape::data<irc,shape::ptr> {
    friend class shape::data<irc,shape::ptr>;

    irc_map ircs;

    irc(const irc &other, const shape::ptr &in_dp)
    : shape::data<irc,shape::ptr>(other.task, other.align, other.sp,
                                  other.params, other.tables, in_dp),
      ircs(other.ircs) {}

    irc(const irc &other,
        const uint8_t *in_sp,
        const shape::type_param *in_params,
        const rust_shape_tables *in_tables = NULL)
    : shape::data<irc,shape::ptr>(other.task,
                                  other.align,
                                  in_sp,
                                  in_params,
                                  in_tables ? in_tables : other.tables,
                                  other.dp),
      ircs(other.ircs) {}

    irc(const irc &other,
        const uint8_t *in_sp,
        const shape::type_param *in_params,
        const rust_shape_tables *in_tables,
        shape::ptr in_dp)
    : shape::data<irc,shape::ptr>(other.task,
                                  other.align,
                                  in_sp,
                                  in_params,
                                  in_tables,
                                  in_dp),
      ircs(other.ircs) {}

    irc(rust_task *in_task,
        bool in_align,
        const uint8_t *in_sp,
        const shape::type_param *in_params,
        const rust_shape_tables *in_tables,
        uint8_t *in_data,
        irc_map &in_ircs)
    : shape::data<irc,shape::ptr>(in_task, in_align, in_sp, in_params,
                                  in_tables, in_data),
      ircs(in_ircs) {}

    void walk_vec(bool is_pod, uint16_t sp_size) {
        if (is_pod || shape::get_dp<void *>(dp) == NULL)
            return;     // There can't be any outbound pointers from this.

        std::pair<uint8_t *,uint8_t *> data_range(get_vec_data_range(dp));
        if (data_range.second - data_range.first > 100000)
            abort();    // FIXME: Temporary sanity check.

        irc sub(*this, data_range.first);
        shape::ptr data_end = sub.end_dp = data_range.second;
        while (sub.dp < data_end) {
            sub.walk_reset();
            align = true;
        }
    }

    void walk_tag(shape::tag_info &tinfo, uint32_t tag_variant) {
        shape::data<irc,shape::ptr>::walk_variant(tinfo, tag_variant);
    }

    void walk_box() {
        shape::data<irc,shape::ptr>::walk_box_contents();
    }

    void walk_fn() {
        shape::data<irc,shape::ptr>::walk_fn_contents(dp);
    }

    void walk_obj() {
        shape::data<irc,shape::ptr>::walk_obj_contents(dp);
    }

    void walk_res(const shape::rust_fn *dtor, unsigned n_params,
                  const shape::type_param *params, const uint8_t *end_sp,
                  bool live) {
        while (this->sp != end_sp) {
            this->walk();
            align = true;
        }
    }

    void walk_subcontext(irc &sub) { sub.walk(); }

    void walk_box_contents(irc &sub, shape::ptr &ref_count_dp) {
        if (!ref_count_dp)
            return;

        // Bump the internal reference count of the box.
        if (ircs.find((void *)dp) == ircs.end())
            ircs[(void *)dp] = 1;
        else
            ++ircs[(void *)dp];

        // Do not traverse the contents of this box; it's in the allocation
        // somewhere, so we're guaranteed to come back to it (if we haven't
        // traversed it already).
    }

    void walk_struct(const uint8_t *end_sp) {
        while (this->sp != end_sp) {
            this->walk();
            align = true;
        }
    }

    void walk_variant(shape::tag_info &tinfo, uint32_t variant_id,
                      const std::pair<const uint8_t *,const uint8_t *>
                      variant_ptr_and_end);

    template<typename T>
    inline void walk_number() { /* no-op */ }

public:
    static void compute_ircs(rust_task *task, irc_map &ircs);
};

void
irc::walk_variant(shape::tag_info &tinfo, uint32_t variant_id,
                  const std::pair<const uint8_t *,const uint8_t *>
                  variant_ptr_and_end) {
    irc sub(*this, variant_ptr_and_end.first, tinfo.params);

    assert(variant_id < 256);   // FIXME: Temporary sanity check.

    const uint8_t *variant_end = variant_ptr_and_end.second;
    while (sub.sp < variant_end) {
        sub.walk();
        align = true;
    }
}

void
irc::compute_ircs(rust_task *task, irc_map &ircs) {
    std::map<void *,type_desc *>::iterator begin(task->local_allocs.begin()),
                                           end(task->local_allocs.end());
    while (begin != end) {
        uint8_t *p = reinterpret_cast<uint8_t *>(begin->first);
        type_desc *tydesc = begin->second;

        DPRINT("determining internal ref counts: %p, tydesc=%p\n", p, tydesc);

        // Prevents warnings for now
        shape::arena arena;
        shape::type_param *params =
            shape::type_param::from_tydesc(tydesc, arena);
        irc irc(task, true, tydesc->shape, params, tydesc->shape_tables, p,
                ircs);
        irc.walk();

        ++begin;
    }
}


void
do_cc(rust_task *task) {
    irc_map ircs;
    irc::compute_ircs(task, ircs);
}

void
maybe_cc(rust_task *task) {
    // FIXME: We ought to lock this.
    static int zeal = -1;
    if (zeal == -1) {
        char *ev = getenv("RUST_CC_ZEAL");
        zeal = ev && ev[0] != '\0' && ev[0] != '0';
    }

    if (zeal)
        do_cc(task);
}

}   // end namespace cc

