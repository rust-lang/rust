// Rust cycle collector. Temporary, but will probably stick around for some
// time until LLVM's GC infrastructure is more mature.

#include "rust_debug.h"
#include "rust_gc.h"
#include "rust_internal.h"
#include "rust_shape.h"
#include "rust_task.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <vector>
#include <stdint.h>

// The number of allocations Rust code performs before performing cycle
// collection.
#define RUST_CC_FREQUENCY   5000

namespace cc {

// Internal reference count computation

typedef std::map<void *,uintptr_t> irc_map;

class irc : public shape::data<irc,shape::ptr> {
    friend class shape::data<irc,shape::ptr>;

    irc_map &ircs;

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
        // Record an irc for the environment box, but don't descend
        // into it since it will be walked via the box's allocation
        dp += sizeof(void *); // skip code pointer
        uint8_t * box_ptr = shape::bump_dp<uint8_t *>(dp);
        shape::ptr ref_count_dp(box_ptr);
        maybe_record_irc(ref_count_dp);
    }

    void walk_obj() {
        dp += sizeof(void *); // skip vtable
        uint8_t *box_ptr = shape::bump_dp<uint8_t *>(dp);
        shape::ptr ref_count_dp(box_ptr);
        maybe_record_irc(ref_count_dp);
    }

    void walk_iface() {
        //shape::data<irc,shape::ptr>::walk_iface_contents(dp);
        shape::data<irc,shape::ptr>::walk_box_contents();
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
        maybe_record_irc(ref_count_dp);

        // Do not traverse the contents of this box; it's in the allocation
        // somewhere, so we're guaranteed to come back to it (if we haven't
        // traversed it already).
    }

    void maybe_record_irc(shape::ptr &ref_count_dp) {
        if (!ref_count_dp)
            return;

        // Bump the internal reference count of the box.
        if (ircs.find((void *)ref_count_dp) == ircs.end()) {
          LOG(task, gc,
              "setting internal reference count for %p to 1",
              (void *)ref_count_dp);
          ircs[(void *)ref_count_dp] = 1;
        } else {
          uintptr_t newcount = ircs[(void *)ref_count_dp] + 1;
          LOG(task, gc,
              "bumping internal reference count for %p to %lu",
              (void *)ref_count_dp, newcount);
          ircs[(void *)ref_count_dp] = newcount;
        }
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
    std::map<void *,const type_desc *>::iterator
        begin(task->local_allocs.begin()), end(task->local_allocs.end());
    while (begin != end) {
        uint8_t *p = reinterpret_cast<uint8_t *>(begin->first);

        const type_desc *tydesc = begin->second;

        LOG(task, gc, "determining internal ref counts: %p, tydesc=%p", p,
            tydesc);

        shape::arena arena;
        shape::type_param *params =
            shape::type_param::from_tydesc_and_data(tydesc, p, arena);

#if 0
        shape::print print(task, true, tydesc->shape, params,
                           tydesc->shape_tables);
        print.walk();

        shape::log log(task, true, tydesc->shape, params,
                       tydesc->shape_tables, p + sizeof(uintptr_t),
                       std::cerr);
        log.walk();
#endif

        irc irc(task, true, tydesc->shape, params, tydesc->shape_tables,
                p + sizeof(uintptr_t), ircs);
        irc.walk();

        ++begin;
    }
}


// Root finding

void
find_roots(rust_task *task, irc_map &ircs, std::vector<void *> &roots) {
    std::map<void *,const type_desc *>::iterator
        begin(task->local_allocs.begin()), end(task->local_allocs.end());
    while (begin != end) {
        void *alloc = begin->first;
        uintptr_t *ref_count_ptr = reinterpret_cast<uintptr_t *>(alloc);
        uintptr_t ref_count = *ref_count_ptr;

        uintptr_t irc;
        if (ircs.find(alloc) != ircs.end())
            irc = ircs[alloc];
        else
            irc = 0;

        if (irc < ref_count) {
            // This allocation must be a root, because the internal reference
            // count is smaller than the total reference count.
            LOG(task, gc,"root found: %p, irc %lu, ref count %lu",
                alloc, irc, ref_count);
            roots.push_back(alloc);
        } else {
            LOG(task, gc, "nonroot found: %p, irc %lu, ref count %lu",
                alloc, irc, ref_count);
            assert(irc == ref_count && "Internal reference count must be "
                   "less than or equal to the total reference count!");
        }

        ++begin;
    }
}


// Marking

class mark : public shape::data<mark,shape::ptr> {
    friend class shape::data<mark,shape::ptr>;

    std::set<void *> &marked;

    mark(const mark &other, const shape::ptr &in_dp)
    : shape::data<mark,shape::ptr>(other.task, other.align, other.sp,
                                   other.params, other.tables, in_dp),
      marked(other.marked) {}

    mark(const mark &other,
         const uint8_t *in_sp,
         const shape::type_param *in_params,
         const rust_shape_tables *in_tables = NULL)
    : shape::data<mark,shape::ptr>(other.task,
                                   other.align,
                                   in_sp,
                                   in_params,
                                   in_tables ? in_tables : other.tables,
                                   other.dp),
      marked(other.marked) {}

    mark(const mark &other,
         const uint8_t *in_sp,
         const shape::type_param *in_params,
         const rust_shape_tables *in_tables,
         shape::ptr in_dp)
    : shape::data<mark,shape::ptr>(other.task,
                                   other.align,
                                   in_sp,
                                   in_params,
                                   in_tables,
                                   in_dp),
      marked(other.marked) {}

    mark(rust_task *in_task,
         bool in_align,
         const uint8_t *in_sp,
         const shape::type_param *in_params,
         const rust_shape_tables *in_tables,
         uint8_t *in_data,
         std::set<void *> &in_marked)
    : shape::data<mark,shape::ptr>(in_task, in_align, in_sp, in_params,
                                   in_tables, in_data),
      marked(in_marked) {}

    void walk_vec(bool is_pod, uint16_t sp_size) {
        if (is_pod || shape::get_dp<void *>(dp) == NULL)
            return;     // There can't be any outbound pointers from this.

        std::pair<uint8_t *,uint8_t *> data_range(get_vec_data_range(dp));
        if (data_range.second - data_range.first > 100000)
            abort();    // FIXME: Temporary sanity check.

        mark sub(*this, data_range.first);
        shape::ptr data_end = sub.end_dp = data_range.second;
        while (sub.dp < data_end) {
            sub.walk_reset();
            align = true;
        }
    }

    void walk_tag(shape::tag_info &tinfo, uint32_t tag_variant) {
        shape::data<mark,shape::ptr>::walk_variant(tinfo, tag_variant);
    }

    void walk_box() {
        shape::data<mark,shape::ptr>::walk_box_contents();
    }

    void walk_fn() {
        shape::data<mark,shape::ptr>::walk_fn_contents(dp);
    }

    void walk_obj() {
        shape::data<mark,shape::ptr>::walk_obj_contents(dp);
    }

    void walk_res(const shape::rust_fn *dtor, unsigned n_params,
                  const shape::type_param *params, const uint8_t *end_sp,
                  bool live) {
        while (this->sp != end_sp) {
            this->walk();
            align = true;
        }
    }

    void walk_subcontext(mark &sub) { sub.walk(); }

    void walk_box_contents(mark &sub, shape::ptr &ref_count_dp) {
        if (!ref_count_dp)
            return;

        if (marked.find((void *)ref_count_dp) != marked.end())
            return; // Skip to avoid chasing cycles.

        marked.insert((void *)ref_count_dp);
        sub.walk();
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
    static void do_mark(rust_task *task, const std::vector<void *> &roots,
                        std::set<void *> &marked);
};

void
mark::walk_variant(shape::tag_info &tinfo, uint32_t variant_id,
                   const std::pair<const uint8_t *,const uint8_t *>
                   variant_ptr_and_end) {
    mark sub(*this, variant_ptr_and_end.first, tinfo.params);

    assert(variant_id < 256);   // FIXME: Temporary sanity check.

    const uint8_t *variant_end = variant_ptr_and_end.second;
    while (sub.sp < variant_end) {
        sub.walk();
        align = true;
    }
}

void
mark::do_mark(rust_task *task, const std::vector<void *> &roots,
              std::set<void *> &marked) {
    std::vector<void *>::const_iterator begin(roots.begin()),
                                        end(roots.end());
    while (begin != end) {
        void *alloc = *begin;
        if (marked.find(alloc) == marked.end()) {
            marked.insert(alloc);

            const type_desc *tydesc = task->local_allocs[alloc];

            LOG(task, gc, "marking: %p, tydesc=%p", alloc, tydesc);

            uint8_t *p = reinterpret_cast<uint8_t *>(alloc);
            shape::arena arena;
            shape::type_param *params =
                shape::type_param::from_tydesc_and_data(tydesc, p, arena);

#if 0
            // We skip over the reference count here.
            shape::log log(task, true, tydesc->shape, params,
                           tydesc->shape_tables, p + sizeof(uintptr_t),
                           std::cerr);
            log.walk();
#endif

            // We skip over the reference count here.
            mark mark(task, true, tydesc->shape, params, tydesc->shape_tables,
                      p + sizeof(uintptr_t), marked);
            mark.walk();
        }

        ++begin;
    }
}

class sweep : public shape::data<sweep,shape::ptr> {
    friend class shape::data<sweep,shape::ptr>;

    sweep(const sweep &other, const shape::ptr &in_dp)
        : shape::data<sweep,shape::ptr>(other.task, other.align,
                                        other.sp, other.params,
                                        other.tables, in_dp) {}

    sweep(const sweep &other,
          const uint8_t *in_sp,
          const shape::type_param *in_params,
          const rust_shape_tables *in_tables = NULL)
        : shape::data<sweep,shape::ptr>(other.task,
                                        other.align,
                                        in_sp,
                                        in_params,
                                        in_tables ? in_tables : other.tables,
                                        other.dp) {}

    sweep(const sweep &other,
          const uint8_t *in_sp,
          const shape::type_param *in_params,
          const rust_shape_tables *in_tables,
          shape::ptr in_dp)
        : shape::data<sweep,shape::ptr>(other.task,
                                        other.align,
                                        in_sp,
                                        in_params,
                                        in_tables,
                                        in_dp) {}

    sweep(rust_task *in_task,
          bool in_align,
          const uint8_t *in_sp,
          const shape::type_param *in_params,
          const rust_shape_tables *in_tables,
          uint8_t *in_data)
        : shape::data<sweep,shape::ptr>(in_task, in_align, in_sp,
                                        in_params, in_tables, in_data) {}

    void walk_vec(bool is_pod, uint16_t sp_size) {
        void *vec = shape::get_dp<void *>(dp);
        walk_vec(is_pod, get_vec_data_range(dp));
        task->kernel->free(vec);
    }

    void walk_vec(bool is_pod,
                  const std::pair<shape::ptr,shape::ptr> &data_range) {
        sweep sub(*this, data_range.first);
        shape::ptr data_end = sub.end_dp = data_range.second;
        while (sub.dp < data_end) {
            sub.walk_reset();
            sub.align = true;
        }
    }

    void walk_tag(shape::tag_info &tinfo, uint32_t tag_variant) {
        shape::data<sweep,shape::ptr>::walk_variant(tinfo, tag_variant);
    }

    void walk_box() {
        shape::data<sweep,shape::ptr>::walk_box_contents();
    }

    void walk_fn() {
        return;
    }

    void walk_obj() {
        return;
    }

    void walk_iface() {
        //shape::data<sweep,shape::ptr>::walk_iface_contents(dp);
        shape::data<sweep,shape::ptr>::walk_box_contents();
    }

    void walk_res(const shape::rust_fn *dtor, unsigned n_params,
                  const shape::type_param *params, const uint8_t *end_sp,
                  bool live) {
        while (this->sp != end_sp) {
            this->walk();
            align = true;
        }
    }

    void walk_subcontext(sweep &sub) { sub.walk(); }

    void walk_box_contents(sweep &sub, shape::ptr &ref_count_dp) {
        return;
    }

    void walk_struct(const uint8_t *end_sp) {
        while (this->sp != end_sp) {
            this->walk();
            align = true;
        }
    }

    void walk_variant(shape::tag_info &tinfo, uint32_t variant_id,
                      const std::pair<const uint8_t *,const uint8_t *>
                      variant_ptr_and_end) {
        sweep sub(*this, variant_ptr_and_end.first, tinfo.params);

        const uint8_t *variant_end = variant_ptr_and_end.second;
        while (sub.sp < variant_end) {
            sub.walk();
            align = true;
        }
    }

    template<typename T>
    inline void walk_number() { /* no-op */ }

public:
    static void do_sweep(rust_task *task, const std::set<void *> &marked);
};

void
sweep::do_sweep(rust_task *task, const std::set<void *> &marked) {
    std::map<void *,const type_desc *>::iterator
        begin(task->local_allocs.begin()), end(task->local_allocs.end());
    while (begin != end) {
        void *alloc = begin->first;

        if (marked.find(alloc) == marked.end()) {
            LOG(task, gc, "object is part of a cycle: %p", alloc);

            const type_desc *tydesc = begin->second;
            uint8_t *p = reinterpret_cast<uint8_t *>(alloc);
            shape::arena arena;
            shape::type_param *params =
                shape::type_param::from_tydesc_and_data(tydesc, p, arena);

            sweep sweep(task, true, tydesc->shape,
                        params, tydesc->shape_tables,
                        p + sizeof(uintptr_t));
            sweep.walk();

            // FIXME: Run the destructor, *if* it's a resource.
            task->free(alloc);
        }
        ++begin;
    }
}


void
do_cc(rust_task *task) {
    LOG(task, gc, "cc; n allocs = %lu",
        (long unsigned int)task->local_allocs.size());

    irc_map ircs;
    irc::compute_ircs(task, ircs);

    std::vector<void *> roots;
    find_roots(task, ircs, roots);

    std::set<void *> marked;
    mark::do_mark(task, roots, marked);

    sweep::do_sweep(task, marked);
}

void
maybe_cc(rust_task *task) {
    static debug::flag zeal("RUST_CC_ZEAL");
    if (*zeal) {
        do_cc(task);
        return;
    }

    // FIXME: Needs a snapshot.
#if 0
    if (task->cc_counter++ > RUST_CC_FREQUENCY) {
        task->cc_counter = 0;
        do_cc(task);
    }
#endif
}

}   // end namespace cc

