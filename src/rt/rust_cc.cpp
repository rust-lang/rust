// Rust cycle collector. Temporary, but will probably stick around for some
// time until LLVM's GC infrastructure is more mature.

#include "rust_debug.h"
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
#include <ios>

// The number of allocations Rust code performs before performing cycle
// collection.
#define RUST_CC_FREQUENCY   5000

// defined in rust_upcall.cpp:
void upcall_s_free_shared_type_desc(type_desc *td);

using namespace std;

namespace cc {

// Internal reference count computation

typedef std::map<rust_opaque_box*,uintptr_t> irc_map;

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

    void walk_vec2(bool is_pod, uint16_t sp_size) {
        if (is_pod || shape::get_dp<void *>(dp) == NULL)
            return;     // There can't be any outbound pointers from this.

        std::pair<uint8_t *,uint8_t *> data_range(get_vec_data_range(dp));
        irc sub(*this, data_range.first);
        shape::ptr data_end = sub.end_dp = data_range.second;
        while (sub.dp < data_end) {
            sub.walk_reset();
            align = true;
        }
    }

    void walk_tag2(shape::tag_info &tinfo, uint32_t tag_variant) {
        shape::data<irc,shape::ptr>::walk_variant1(tinfo, tag_variant);
    }

    void walk_box2() {
        // the box ptr can be NULL for env ptrs in closures and data
        // not fully initialized
        rust_opaque_box *box = *(rust_opaque_box**)dp;
        if (box)
            shape::data<irc,shape::ptr>::walk_box_contents1();
    }

    void walk_uniq2() {
        shape::data<irc,shape::ptr>::walk_uniq_contents1();
    }

    void walk_fn2(char code) {
        switch (code) {
          case shape::SHAPE_BOX_FN: {
              shape::bump_dp<void*>(dp); // skip over the code ptr
              walk_box2();               // walk over the environment ptr
              break;
          }
          case shape::SHAPE_BARE_FN:        // Does not close over data.
          case shape::SHAPE_STACK_FN:       // Not reachable from heap.
          case shape::SHAPE_UNIQ_FN: break; /* Can only close over sendable
                                             * (and hence acyclic) data */
          default: abort();
        }
    }

    void walk_iface2() {
        walk_box2();
    }

    void walk_tydesc2(char) {
    }

    void walk_res2(const shape::rust_fn *dtor, unsigned n_params,
                   const shape::type_param *params, const uint8_t *end_sp,
                   bool live) {
        while (this->sp != end_sp) {
            this->walk();
            align = true;
        }
    }

    void walk_subcontext2(irc &sub) { sub.walk(); }

    void walk_uniq_contents2(irc &sub) { sub.walk(); }

    void walk_box_contents2(irc &sub) {
        maybe_record_irc();

        // Do not traverse the contents of this box; it's in the allocation
        // somewhere, so we're guaranteed to come back to it (if we haven't
        // traversed it already).
    }

    void maybe_record_irc() {
        rust_opaque_box *box_ptr = *(rust_opaque_box **) dp;

        if (!box_ptr)
            return;

        // Bump the internal reference count of the box.
        if (ircs.find(box_ptr) == ircs.end()) {
          LOG(task, gc,
              "setting internal reference count for %p to 1",
              box_ptr);
          ircs[box_ptr] = 1;
        } else {
          uintptr_t newcount = ircs[box_ptr] + 1;
          LOG(task, gc,
              "bumping internal reference count for %p to %lu",
              box_ptr, newcount);
          ircs[box_ptr] = newcount;
        }
    }

    void walk_struct2(const uint8_t *end_sp) {
        while (this->sp != end_sp) {
            this->walk();
            align = true;
        }
    }

    void walk_variant2(shape::tag_info &tinfo, uint32_t variant_id,
                      const std::pair<const uint8_t *,const uint8_t *>
                      variant_ptr_and_end);

    template<typename T>
    inline void walk_number2() { /* no-op */ }

public:
    static void compute_ircs(rust_task *task, irc_map &ircs);
};

void
irc::walk_variant2(shape::tag_info &tinfo, uint32_t variant_id,
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
    boxed_region *boxed = &task->boxed;
    for (rust_opaque_box *box = boxed->first_live_alloc();
         box != NULL;
         box = box->next) {
        type_desc *tydesc = box->td;
        uint8_t *body = (uint8_t*) box_body(box);

        LOG(task, gc, 
            "determining internal ref counts: "
            "box=%p tydesc=%p body=%p",
            box, tydesc, body);
        
        shape::arena arena;
        shape::type_param *params =
            shape::type_param::from_tydesc_and_data(tydesc, body, arena);

        irc irc(task, true, tydesc->shape, params, tydesc->shape_tables,
                body, ircs);
        irc.walk();
    }
}


// Root finding

void
find_roots(rust_task *task, irc_map &ircs,
           std::vector<rust_opaque_box *> &roots) {
    boxed_region *boxed = &task->boxed;
    for (rust_opaque_box *box = boxed->first_live_alloc();
         box != NULL;
         box = box->next) {
        uintptr_t ref_count = box->ref_count;

        uintptr_t irc;
        if (ircs.find(box) != ircs.end())
            irc = ircs[box];
        else
            irc = 0;

        if (irc < ref_count) {
            // This allocation must be a root, because the internal reference
            // count is smaller than the total reference count.
            LOG(task, gc,"root found: %p, irc %lu, ref count %lu",
                box, irc, ref_count);
            roots.push_back(box);
        } else {
            LOG(task, gc, "nonroot found: %p, irc %lu, ref count %lu",
                box, irc, ref_count);
            assert(irc == ref_count && "Internal reference count must be "
                   "less than or equal to the total reference count!");
        }
    }
}


// Marking

class mark : public shape::data<mark,shape::ptr> {
    friend class shape::data<mark,shape::ptr>;

    std::set<rust_opaque_box *> &marked;

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
         std::set<rust_opaque_box*> &in_marked)
    : shape::data<mark,shape::ptr>(in_task, in_align, in_sp, in_params,
                                   in_tables, in_data),
      marked(in_marked) {}

    void walk_vec2(bool is_pod, uint16_t sp_size) {
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

    void walk_tag2(shape::tag_info &tinfo, uint32_t tag_variant) {
        shape::data<mark,shape::ptr>::walk_variant1(tinfo, tag_variant);
    }

    void walk_box2() {
        // the box ptr can be NULL for env ptrs in closures and data
        // not fully initialized
        rust_opaque_box *box = *(rust_opaque_box**)dp;
        if (box)
            shape::data<mark,shape::ptr>::walk_box_contents1();
    }

    void walk_uniq2() {
        shape::data<mark,shape::ptr>::walk_uniq_contents1();
    }

    void walk_fn2(char code) {
        switch (code) {
          case shape::SHAPE_BOX_FN: {
              shape::data<mark,shape::ptr>::walk_fn_contents1();
              break;
          }
          case shape::SHAPE_BARE_FN:        // Does not close over data.
          case shape::SHAPE_STACK_FN:       // Not reachable from heap.
          case shape::SHAPE_UNIQ_FN: break; /* Can only close over sendable
                                             * (and hence acyclic) data */
          default: abort();
        }
    }

    void walk_res2(const shape::rust_fn *dtor, unsigned n_params,
                  const shape::type_param *params, const uint8_t *end_sp,
                  bool live) {
        while (this->sp != end_sp) {
            this->walk();
            align = true;
        }
    }

    void walk_iface2() {
        walk_box2();
    }

    void walk_tydesc2(char) {
    }

    void walk_subcontext2(mark &sub) { sub.walk(); }

    void walk_uniq_contents2(mark &sub) { sub.walk(); }

    void walk_box_contents2(mark &sub) {
        rust_opaque_box *box_ptr = *(rust_opaque_box **) dp;

        if (!box_ptr)
            return;

        if (marked.find(box_ptr) != marked.end())
            return; // Skip to avoid chasing cycles.

        marked.insert(box_ptr);
        sub.walk();
    }

    void walk_struct2(const uint8_t *end_sp) {
        while (this->sp != end_sp) {
            this->walk();
            align = true;
        }
    }

    void walk_variant2(shape::tag_info &tinfo, uint32_t variant_id,
                      const std::pair<const uint8_t *,const uint8_t *>
                      variant_ptr_and_end);

    template<typename T>
    inline void walk_number2() { /* no-op */ }

public:
    static void do_mark(rust_task *task,
                        const std::vector<rust_opaque_box *> &roots,
                        std::set<rust_opaque_box*> &marked);
};

void
mark::walk_variant2(shape::tag_info &tinfo, uint32_t variant_id,
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
mark::do_mark(rust_task *task,
              const std::vector<rust_opaque_box *> &roots,
              std::set<rust_opaque_box *> &marked) {
    std::vector<rust_opaque_box *>::const_iterator 
      begin(roots.begin()),
      end(roots.end());
    while (begin != end) {
        rust_opaque_box *box = *begin;
        if (marked.find(box) == marked.end()) {
            marked.insert(box);

            const type_desc *tydesc = box->td;

            LOG(task, gc, "marking: %p, tydesc=%p", box, tydesc);

            uint8_t *p = (uint8_t*) box_body(box);
            shape::arena arena;
            shape::type_param *params =
                shape::type_param::from_tydesc_and_data(tydesc, p, arena);

            mark mark(task, true, tydesc->shape, params, tydesc->shape_tables,
                      p, marked);
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

    void walk_vec2(bool is_pod, uint16_t sp_size) {
        void *vec = shape::get_dp<void *>(dp);
        walk_vec2(is_pod, get_vec_data_range(dp));
        task->kernel->free(vec);
    }

    void walk_vec2(bool is_pod,
                  const std::pair<shape::ptr,shape::ptr> &data_range) {
        sweep sub(*this, data_range.first);
        shape::ptr data_end = sub.end_dp = data_range.second;
        while (sub.dp < data_end) {
            sub.walk_reset();
            sub.align = true;
        }
    }

    void walk_tag2(shape::tag_info &tinfo, uint32_t tag_variant) {
        shape::data<sweep,shape::ptr>::walk_variant1(tinfo, tag_variant);
    }

    void walk_uniq2() {
        void *x = *((void **)dp);
        // free contents first:
        shape::data<sweep,shape::ptr>::walk_uniq_contents1();
        // now free the ptr:
        task->kernel->free(x);
    }

    void walk_box2() {
        // In sweep phase, do not walk the box contents.  There is an
        // outer loop walking all remaining boxes, and this box may well
        // have been freed already!
    }

    void walk_fn2(char code) {
        switch (code) {
          case shape::SHAPE_UNIQ_FN: {
              fn_env_pair pair = *(fn_env_pair*)dp;

              if (pair.env) {
                  // free closed over data:
                  shape::data<sweep,shape::ptr>::walk_fn_contents1();
                  
                  // now free the embedded type descr:
                  upcall_s_free_shared_type_desc((type_desc*)pair.env->td);
                  
                  // now free the ptr:
                  task->kernel->free(pair.env);
              }
              break;
          }
          case shape::SHAPE_BOX_FN: {
              // the box will be visited separately:
              shape::bump_dp<void*>(dp); // skip over the code ptr
              walk_box2();               // walk over the environment ptr
              break;
          }
          case shape::SHAPE_BARE_FN:         // Does not close over data.
          case shape::SHAPE_STACK_FN: break; // Not reachable from heap.
          default: abort();
        }
    }

    void walk_obj2() {
        return;
    }

    void walk_iface2() {
        walk_box2();
    }

    void walk_tydesc2(char kind) {
        type_desc *td = *(type_desc **)dp;
        switch(kind) {
          case shape::SHAPE_TYDESC:
            break;
          case shape::SHAPE_SEND_TYDESC:
            upcall_s_free_shared_type_desc(td);
            break;
          default: abort();
        }
    }

    void walk_res2(const shape::rust_fn *dtor, unsigned n_params,
                   const shape::type_param *params, const uint8_t *end_sp,
                   bool live) {
        while (this->sp != end_sp) {
            this->walk();
            align = true;
        }
    }

    void walk_subcontext2(sweep &sub) { sub.walk(); }

    void walk_uniq_contents2(sweep &sub) { sub.walk(); }

    void walk_struct2(const uint8_t *end_sp) {
        while (this->sp != end_sp) {
            this->walk();
            align = true;
        }
    }

    void walk_variant2(shape::tag_info &tinfo, uint32_t variant_id,
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
    inline void walk_number2() { /* no-op */ }

public:
    static void do_sweep(rust_task *task,
                         const std::set<rust_opaque_box*> &marked);
};

void
sweep::do_sweep(rust_task *task,
                const std::set<rust_opaque_box*> &marked) {
    boxed_region *boxed = &task->boxed;
    rust_opaque_box *box = boxed->first_live_alloc();
    while (box != NULL) {
        // save next ptr as we may be freeing box
        rust_opaque_box *box_next = box->next;
        if (marked.find(box) == marked.end()) {
            LOG(task, gc, "object is part of a cycle: %p", box);

            const type_desc *tydesc = box->td;
            uint8_t *p = (uint8_t*) box_body(box);
            shape::arena arena;
            shape::type_param *params =
                shape::type_param::from_tydesc_and_data(tydesc, p, arena);

            sweep sweep(task, true, tydesc->shape,
                        params, tydesc->shape_tables,
                        p);
            sweep.walk();

            boxed->free(box);
        }
        box = box_next;
    }
}


void
do_cc(rust_task *task) {
    LOG(task, gc, "cc");

    irc_map ircs;
    irc::compute_ircs(task, ircs);

    std::vector<rust_opaque_box*> roots;
    find_roots(task, ircs, roots);

    std::set<rust_opaque_box*> marked;
    mark::do_mark(task, roots, marked);

    sweep::do_sweep(task, marked);
}

void
do_final_cc(rust_task *task) {
    do_cc(task);

    boxed_region *boxed = &task->boxed;
    for (rust_opaque_box *box = boxed->first_live_alloc();
         box != NULL;
         box = box->next) {
        cerr << "Unreclaimed object found at " << (void*) box << ": ";
        const type_desc *td = box->td;
        shape::arena arena;
        shape::type_param *params = shape::type_param::from_tydesc(td, arena);
        shape::log log(task, true, td->shape, params, td->shape_tables,
                       (uint8_t*)box_body(box), cerr);
        log.walk();
        cerr << "\n";
    }
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

