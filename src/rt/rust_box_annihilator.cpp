
#include "rust_globals.h"
#include "rust_task.h"
#include "rust_shape.h"

class annihilator : public shape::data<annihilator,shape::ptr> {
    friend class shape::data<annihilator,shape::ptr>;

    annihilator(const annihilator &other, const shape::ptr &in_dp)
        : shape::data<annihilator,shape::ptr>(other.task, other.align,
                                        other.sp,
                                        other.tables, in_dp) {}

    annihilator(const annihilator &other,
          const uint8_t *in_sp,
          const rust_shape_tables *in_tables = NULL)
        : shape::data<annihilator,shape::ptr>(other.task,
                                        other.align,
                                        in_sp,
                                        in_tables ? in_tables : other.tables,
                                        other.dp) {}

    annihilator(const annihilator &other,
          const uint8_t *in_sp,
          const rust_shape_tables *in_tables,
          shape::ptr in_dp)
        : shape::data<annihilator,shape::ptr>(other.task,
                                        other.align,
                                        in_sp,
                                        in_tables,
                                        in_dp) {}

    annihilator(rust_task *in_task,
          bool in_align,
          const uint8_t *in_sp,
          const rust_shape_tables *in_tables,
          uint8_t *in_data)
        : shape::data<annihilator,shape::ptr>(in_task, in_align, in_sp,
                                              in_tables,
                                              shape::ptr(in_data)) {}

    void walk_vec2(bool is_pod) {
        void *vec = shape::get_dp<void *>(dp);
        walk_vec2(is_pod, get_vec_data_range(dp));
        task->kernel->free(vec);
    }

    void walk_unboxed_vec2(bool is_pod) {
        walk_vec2(is_pod, get_unboxed_vec_data_range(dp));
    }

    void walk_fixedvec2(uint16_t n_elts, size_t elt_sz, bool is_pod) {
        walk_vec2(is_pod, get_fixedvec_data_range(n_elts, elt_sz, dp));
    }

    void walk_vec2(bool is_pod,
                  const std::pair<shape::ptr,shape::ptr> &data_range) {

        if (is_pod)
            return;

        annihilator sub(*this, data_range.first);
        shape::ptr data_end = sub.end_dp = data_range.second;
        while (sub.dp < data_end) {
            sub.walk_reset();
            sub.align = true;
        }
    }

    void walk_tag2(shape::tag_info &tinfo, uint32_t tag_variant) {
        shape::data<annihilator,shape::ptr>
          ::walk_variant1(tinfo, tag_variant);
    }

    void walk_rptr2() { }

    void walk_slice2(bool, bool) { }

    void walk_uniq2() {
        void *x = *((void **)dp);
        // free contents first:
        shape::data<annihilator,shape::ptr>::walk_uniq_contents1();
        // now free the ptr:
        task->kernel->free(x);
    }

    void walk_box2() {
        // In annihilator phase, do not walk the box contents.  There is an
        // outer loop walking all remaining boxes, and this box may well
        // have been freed already!
    }

    void walk_fn2(char code) {
        switch (code) {
          case shape::SHAPE_UNIQ_FN: {
              fn_env_pair pair = *(fn_env_pair*)dp;

              if (pair.env) {
                  // free closed over data:
                  shape::data<annihilator,shape::ptr>::walk_fn_contents1();

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

    void walk_trait2() {
        walk_box2();
    }

    void walk_tydesc2(char kind) {
        switch(kind) {
          case shape::SHAPE_TYDESC:
          case shape::SHAPE_SEND_TYDESC:
            break;
          default: abort();
        }
    }

    struct run_dtor_args {
        const shape::rust_fn *dtor;
        void *data;
    };

    typedef void (*dtor)(void **retptr, void *dptr);

    static void run_dtor(run_dtor_args *args) {
        dtor f = (dtor)args->dtor;
        f(NULL, args->data);
    }

    void walk_res2(const shape::rust_fn *dtor, const uint8_t *end_sp) {
        void *data = (void*)(uintptr_t)dp;
        // Switch back to the Rust stack to run the destructor
        run_dtor_args args = {dtor, data};
        task->call_on_rust_stack((void*)&args, (void*)run_dtor);

        while (this->sp != end_sp) {
            this->walk();
            align = true;
        }
    }

    void walk_subcontext2(annihilator &sub) { sub.walk(); }

    void walk_uniq_contents2(annihilator &sub) { sub.walk(); }

    void walk_struct2(const uint8_t *end_sp) {
        while (this->sp != end_sp) {
            this->walk();
            align = true;
        }
    }

    void walk_variant2(shape::tag_info &tinfo, uint32_t variant_id,
                      const std::pair<const uint8_t *,const uint8_t *>
                      variant_ptr_and_end) {
        annihilator sub(*this, variant_ptr_and_end.first);

        const uint8_t *variant_end = variant_ptr_and_end.second;
        while (sub.sp < variant_end) {
            sub.walk();
            align = true;
        }
    }

    template<typename T>
    inline void walk_number2() { /* no-op */ }

public:
    static void do_annihilate(rust_task *task, rust_opaque_box *box);
};

void
annihilator::do_annihilate(rust_task *task, rust_opaque_box *box) {
    const type_desc *tydesc = box->td;
    uint8_t *p = (uint8_t*) box_body(box);
    shape::arena arena;

    annihilator annihilator(task, true, tydesc->shape,
                            tydesc->shape_tables, p);
    annihilator.walk();
    task->boxed.free(box);
}

void
annihilate_box(rust_task *task, rust_opaque_box *box) {
    annihilator::do_annihilate(task, box);
}

void
annihilate_boxes(rust_task *task) {
    LOG(task, gc, "annihilating boxes for task %p", task);

    boxed_region *boxed = &task->boxed;
    rust_opaque_box *box = boxed->first_live_alloc();
    while (box != NULL) {
        rust_opaque_box *tmp = box;
        box = box->next;
        annihilate_box(task, tmp);
    }
}

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
