// FIXME: Most of these should be uints.

const int rc_base_field_refcnt = 0;

// FIXME: import from std::dbg when imported consts work.
const uint const_refcount = 0x7bad_face_u;

const int task_field_refcnt = 0;
const int task_field_stk = 2;
const int task_field_runtime_sp = 3;
const int task_field_rust_sp = 4;
const int task_field_gc_alloc_chain = 5;
const int task_field_dom = 6;
const int n_visible_task_fields = 7;

const int dom_field_interrupt_flag = 1;

const int frame_glue_fns_field_mark = 0;
const int frame_glue_fns_field_drop = 1;
const int frame_glue_fns_field_reloc = 2;

const int box_rc_field_refcnt = 0;
const int box_rc_field_body = 1;

const int general_code_alignment = 16;

const int vec_elt_rc = 0;
const int vec_elt_alloc = 1;
const int vec_elt_fill = 2;
const int vec_elt_pad = 3;
const int vec_elt_data = 4;

const int tydesc_field_first_param = 0;
const int tydesc_field_size = 1;
const int tydesc_field_align = 2;
const int tydesc_field_take_glue = 3;
const int tydesc_field_drop_glue = 4;
const int tydesc_field_free_glue = 5;
const int tydesc_field_sever_glue = 6;
const int tydesc_field_mark_glue = 7;
// FIXME no longer used in rustc, drop when rustboot is gone
const int tydesc_field_obj_drop_glue = 8;
const int tydesc_field_is_stateful = 9;
const int tydesc_field_cmp_glue = 10;
const int n_tydesc_fields = 11;

const uint cmp_glue_op_eq = 0u;
const uint cmp_glue_op_lt = 1u;
const uint cmp_glue_op_le = 2u;


const int obj_field_vtbl = 0;
const int obj_field_box = 1;

const int obj_body_elt_tydesc = 0;
const int obj_body_elt_typarams = 1;
const int obj_body_elt_fields = 2;
const int obj_body_elt_with_obj = 3; /* The base object to which an anonymous
                                      * object is attached */

const int fn_field_code = 0;
const int fn_field_box = 1;

const int closure_elt_tydesc = 0;
const int closure_elt_target = 1;
const int closure_elt_bindings = 2;
const int closure_elt_ty_params = 3;

const uint ivec_default_size = 64u;
const uint ivec_default_length = 8u;

const uint ivec_elt_len = 0u;
const uint ivec_elt_alen = 1u;
const uint ivec_elt_elems = 2u;
const uint ivec_heap_stub_elt_zero = 0u;
const uint ivec_heap_stub_elt_alen = 1u;
const uint ivec_heap_stub_elt_ptr = 2u;
const uint ivec_heap_elt_len = 0u;
const uint ivec_heap_elt_elems = 1u;


const int worst_case_glue_call_args = 7;

fn memcpy_glue_name() -> str {
    ret "rust_memcpy_glue";
}

fn bzero_glue_name() -> str {
    ret "rust_bzero_glue";
}

fn vec_append_glue_name() -> str {
    ret "rust_vec_append_glue";
}

fn yield_glue_name() -> str {
    ret "rust_yield_glue";
}

fn no_op_type_glue_name() -> str {
    ret "rust_no_op_type_glue";
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
