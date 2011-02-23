
const int rc_base_field_refcnt = 0;

// FIXME: import from std.dbg when imported consts work.
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
const int vec_elt_data = 3;

const int tydesc_field_first_param = 0;
const int tydesc_field_size = 1;
const int tydesc_field_align = 2;
const int tydesc_field_take_glue_off = 3;
const int tydesc_field_drop_glue_off = 4;
const int tydesc_field_free_glue_off = 5;
const int tydesc_field_sever_glue_off = 6;
const int tydesc_field_mark_glue_off = 7;
const int tydesc_field_obj_drop_glue_off = 8;
const int tydesc_field_is_stateful = 9;


const int obj_field_vtbl = 0;
const int obj_field_box = 1;

const int obj_body_elt_tydesc = 0;
const int obj_body_elt_typarams = 1;
const int obj_body_elt_fields = 2;

const int fn_field_code = 0;
const int fn_field_box = 1;

const int closure_elt_tydesc = 0;
const int closure_elt_target = 1;
const int closure_elt_bindings = 2;
const int closure_elt_ty_params = 3;


const int worst_case_glue_call_args = 7;

const int n_upcall_glues = 7;

const int abi_x86_rustboot_cdecl = 1;
const int abi_x86_rustc_fastcall = 2;

fn memcpy_glue_name() -> str {
    ret "rust_memcpy_glue";
}

fn bzero_glue_name() -> str {
    ret "rust_bzero_glue";
}

fn upcall_glue_name(int n) -> str {
    ret "rust_upcall_" + util.common.istr(n);
}

fn activate_glue_name() -> str {
    ret "rust_activate_glue";
}

fn yield_glue_name() -> str {
    ret "rust_yield_glue";
}

fn exit_task_glue_name() -> str {
    ret "rust_exit_task_glue";
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
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
