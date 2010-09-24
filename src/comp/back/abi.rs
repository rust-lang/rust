
const int rc_base_field_refcnt = 0;

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

const int calltup_elt_out_ptr = 0;
const int calltup_elt_task_ptr = 1;
const int calltup_elt_indirect_args = 2;
const int calltup_elt_ty_params = 3;
const int calltup_elt_args = 4;
const int calltup_elt_iterator_args = 5;

const int worst_case_glue_call_args = 7;

const int n_upcall_glues = 7;

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
