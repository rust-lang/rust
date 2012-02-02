


// FIXME: Most of these should be uints.
const rc_base_field_refcnt: int = 0;

const task_field_refcnt: int = 0;

const task_field_stk: int = 2;

const task_field_runtime_sp: int = 3;

const task_field_rust_sp: int = 4;

const task_field_gc_alloc_chain: int = 5;

const task_field_dom: int = 6;

const n_visible_task_fields: int = 7;

const dom_field_interrupt_flag: int = 1;

const frame_glue_fns_field_mark: int = 0;

const frame_glue_fns_field_drop: int = 1;

const frame_glue_fns_field_reloc: int = 2;

// n.b. must be same as cbox_elt_refcnt
const box_rc_field_refcnt: int = 0;
const box_rc_field_body: int = 1;

const general_code_alignment: int = 16;

const tydesc_field_first_param: int = 0;
const tydesc_field_size: int = 1;
const tydesc_field_align: int = 2;
const tydesc_field_take_glue: int = 3;
const tydesc_field_drop_glue: int = 4;
const tydesc_field_free_glue: int = 5;
const tydesc_field_unused: int = 6;
const tydesc_field_sever_glue: int = 7;
const tydesc_field_mark_glue: int = 8;
const tydesc_field_unused2: int = 9;
const tydesc_field_cmp_glue: int = 10;
const tydesc_field_shape: int = 11;
const tydesc_field_shape_tables: int = 12;
const tydesc_field_n_params: int = 13;
const tydesc_field_obj_params: int = 14; // FIXME unused
const n_tydesc_fields: int = 15;

const cmp_glue_op_eq: uint = 0u;

const cmp_glue_op_lt: uint = 1u;

const cmp_glue_op_le: uint = 2u;

// The two halves of a closure: code and environment.
const fn_field_code: int = 0;
const fn_field_box: int = 1;

// closure_box, see trans_closure.rs
//
// n.b. the refcnt must be compatible with a normal box
const cbox_elt_refcnt: int = 0;
const cbox_elt_tydesc: int = 1;
const cbox_elt_ty_params: int = 2;
const cbox_elt_bindings: int = 3;

const vec_elt_fill: int = 0;

const vec_elt_alloc: int = 1;

const vec_elt_elems: int = 2;

const worst_case_glue_call_args: int = 7;

const abi_version: uint = 1u;

fn memcpy_glue_name() -> str { ret "rust_memcpy_glue"; }

fn bzero_glue_name() -> str { ret "rust_bzero_glue"; }

fn yield_glue_name() -> str { ret "rust_yield_glue"; }

fn no_op_type_glue_name() -> str { ret "rust_no_op_type_glue"; }
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
