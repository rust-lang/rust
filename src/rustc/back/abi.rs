


const rc_base_field_refcnt: uint = 0u;

const task_field_refcnt: uint = 0u;

const task_field_stk: uint = 2u;

const task_field_runtime_sp: uint = 3u;

const task_field_rust_sp: uint = 4u;

const task_field_gc_alloc_chain: uint = 5u;

const task_field_dom: uint = 6u;

const n_visible_task_fields: uint = 7u;

const dom_field_interrupt_flag: uint = 1u;

const frame_glue_fns_field_mark: uint = 0u;

const frame_glue_fns_field_drop: uint = 1u;

const frame_glue_fns_field_reloc: uint = 2u;

const box_field_refcnt: uint = 0u;
const box_field_tydesc: uint = 1u;
const box_field_prev: uint = 2u;
const box_field_next: uint = 3u;
const box_field_body: uint = 4u;

const general_code_alignment: uint = 16u;

const tydesc_field_first_param: uint = 0u;
const tydesc_field_size: uint = 1u;
const tydesc_field_align: uint = 2u;
const tydesc_field_take_glue: uint = 3u;
const tydesc_field_drop_glue: uint = 4u;
const tydesc_field_free_glue: uint = 5u;
const tydesc_field_visit_glue: uint = 6u;
const tydesc_field_sever_glue: uint = 7u;
const tydesc_field_mark_glue: uint = 8u;
const tydesc_field_unused2: uint = 9u;
const tydesc_field_unused_2: uint = 10u;
const tydesc_field_shape: uint = 11u;
const tydesc_field_shape_tables: uint = 12u;
const tydesc_field_n_params: uint = 13u;
const tydesc_field_obj_params: uint = 14u; // FIXME unused (#2351)
const n_tydesc_fields: uint = 15u;

const cmp_glue_op_eq: uint = 0u;

const cmp_glue_op_lt: uint = 1u;

const cmp_glue_op_le: uint = 2u;

// The two halves of a closure: code and environment.
const fn_field_code: uint = 0u;
const fn_field_box: uint = 1u;

// closures, see trans_closure.rs
const closure_body_ty_params: uint = 0u;
const closure_body_bindings: uint = 1u;

const vec_elt_fill: uint = 0u;

const vec_elt_alloc: uint = 1u;

const vec_elt_elems: uint = 2u;

const slice_elt_base: uint = 0u;
const slice_elt_len: uint = 1u;

const worst_case_glue_call_args: uint = 7u;

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
