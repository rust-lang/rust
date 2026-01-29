//@ check-fail
//@ compile-flags: --crate-type=lib
//@ revisions: fixed_x18 indirect_branch_cs_prefix regparm
//@ revisions: reg_struct_return retpoline retpoline_external_thunk
//@[fixed_x18] compile-flags: --cfg target_modifier_fixed_x18
//@[indirect_branch_cs_prefix] compile-flags: --cfg target_modifier_indirect_branch_cs_prefix
//@[regparm] compile-flags: --cfg target_modifier_regparm="0"
//@[reg_struct_return] compile-flags: --cfg target_modifier_reg_struct_return
//@[retpoline] compile-flags: --cfg target_modifier_retpoline
//@[retpoline_external_thunk] compile-flags: --cfg target_modifier_retpoline_external_thunk

fn main() {}

//~? ERROR unexpected `--cfg
