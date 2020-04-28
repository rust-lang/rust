; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s
source_filename = "julia"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-linux-gnu"

%jl_value_t = type opaque
%jl_array_t = type { i8 addrspace(13)*, i64, i16, i16, i32 }

@__stack_chk_guard = external constant %jl_value_t*
@jl_true = external constant %jl_value_t*
@jl_false = external constant %jl_value_t*
@jl_emptysvec = external constant %jl_value_t*
@jl_emptytuple = external constant %jl_value_t*
@jl_diverror_exception = external constant %jl_value_t*
@jl_undefref_exception = external constant %jl_value_t*
@jl_RTLD_DEFAULT_handle = external constant i8*
@jl_world_counter = external global i64

; Function Attrs: noreturn
declare void @__stack_chk_fail() #0

; Function Attrs: noreturn
declare void @jl_error(i8*) #0

; Function Attrs: noreturn
declare void @jl_throw(%jl_value_t addrspace(12)*) #0

; Function Attrs: noreturn
declare void @jl_undefined_var_error(%jl_value_t addrspace(12)*) #0

; Function Attrs: noreturn
declare void @jl_bounds_error_ints(%jl_value_t addrspace(12)*, i64*, i64) #0

; Function Attrs: noreturn
declare void @jl_bounds_error_int(%jl_value_t addrspace(12)*, i64) #0

; Function Attrs: noreturn
declare void @jl_bounds_error_tuple_int(%jl_value_t addrspace(10)**, i64, i64) #0

; Function Attrs: noreturn
declare void @jl_bounds_error_unboxed_int(i8 addrspace(11)*, %jl_value_t*, i64) #0

declare nonnull %jl_value_t addrspace(10)* @jl_new_structv(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_new_structt(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*) #1

; Function Attrs: returns_twice
declare i32 @__sigsetjmp(i8*, i32) #2

; Function Attrs: argmemonly nounwind readonly
declare i32 @memcmp(i8*, i8*, i64) #3

; Function Attrs: noreturn
declare void @jl_type_error(i8*, %jl_value_t addrspace(10)*, %jl_value_t addrspace(12)*) #0

declare void @jl_checked_assignment(%jl_value_t*, %jl_value_t addrspace(12)*)

declare void @jl_declare_constant(%jl_value_t*)

declare %jl_value_t* @jl_get_binding_or_error(%jl_value_t*, %jl_value_t*)

declare i32 @jl_boundp(%jl_value_t*, %jl_value_t*)

declare nonnull %jl_value_t addrspace(10)* @jl_f_is(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_typeof(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_sizeof(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_issubtype(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_isa(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_typeassert(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_ifelse(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f__apply(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f__apply_iterate(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f__apply_pure(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f__apply_latest(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_throw(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_tuple(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_svec(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_applicable(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_invoke(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_invoke_kwsorter(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_isdefined(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_getfield(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_setfield(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_fieldtype(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_nfields(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f__expr(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f__typevar(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_arrayref(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_const_arrayref(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_arrayset(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_arraysize(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_f_apply_type(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_apply_generic(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1

declare nonnull %jl_value_t addrspace(10)* @jl_invoke(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** nocapture readonly, i32, %jl_value_t addrspace(10)*)

; Function Attrs: nounwind readnone
declare i1 @llvm.expect.i1(i1, i1) #4

declare nonnull %jl_value_t* @jl_toplevel_eval(%jl_value_t*, %jl_value_t*)

declare nonnull %jl_value_t addrspace(10)* @jl_copy_ast(%jl_value_t addrspace(10)*)

declare nonnull %jl_value_t* @jl_svec(i64, ...)

declare void @jl_method_def(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*, %jl_value_t*)

declare %jl_value_t addrspace(10)* @jl_generic_function_def(%jl_value_t*, %jl_value_t*, %jl_value_t addrspace(10)**, %jl_value_t*, %jl_value_t*)

declare void @jl_enter_handler(i8*)

declare %jl_value_t addrspace(10)* @jl_current_exception()

declare void @jl_pop_handler(i32)

declare void @jl_restore_excstack(i64)

declare i64 @jl_excstack_state()

declare i32 @jl_egal(%jl_value_t addrspace(12)*, %jl_value_t addrspace(12)*)

declare i32 @jl_isa(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare i32 @jl_subtype(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare void @jl_typeassert(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare nonnull %jl_value_t addrspace(10)* @jl_instantiate_type_in_env(%jl_value_t*, %jl_value_t*, %jl_value_t addrspace(10)**)

declare i64 @jl_object_id_(%jl_value_t addrspace(10)*, i8 addrspace(11)*)

declare void ()* @jl_load_and_lookup(i8*, i8*, i8**)

declare nonnull %jl_value_t addrspace(10)* @jl_get_cfunction_trampoline(%jl_value_t addrspace(10)*, %jl_value_t*, i8*, %jl_value_t*, i8* (i8*, %jl_value_t**)*, %jl_value_t*, %jl_value_t addrspace(10)**)

declare nonnull %jl_value_t addrspace(10)* @jl_get_nth_field_checked(%jl_value_t addrspace(10)*, i64)

declare i64 @jl_gc_diff_total_bytes()

declare i64 @jl_gc_sync_total_bytes(i64)

; Function Attrs: nounwind readonly
declare nonnull %jl_value_t addrspace(10)* @jl_array_data_owner(%jl_value_t addrspace(10)*) #5

declare %jl_value_t* @jl_box_int8(i8 signext)

declare %jl_value_t* @jl_box_uint8(i8 zeroext)

declare %jl_value_t addrspace(10)* @jl_box_int16(i16 signext)

declare %jl_value_t addrspace(10)* @jl_box_uint16(i16 zeroext)

declare %jl_value_t addrspace(10)* @jl_box_int32(i32 signext)

declare %jl_value_t addrspace(10)* @jl_box_uint32(i32 zeroext)

declare %jl_value_t addrspace(10)* @jl_box_int64(i64 signext)

declare %jl_value_t addrspace(10)* @jl_box_uint64(i64 zeroext)

declare %jl_value_t addrspace(10)* @jl_box_float32(float)

declare %jl_value_t addrspace(10)* @jl_box_float64(double)

declare %jl_value_t addrspace(10)* @jl_box_char(i32 zeroext)

declare %jl_value_t addrspace(10)* @jl_box_ssavalue(i64 zeroext)

declare %jl_value_t addrspace(10)* @jl_bitcast(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_neg_int(%jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_add_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_sub_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_mul_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_sdiv_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_udiv_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_srem_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_urem_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_add_ptr(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_sub_ptr(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_neg_float(%jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_add_float(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_sub_float(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_mul_float(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_div_float(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_rem_float(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_fma_float(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_muladd_float(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_eq_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_ne_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_slt_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_ult_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_sle_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_ule_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_eq_float(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_ne_float(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_lt_float(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_le_float(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_fpiseq(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_fpislt(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_and_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_or_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_xor_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_not_int(%jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_shl_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_lshr_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_ashr_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_bswap_int(%jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_ctpop_int(%jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_ctlz_int(%jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_cttz_int(%jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_sext_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_zext_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_trunc_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_fptoui(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_fptosi(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_uitofp(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_sitofp(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_fptrunc(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_fpext(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_checked_sadd_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_checked_uadd_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_checked_ssub_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_checked_usub_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_checked_smul_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_checked_umul_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_checked_sdiv_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_checked_udiv_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_checked_srem_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_checked_urem_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_abs_float(%jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_copysign_float(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_flipsign_int(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_ceil_llvm(%jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_floor_llvm(%jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_trunc_llvm(%jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_rint_llvm(%jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_sqrt_llvm(%jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_sqrt_llvm_fast(%jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_pointerref(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_pointerset(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_cglobal(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_arraylen(%jl_value_t addrspace(10)*)

declare %jl_value_t addrspace(10)* @jl_cglobal_auto(%jl_value_t addrspace(10)*)

define internal i64 @julia_factorial_lookup_985(i64, %jl_value_t addrspace(10)* nonnull align 16 dereferenceable(40), i64) {
top:
  %3 = alloca [4 x %jl_value_t addrspace(10)*], align 8
  %gcframe4 = alloca [4 x %jl_value_t addrspace(10)*], align 16
  %gcframe4.sub = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %gcframe4, i64 0, i64 0
  %.sub = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %3, i64 0, i64 0
  %4 = bitcast [4 x %jl_value_t addrspace(10)*]* %gcframe4 to i8*
  call void @llvm.memset.p0i8.i32(i8* nonnull align 16 %4, i8 0, i32 32, i1 false), !tbaa !2
  %thread_ptr = call i8* asm "movq %fs:0, $0", "=r"() #15
  %ptls_i8 = getelementptr i8, i8* %thread_ptr, i64 -15712
  %5 = bitcast [4 x %jl_value_t addrspace(10)*]* %gcframe4 to i64*
  store i64 8, i64* %5, align 16, !tbaa !2
  %6 = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %gcframe4, i64 0, i64 1
  %7 = bitcast i8* %ptls_i8 to i64*
  %8 = load i64, i64* %7, align 8
  %9 = bitcast %jl_value_t addrspace(10)** %6 to i64*
  store i64 %8, i64* %9, align 8, !tbaa !2
  %10 = bitcast i8* %ptls_i8 to %jl_value_t addrspace(10)***
  store %jl_value_t addrspace(10)** %gcframe4.sub, %jl_value_t addrspace(10)*** %10, align 8
  %11 = icmp sgt i64 %0, -1
  br i1 %11, label %L6, label %L3

L3:                                               ; preds = %top
  %12 = call %jl_value_t addrspace(10)* @jl_box_int64(i64 signext %0)
  %13 = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %gcframe4, i64 0, i64 2
  store %jl_value_t addrspace(10)* %12, %jl_value_t addrspace(10)** %13, align 16
  store %jl_value_t addrspace(10)* %12, %jl_value_t addrspace(10)** %.sub, align 8
  %14 = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %3, i64 0, i64 1
  store %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628507237488 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)** %14, align 8
  %15 = call nonnull %jl_value_t addrspace(10)* @jl_invoke(%jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628444826512 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)** nonnull %.sub, i32 2, %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628444826000 to %jl_value_t*) to %jl_value_t addrspace(10)*))
  %16 = addrspacecast %jl_value_t addrspace(10)* %15 to %jl_value_t addrspace(12)*
  %17 = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %gcframe4, i64 0, i64 2
  store %jl_value_t addrspace(10)* %15, %jl_value_t addrspace(10)** %17, align 16
  call void @jl_throw(%jl_value_t addrspace(12)* %16)
  unreachable

L6:                                               ; preds = %top
  %18 = icmp slt i64 %2, %0
  br i1 %18, label %L8, label %L12

L8:                                               ; preds = %L6
  %19 = call %jl_value_t addrspace(10)* @jl_box_int64(i64 signext %0)
  %20 = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %gcframe4, i64 0, i64 3
  store %jl_value_t addrspace(10)* %19, %jl_value_t addrspace(10)** %20, align 8
  %21 = call %jl_value_t addrspace(10)* @jl_box_int64(i64 signext %0)
  %22 = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %gcframe4, i64 0, i64 2
  store %jl_value_t addrspace(10)* %21, %jl_value_t addrspace(10)** %22, align 16
  store %jl_value_t addrspace(10)* %19, %jl_value_t addrspace(10)** %.sub, align 8
  %23 = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %3, i64 0, i64 1
  store %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628507237536 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)** %23, align 8
  %24 = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %3, i64 0, i64 2
  store %jl_value_t addrspace(10)* %21, %jl_value_t addrspace(10)** %24, align 8
  %25 = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %3, i64 0, i64 3
  store %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628507237632 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)** %25, align 8
  %26 = call nonnull %jl_value_t addrspace(10)* @japi1_print_to_string_987(%jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628499267728 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)** nonnull %.sub, i32 4)
  %27 = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %gcframe4, i64 0, i64 2
  store %jl_value_t addrspace(10)* %26, %jl_value_t addrspace(10)** %27, align 16
  %28 = call noalias nonnull %jl_value_t addrspace(10)* @jl_gc_pool_alloc(i8* %ptls_i8, i32 1400, i32 16) #7
  %29 = bitcast %jl_value_t addrspace(10)* %28 to %jl_value_t addrspace(10)* addrspace(10)*
  %30 = getelementptr %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)* addrspace(10)* %29, i64 -1
  store %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628446314480 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)* addrspace(10)* %30, align 8, !tbaa !6
  %31 = bitcast %jl_value_t addrspace(10)* %28 to %jl_value_t addrspace(10)* addrspace(10)*
  store %jl_value_t addrspace(10)* %26, %jl_value_t addrspace(10)* addrspace(10)* %31, align 8, !tbaa !9
  %32 = addrspacecast %jl_value_t addrspace(10)* %28 to %jl_value_t addrspace(12)*
  %33 = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %gcframe4, i64 0, i64 2
  store %jl_value_t addrspace(10)* %28, %jl_value_t addrspace(10)** %33, align 16
  call void @jl_throw(%jl_value_t addrspace(12)* %32)
  unreachable

L12:                                              ; preds = %L6
  %34 = icmp eq i64 %0, 0
  br i1 %34, label %L14, label %L15

L14:                                              ; preds = %L12
  %35 = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %gcframe4, i64 0, i64 1
  %36 = bitcast %jl_value_t addrspace(10)** %35 to i64*
  %37 = load i64, i64* %36, align 8, !tbaa !2
  %38 = bitcast i8* %ptls_i8 to i64*
  store i64 %37, i64* %38, align 8, !tbaa !2
  ret i64 1

L15:                                              ; preds = %L12
  %39 = add i64 %0, -1
  %40 = addrspacecast %jl_value_t addrspace(10)* %1 to %jl_value_t addrspace(11)*
  %41 = bitcast %jl_value_t addrspace(11)* %40 to i64 addrspace(13)* addrspace(11)*
  %42 = load i64 addrspace(13)*, i64 addrspace(13)* addrspace(11)* %41, align 8, !tbaa !12, !nonnull !15
  %43 = getelementptr inbounds i64, i64 addrspace(13)* %42, i64 %39
  %44 = load i64, i64 addrspace(13)* %43, align 8, !tbaa !16
  %45 = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %gcframe4, i64 0, i64 1
  %46 = bitcast %jl_value_t addrspace(10)** %45 to i64*
  %47 = load i64, i64* %46, align 8, !tbaa !2
  %48 = bitcast i8* %ptls_i8 to i64*
  store i64 %47, i64* %48, align 8, !tbaa !2
  ret i64 %44
}

define internal nonnull %jl_value_t addrspace(10)* @jfptr_factorial_lookup_986(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1 {
top:
  %3 = bitcast %jl_value_t addrspace(10)** %1 to i64 addrspace(10)**
  %4 = load i64 addrspace(10)*, i64 addrspace(10)** %3, align 8, !nonnull !15, !dereferenceable !18, !align !18
  %5 = addrspacecast i64 addrspace(10)* %4 to i64 addrspace(11)*
  %6 = load i64, i64 addrspace(11)* %5, align 8
  %7 = getelementptr inbounds %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %1, i64 1
  %8 = load %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %7, align 8, !nonnull !15, !dereferenceable !19, !align !20
  %9 = getelementptr inbounds %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %1, i64 2
  %10 = bitcast %jl_value_t addrspace(10)** %9 to i64 addrspace(10)**
  %11 = load i64 addrspace(10)*, i64 addrspace(10)** %10, align 8, !nonnull !15, !dereferenceable !18, !align !18
  %12 = addrspacecast i64 addrspace(10)* %11 to i64 addrspace(11)*
  %13 = load i64, i64 addrspace(11)* %12, align 8
  %14 = call i64 @julia_factorial_lookup_985(i64 %6, %jl_value_t addrspace(10)* %8, i64 %13)
  %15 = call %jl_value_t addrspace(10)* @jl_box_int64(i64 signext %14)
  ret %jl_value_t addrspace(10)* %15
}

; Function Attrs: cold noreturn nounwind
declare void @llvm.trap() #6

; Function Attrs: allocsize(1)
declare noalias nonnull %jl_value_t addrspace(10)* @julia.gc_alloc_obj(i8*, i64, %jl_value_t addrspace(10)*) #7

define internal double @julia_besselj_983(i64, double, double) {
top:
  %3 = fadd double %1, 0.000000e+00; sitofp i64 %1 to double
  %4 = fmul double %3, 5.000000e-01
  %5 = sitofp i64 %0 to double
  %6 = call double @llvm.pow.f64(double %4, double %5)
  %7 = call i64 @julia_factorial_lookup_985(i64 %0, %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628506941520 to %jl_value_t*) to %jl_value_t addrspace(10)*), i64 20)
  %8 = sitofp i64 %7 to double
  %9 = fdiv double %6, %8
  %10 = call double @llvm.fabs.f64(double %9)
  %11 = fcmp ule double %10, %2
  br i1 %11, label %L27, label %L14.lr.ph

L14.lr.ph:                                        ; preds = %top
  %12 = fmul double %4, %4
  br label %L14

L14:                                              ; preds = %L14.lr.ph, %L14
  %value_phi25 = phi double [ %9, %L14.lr.ph ], [ %21, %L14 ]
  %value_phi14 = phi double [ %9, %L14.lr.ph ], [ %20, %L14 ]
  %value_phi3 = phi i64 [ 0, %L14.lr.ph ], [ %13, %L14 ]
  %13 = add i64 %value_phi3, 1
  %14 = sitofp i64 %13 to double
  %15 = fdiv double -1.000000e+00, %14
  %16 = add i64 %13, %0
  %17 = sitofp i64 %16 to double
  %18 = fdiv double %15, %17
  %19 = fmul double %12, %18
  %20 = fmul double %value_phi14, %19
  %21 = fadd double %value_phi25, %20
  %22 = call double @llvm.fabs.f64(double %20)
  %23 = fcmp ule double %22, %2
  br i1 %23, label %L27, label %L14

L27:                                              ; preds = %L14, %top
  %value_phi2.lcssa = phi double [ %9, %top ], [ %21, %L14 ]
  ret double %value_phi2.lcssa
}

define internal nonnull %jl_value_t addrspace(10)* @jfptr_besselj_984(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1 {
top:
  %thread_ptr = call i8* asm "movq %fs:0, $0", "=r"() #15
  %ptls_i8 = getelementptr i8, i8* %thread_ptr, i64 -15712
  %3 = bitcast %jl_value_t addrspace(10)** %1 to i64 addrspace(10)**
  %4 = load i64 addrspace(10)*, i64 addrspace(10)** %3, align 8, !nonnull !15, !dereferenceable !18, !align !18
  %5 = addrspacecast i64 addrspace(10)* %4 to i64 addrspace(11)*
  %6 = load i64, i64 addrspace(11)* %5, align 8
  %7 = getelementptr inbounds %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %1, i64 1
  %8 = bitcast %jl_value_t addrspace(10)** %7 to i64 addrspace(10)**
  %9 = load i64 addrspace(10)*, i64 addrspace(10)** %8, align 8, !nonnull !15, !dereferenceable !18, !align !18
  %10 = addrspacecast i64 addrspace(10)* %9 to i64 addrspace(11)*
  %11 = load i64, i64 addrspace(11)* %10, align 8
  %12 = getelementptr inbounds %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %1, i64 2
  %13 = bitcast %jl_value_t addrspace(10)** %12 to double addrspace(10)**
  %14 = load double addrspace(10)*, double addrspace(10)** %13, align 8, !nonnull !15, !dereferenceable !18, !align !18
  %15 = addrspacecast double addrspace(10)* %14 to double addrspace(11)*
  %16 = load double, double addrspace(11)* %15, align 8
  %fp = sitofp i64 %11 to double
  %17 = call double @julia_besselj_983(i64 %6, double %fp, double %16)
  %18 = call noalias nonnull %jl_value_t addrspace(10)* @jl_gc_pool_alloc(i8* %ptls_i8, i32 1400, i32 16) #7
  %19 = bitcast %jl_value_t addrspace(10)* %18 to %jl_value_t addrspace(10)* addrspace(10)*
  %20 = getelementptr %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)* addrspace(10)* %19, i64 -1
  store %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628441196864 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)* addrspace(10)* %20, align 8, !tbaa !6
  %21 = bitcast %jl_value_t addrspace(10)* %18 to double addrspace(10)*
  store double %17, double addrspace(10)* %21, align 8, !tbaa !9
  ret %jl_value_t addrspace(10)* %18
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.pow.f64(double, double) #8

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #8

define dso_local double @julia_besselj0_980(double) {
top:
  %1 = call double @julia_besselj_983(i64 0, double %0, double 1.000000e-08)
  ret double %1
}

define internal nonnull %jl_value_t addrspace(10)* @jfptr_besselj0_981(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1 {
top:
  %thread_ptr = call i8* asm "movq %fs:0, $0", "=r"() #15
  %ptls_i8 = getelementptr i8, i8* %thread_ptr, i64 -15712
  %3 = bitcast %jl_value_t addrspace(10)** %1 to i64 addrspace(10)**
  %4 = load i64 addrspace(10)*, i64 addrspace(10)** %3, align 8, !nonnull !15, !dereferenceable !18, !align !18
  %5 = addrspacecast i64 addrspace(10)* %4 to i64 addrspace(11)*
  %6 = load i64, i64 addrspace(11)* %5, align 8
  %cst = sitofp i64 %6 to double
  %7 = call double @julia_besselj0_980(double %cst)
  %8 = call noalias nonnull %jl_value_t addrspace(10)* @jl_gc_pool_alloc(i8* %ptls_i8, i32 1400, i32 16) #7
  %9 = bitcast %jl_value_t addrspace(10)* %8 to %jl_value_t addrspace(10)* addrspace(10)*
  %10 = getelementptr %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)* addrspace(10)* %9, i64 -1
  store %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628441196864 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)* addrspace(10)* %10, align 8, !tbaa !6
  %11 = bitcast %jl_value_t addrspace(10)* %8 to double addrspace(10)*
  store double %7, double addrspace(10)* %11, align 8, !tbaa !9
  ret %jl_value_t addrspace(10)* %8
}

define internal nonnull %jl_value_t addrspace(10)* @japi1_print_to_string_987(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1 {
top:
  %3 = alloca [2 x %jl_value_t addrspace(10)*], align 8
  %gcframe49 = alloca [3 x %jl_value_t addrspace(10)*], align 16
  %gcframe49.sub = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %gcframe49, i64 0, i64 0
  %.sub = getelementptr inbounds [2 x %jl_value_t addrspace(10)*], [2 x %jl_value_t addrspace(10)*]* %3, i64 0, i64 0
  %4 = bitcast [3 x %jl_value_t addrspace(10)*]* %gcframe49 to i8*
  call void @llvm.memset.p0i8.i32(i8* nonnull align 16 %4, i8 0, i32 24, i1 false), !tbaa !2
  %5 = alloca %jl_value_t addrspace(10)**, align 8
  store volatile %jl_value_t addrspace(10)** %1, %jl_value_t addrspace(10)*** %5, align 8
  %thread_ptr = call i8* asm "movq %fs:0, $0", "=r"() #15
  %ptls_i8 = getelementptr i8, i8* %thread_ptr, i64 -15712
  %6 = bitcast [3 x %jl_value_t addrspace(10)*]* %gcframe49 to i64*
  store i64 4, i64* %6, align 16, !tbaa !2
  %7 = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %gcframe49, i64 0, i64 1
  %8 = bitcast i8* %ptls_i8 to i64*
  %9 = load i64, i64* %8, align 8
  %10 = bitcast %jl_value_t addrspace(10)** %7 to i64*
  store i64 %9, i64* %10, align 8, !tbaa !2
  %11 = bitcast i8* %ptls_i8 to %jl_value_t addrspace(10)***
  store %jl_value_t addrspace(10)** %gcframe49.sub, %jl_value_t addrspace(10)*** %11, align 8
  %12 = icmp slt i32 %2, 1
  br i1 %12, label %L114.thread, label %L18

L114.thread:                                      ; preds = %top
  %13 = call nonnull %jl_value_t addrspace(10)* @julia_YY.IOBufferYY.328_993(i8 1, i8 1, i8 1, i64 9223372036854775807, i64 0, %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628440747648 to %jl_value_t*) to %jl_value_t addrspace(10)*))
  br label %L155

L18:                                              ; preds = %top
  %14 = sext i32 %2 to i64
  br label %L20

L20:                                              ; preds = %L92, %L18
  %value_phi4 = phi i64 [ 0, %L18 ], [ %38, %L92 ]
  %value_phi5.in = phi %jl_value_t addrspace(10)** [ %1, %L18 ], [ %41, %L92 ]
  %value_phi6 = phi i64 [ 2, %L18 ], [ %42, %L92 ]
  %value_phi5 = load %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %value_phi5.in, align 8, !tbaa !21, !nonnull !15
  %15 = bitcast %jl_value_t addrspace(10)* %value_phi5 to i64 addrspace(10)*
  %16 = getelementptr i64, i64 addrspace(10)* %15, i64 -1
  %17 = load i64, i64 addrspace(10)* %16, align 8, !tbaa !6, !range !22
  %18 = and i64 %17, -16
  %19 = inttoptr i64 %18 to %jl_value_t*
  %20 = addrspacecast %jl_value_t* %19 to %jl_value_t addrspace(10)*
  %21 = icmp eq %jl_value_t addrspace(10)* %20, addrspacecast (%jl_value_t* inttoptr (i64 139628441196864 to %jl_value_t*) to %jl_value_t addrspace(10)*)
  br i1 %21, label %L78, label %L26

L26:                                              ; preds = %L20
  %22 = icmp eq %jl_value_t addrspace(10)* %20, addrspacecast (%jl_value_t* inttoptr (i64 139628441196512 to %jl_value_t*) to %jl_value_t addrspace(10)*)
  br i1 %22, label %L78, label %L29

L29:                                              ; preds = %L26
  %23 = icmp eq %jl_value_t addrspace(10)* %20, addrspacecast (%jl_value_t* inttoptr (i64 139628440673312 to %jl_value_t*) to %jl_value_t addrspace(10)*)
  br i1 %23, label %L38, label %L33

L33:                                              ; preds = %L29
  %24 = icmp eq %jl_value_t addrspace(10)* %20, addrspacecast (%jl_value_t* inttoptr (i64 139628441684976 to %jl_value_t*) to %jl_value_t addrspace(10)*)
  br i1 %24, label %L44, label %L52

L38:                                              ; preds = %L29
  %25 = bitcast %jl_value_t addrspace(10)* %value_phi5 to i64 addrspace(10)*
  %26 = load i64, i64 addrspace(10)* %25, align 8, !tbaa !23
  br label %L78

L44:                                              ; preds = %L33
  %27 = addrspacecast %jl_value_t addrspace(10)* %value_phi5 to %jl_value_t addrspace(11)*
  %28 = bitcast %jl_value_t addrspace(11)* %27 to { %jl_value_t addrspace(10)*, i64, i64 } addrspace(11)*
  %29 = getelementptr inbounds { %jl_value_t addrspace(10)*, i64, i64 }, { %jl_value_t addrspace(10)*, i64, i64 } addrspace(11)* %28, i64 0, i32 2
  %30 = load i64, i64 addrspace(11)* %29, align 8, !tbaa !21
  br label %L78

L52:                                              ; preds = %L33
  %31 = icmp eq %jl_value_t addrspace(10)* %20, addrspacecast (%jl_value_t* inttoptr (i64 139628441089216 to %jl_value_t*) to %jl_value_t addrspace(10)*)
  br i1 %31, label %L54, label %L78

L54:                                              ; preds = %L52
  %32 = bitcast %jl_value_t addrspace(10)* %value_phi5 to i32 addrspace(10)*
  %33 = load i32, i32 addrspace(10)* %32, align 4, !tbaa !21
  %34 = call i32 @llvm.bswap.i32(i32 %33)
  br label %L57

L57:                                              ; preds = %L57, %L54
  %value_phi31 = phi i32 [ %34, %L54 ], [ %35, %L57 ]
  %value_phi32 = phi i64 [ 1, %L54 ], [ %37, %L57 ]
  %35 = lshr i32 %value_phi31, 8
  %36 = icmp eq i32 %35, 0
  %37 = add nuw nsw i64 %value_phi32, 1
  br i1 %36, label %L78, label %L57

L78:                                              ; preds = %L57, %L52, %L26, %L20, %L38, %L44
  %value_phi7 = phi i64 [ 20, %L20 ], [ 12, %L26 ], [ %26, %L38 ], [ %30, %L44 ], [ 8, %L52 ], [ %value_phi32, %L57 ]
  %38 = add i64 %value_phi7, %value_phi4
  %39 = icmp sgt i64 %value_phi6, %14
  br i1 %39, label %L132, label %L92

L92:                                              ; preds = %L78
  %40 = add nsw i64 %value_phi6, -1
  %41 = getelementptr inbounds %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %1, i64 %40
  %42 = add nuw nsw i64 %value_phi6, 1
  br label %L20

L132:                                             ; preds = %L78
  %43 = call nonnull %jl_value_t addrspace(10)* @julia_YY.IOBufferYY.328_993(i8 1, i8 1, i8 1, i64 9223372036854775807, i64 %38, %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628440747648 to %jl_value_t*) to %jl_value_t addrspace(10)*))
  %44 = bitcast %jl_value_t addrspace(10)** %1 to i64*
  %value_phi184650 = load i64, i64* %44, align 8, !tbaa !21, !range !25
  %45 = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %gcframe49, i64 0, i64 2
  store %jl_value_t addrspace(10)* %43, %jl_value_t addrspace(10)** %45, align 16
  store %jl_value_t addrspace(10)* %43, %jl_value_t addrspace(10)** %.sub, align 8
  %46 = getelementptr inbounds [2 x %jl_value_t addrspace(10)*], [2 x %jl_value_t addrspace(10)*]* %3, i64 0, i64 1
  %47 = bitcast %jl_value_t addrspace(10)** %46 to i64*
  store i64 %value_phi184650, i64* %47, align 8
  %48 = call nonnull %jl_value_t addrspace(10)* @jl_apply_generic(%jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628455362944 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)** nonnull %.sub, i32 2)
  %49 = icmp sgt i32 %2, 1
  br i1 %49, label %L149, label %L155

L149:                                             ; preds = %L132, %L149
  %value_phi1947 = phi i64 [ %52, %L149 ], [ 2, %L132 ]
  %50 = add nsw i64 %value_phi1947, -1
  %51 = getelementptr inbounds %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %1, i64 %50
  %52 = add nuw nsw i64 %value_phi1947, 1
  %53 = bitcast %jl_value_t addrspace(10)** %51 to i64*
  %value_phi1851 = load i64, i64* %53, align 8, !tbaa !21, !range !25
  store %jl_value_t addrspace(10)* %43, %jl_value_t addrspace(10)** %.sub, align 8
  %54 = getelementptr inbounds [2 x %jl_value_t addrspace(10)*], [2 x %jl_value_t addrspace(10)*]* %3, i64 0, i64 1
  %55 = bitcast %jl_value_t addrspace(10)** %54 to i64*
  store i64 %value_phi1851, i64* %55, align 8
  %56 = call nonnull %jl_value_t addrspace(10)* @jl_apply_generic(%jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628455362944 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)** nonnull %.sub, i32 2)
  %57 = icmp slt i64 %value_phi1947, %14
  br i1 %57, label %L149, label %L155

L155:                                             ; preds = %L149, %L132, %L114.thread
  %58 = phi %jl_value_t addrspace(10)* [ %13, %L114.thread ], [ %43, %L132 ], [ %43, %L149 ]
  %59 = addrspacecast %jl_value_t addrspace(10)* %58 to %jl_value_t addrspace(11)*
  %60 = bitcast %jl_value_t addrspace(11)* %59 to %jl_value_t addrspace(10)* addrspace(11)*
  %61 = load %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)* addrspace(11)* %60, align 8, !tbaa !23, !nonnull !15, !dereferenceable !19, !align !20
  %62 = bitcast %jl_value_t addrspace(11)* %59 to i8 addrspace(11)*
  %63 = getelementptr inbounds i8, i8 addrspace(11)* %62, i64 16
  %64 = bitcast i8 addrspace(11)* %63 to i64 addrspace(11)*
  %65 = load i64, i64 addrspace(11)* %64, align 8, !tbaa !23
  %66 = addrspacecast %jl_value_t addrspace(10)* %61 to %jl_value_t addrspace(11)*
  %67 = bitcast %jl_value_t addrspace(11)* %66 to %jl_array_t addrspace(11)*
  %68 = getelementptr inbounds %jl_array_t, %jl_array_t addrspace(11)* %67, i64 0, i32 1
  %69 = load i64, i64 addrspace(11)* %68, align 8, !tbaa !26
  %70 = icmp slt i64 %69, %65
  br i1 %70, label %L160, label %L176

L160:                                             ; preds = %L155
  %71 = sub i64 %65, %69
  %72 = icmp sgt i64 %71, -1
  br i1 %72, label %L173, label %L165

L165:                                             ; preds = %L160
  %73 = call nonnull %jl_value_t addrspace(10)* @julia_throw_inexacterror_992(%jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628360224184 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628440672960 to %jl_value_t*) to %jl_value_t addrspace(10)*), i64 %71)
  unreachable

L173:                                             ; preds = %L160
  %74 = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %gcframe49, i64 0, i64 2
  store %jl_value_t addrspace(10)* %61, %jl_value_t addrspace(10)** %74, align 16
  call void inttoptr (i64 139628664694672 to void (%jl_value_t addrspace(10)*, i64)*)(%jl_value_t addrspace(10)* nonnull %61, i64 %71)
  br label %L201

L176:                                             ; preds = %L155
  %75 = icmp eq i64 %65, %69
  br i1 %75, label %L201, label %L179

L179:                                             ; preds = %L176
  %76 = icmp sgt i64 %65, -1
  br i1 %76, label %L184, label %L181

L181:                                             ; preds = %L179
  %77 = call noalias nonnull %jl_value_t addrspace(10)* @jl_gc_pool_alloc(i8* %ptls_i8, i32 1400, i32 16) #7
  %78 = bitcast %jl_value_t addrspace(10)* %77 to %jl_value_t addrspace(10)* addrspace(10)*
  %79 = getelementptr %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)* addrspace(10)* %78, i64 -1
  store %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628441319696 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)* addrspace(10)* %79, align 8, !tbaa !6
  %80 = bitcast %jl_value_t addrspace(10)* %77 to %jl_value_t addrspace(10)* addrspace(10)*
  store %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628447135904 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)* addrspace(10)* %80, align 8, !tbaa !9
  %81 = addrspacecast %jl_value_t addrspace(10)* %77 to %jl_value_t addrspace(12)*
  %82 = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %gcframe49, i64 0, i64 2
  store %jl_value_t addrspace(10)* %77, %jl_value_t addrspace(10)** %82, align 16
  call void @jl_throw(%jl_value_t addrspace(12)* %81)
  unreachable

L184:                                             ; preds = %L179
  %83 = sub i64 %69, %65
  %84 = icmp sgt i64 %83, -1
  br i1 %84, label %L197, label %L189

L189:                                             ; preds = %L184
  %85 = call nonnull %jl_value_t addrspace(10)* @julia_throw_inexacterror_992(%jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628360224184 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628440672960 to %jl_value_t*) to %jl_value_t addrspace(10)*), i64 %83)
  unreachable

L197:                                             ; preds = %L184
  %86 = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %gcframe49, i64 0, i64 2
  store %jl_value_t addrspace(10)* %61, %jl_value_t addrspace(10)** %86, align 16
  call void inttoptr (i64 139628664698000 to void (%jl_value_t addrspace(10)*, i64)*)(%jl_value_t addrspace(10)* nonnull %61, i64 %83)
  br label %L201

L201:                                             ; preds = %L173, %L176, %L197
  %87 = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %gcframe49, i64 0, i64 2
  store %jl_value_t addrspace(10)* %61, %jl_value_t addrspace(10)** %87, align 16
  %88 = call %jl_value_t addrspace(10)* inttoptr (i64 139628664689008 to %jl_value_t addrspace(10)* (%jl_value_t addrspace(10)*)*)(%jl_value_t addrspace(10)* nonnull %61)
  %89 = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %gcframe49, i64 0, i64 1
  %90 = bitcast %jl_value_t addrspace(10)** %89 to i64*
  %91 = load i64, i64* %90, align 8, !tbaa !2
  %92 = bitcast i8* %ptls_i8 to i64*
  store i64 %91, i64* %92, align 8, !tbaa !2
  ret %jl_value_t addrspace(10)* %88
}

; Function Attrs: argmemonly norecurse nounwind readonly
declare nonnull %jl_value_t addrspace(10)* @julia.typeof(%jl_value_t addrspace(10)*) #9

declare nonnull %jl_value_t addrspace(10)* @j_throw_inexacterror_990(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*, i64)

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.bswap.i32(i32) #8

define internal nonnull %jl_value_t addrspace(10)* @julia_getindex_999(%jl_value_t addrspace(10)* nonnull, i64) {
top:
  %2 = alloca [3 x %jl_value_t addrspace(10)*], align 8
  %gcframe2 = alloca [3 x %jl_value_t addrspace(10)*], align 16
  %gcframe2.sub = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %gcframe2, i64 0, i64 0
  %.sub = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %2, i64 0, i64 0
  %3 = bitcast [3 x %jl_value_t addrspace(10)*]* %gcframe2 to i8*
  call void @llvm.memset.p0i8.i32(i8* nonnull align 16 %3, i8 0, i32 24, i1 false), !tbaa !2
  %thread_ptr = call i8* asm "movq %fs:0, $0", "=r"() #15
  %ptls_i8 = getelementptr i8, i8* %thread_ptr, i64 -15712
  %4 = bitcast [3 x %jl_value_t addrspace(10)*]* %gcframe2 to i64*
  store i64 4, i64* %4, align 16, !tbaa !2
  %5 = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %gcframe2, i64 0, i64 1
  %6 = bitcast i8* %ptls_i8 to i64*
  %7 = load i64, i64* %6, align 8
  %8 = bitcast %jl_value_t addrspace(10)** %5 to i64*
  store i64 %7, i64* %8, align 8, !tbaa !2
  %9 = bitcast i8* %ptls_i8 to %jl_value_t addrspace(10)***
  store %jl_value_t addrspace(10)** %gcframe2.sub, %jl_value_t addrspace(10)*** %9, align 8
  %10 = call %jl_value_t addrspace(10)* @jl_box_int64(i64 signext %1)
  %11 = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %gcframe2, i64 0, i64 2
  store %jl_value_t addrspace(10)* %10, %jl_value_t addrspace(10)** %11, align 16
  store %jl_value_t addrspace(10)* %0, %jl_value_t addrspace(10)** %.sub, align 8
  %12 = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %2, i64 0, i64 1
  store %jl_value_t addrspace(10)* %10, %jl_value_t addrspace(10)** %12, align 8
  %13 = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %2, i64 0, i64 2
  store %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628442281184 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)** %13, align 8
  %14 = call nonnull %jl_value_t addrspace(10)* @jl_f_getfield(%jl_value_t addrspace(10)* addrspacecast (%jl_value_t* null to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)** nonnull %.sub, i32 3)
  %15 = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %gcframe2, i64 0, i64 1
  %16 = bitcast %jl_value_t addrspace(10)** %15 to i64*
  %17 = load i64, i64* %16, align 8, !tbaa !2
  %18 = bitcast i8* %ptls_i8 to i64*
  store i64 %17, i64* %18, align 8, !tbaa !2
  ret %jl_value_t addrspace(10)* %14
}

define internal nonnull %jl_value_t addrspace(10)* @jfptr_getindex_1000(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1 {
top:
  %3 = load %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %1, align 8, !nonnull !15
  %4 = getelementptr inbounds %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %1, i64 1
  %5 = bitcast %jl_value_t addrspace(10)** %4 to i64 addrspace(10)**
  %6 = load i64 addrspace(10)*, i64 addrspace(10)** %5, align 8, !nonnull !15, !dereferenceable !18, !align !18
  %7 = addrspacecast i64 addrspace(10)* %6 to i64 addrspace(11)*
  %8 = load i64, i64 addrspace(11)* %7, align 8
  %9 = call nonnull %jl_value_t addrspace(10)* @julia_getindex_999(%jl_value_t addrspace(10)* %3, i64 %8)
  ret %jl_value_t addrspace(10)* %9
}

define internal nonnull %jl_value_t addrspace(10)* @julia_YY.IOBufferYY.328_993(i8, i8, i8, i64, i64, %jl_value_t addrspace(10)*) {
top:
  %gcframe30 = alloca [4 x %jl_value_t addrspace(10)*], align 16
  %gcframe30.sub = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %gcframe30, i64 0, i64 0
  %6 = bitcast [4 x %jl_value_t addrspace(10)*]* %gcframe30 to i8*
  call void @llvm.memset.p0i8.i32(i8* nonnull align 16 %6, i8 0, i32 32, i1 false), !tbaa !2
  %thread_ptr = call i8* asm "movq %fs:0, $0", "=r"() #15
  %ptls_i8 = getelementptr i8, i8* %thread_ptr, i64 -15712
  %7 = bitcast [4 x %jl_value_t addrspace(10)*]* %gcframe30 to i64*
  store i64 8, i64* %7, align 16, !tbaa !2
  %8 = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %gcframe30, i64 0, i64 1
  %9 = bitcast i8* %ptls_i8 to i64*
  %10 = load i64, i64* %9, align 8
  %11 = bitcast %jl_value_t addrspace(10)** %8 to i64*
  store i64 %10, i64* %11, align 8, !tbaa !2
  %12 = bitcast i8* %ptls_i8 to %jl_value_t addrspace(10)***
  store %jl_value_t addrspace(10)** %gcframe30.sub, %jl_value_t addrspace(10)*** %12, align 8
  %13 = icmp sgt i64 %4, -1
  br i1 %13, label %L65, label %L57

L57:                                              ; preds = %top
  %14 = call nonnull %jl_value_t addrspace(10)* @julia_throw_inexacterror_992(%jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628360224184 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628440672960 to %jl_value_t*) to %jl_value_t addrspace(10)*), i64 %4)
  unreachable

L65:                                              ; preds = %top
  %15 = call %jl_value_t addrspace(10)* inttoptr (i64 139628664689216 to %jl_value_t addrspace(10)* (i64)*)(i64 %4)
  %16 = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %gcframe30, i64 0, i64 2
  store %jl_value_t addrspace(10)* %15, %jl_value_t addrspace(10)** %16, align 16
  %17 = call %jl_value_t addrspace(10)* inttoptr (i64 139628664683776 to %jl_value_t addrspace(10)* (%jl_value_t addrspace(10)*)*)(%jl_value_t addrspace(10)* %15)
  %18 = icmp sgt i64 %3, -1
  br i1 %18, label %L100, label %L84

L84:                                              ; preds = %L65
  %19 = call noalias nonnull %jl_value_t addrspace(10)* @jl_gc_pool_alloc(i8* %ptls_i8, i32 1400, i32 16) #7
  %20 = bitcast %jl_value_t addrspace(10)* %19 to %jl_value_t addrspace(10)* addrspace(10)*
  %21 = getelementptr %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)* addrspace(10)* %20, i64 -1
  store %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628441319696 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)* addrspace(10)* %21, align 8, !tbaa !6
  %22 = bitcast %jl_value_t addrspace(10)* %19 to %jl_value_t addrspace(10)* addrspace(10)*
  store %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628458017232 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)* addrspace(10)* %22, align 8, !tbaa !9
  %23 = addrspacecast %jl_value_t addrspace(10)* %19 to %jl_value_t addrspace(12)*
  %24 = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %gcframe30, i64 0, i64 2
  store %jl_value_t addrspace(10)* %19, %jl_value_t addrspace(10)** %24, align 16
  call void @jl_throw(%jl_value_t addrspace(12)* %23)
  unreachable

L100:                                             ; preds = %L65
  %25 = addrspacecast %jl_value_t addrspace(10)* %17 to %jl_value_t addrspace(11)*
  %26 = bitcast %jl_value_t addrspace(11)* %25 to %jl_array_t addrspace(11)*
  %27 = getelementptr inbounds %jl_array_t, %jl_array_t addrspace(11)* %26, i64 0, i32 1
  %28 = load i64, i64 addrspace(11)* %27, align 8, !tbaa !26
  %29 = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %gcframe30, i64 0, i64 3
  store %jl_value_t addrspace(10)* %17, %jl_value_t addrspace(10)** %29, align 8
  %30 = call noalias nonnull %jl_value_t addrspace(10)* @jl_gc_pool_alloc(i8* %ptls_i8, i32 1472, i32 64) #7
  %31 = bitcast %jl_value_t addrspace(10)* %30 to %jl_value_t addrspace(10)* addrspace(10)*
  %32 = getelementptr %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)* addrspace(10)* %31, i64 -1
  store %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628440747648 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)* addrspace(10)* %32, align 8, !tbaa !6
  %33 = addrspacecast %jl_value_t addrspace(10)* %30 to %jl_value_t addrspace(11)*
  %34 = bitcast %jl_value_t addrspace(10)* %30 to %jl_value_t addrspace(10)* addrspace(10)*
  store %jl_value_t addrspace(10)* %17, %jl_value_t addrspace(10)* addrspace(10)* %34, align 8, !tbaa !23
  %35 = bitcast %jl_value_t addrspace(11)* %33 to i8 addrspace(11)*
  %36 = getelementptr inbounds i8, i8 addrspace(11)* %35, i64 8
  store i8 %0, i8 addrspace(11)* %36, align 8, !tbaa !23
  %37 = getelementptr inbounds i8, i8 addrspace(11)* %35, i64 9
  store i8 %1, i8 addrspace(11)* %37, align 1, !tbaa !23
  %38 = getelementptr inbounds i8, i8 addrspace(11)* %35, i64 10
  store i8 1, i8 addrspace(11)* %38, align 2, !tbaa !23
  %39 = getelementptr inbounds i8, i8 addrspace(11)* %35, i64 11
  store i8 0, i8 addrspace(11)* %39, align 1, !tbaa !23
  %40 = getelementptr inbounds i8, i8 addrspace(11)* %35, i64 16
  %41 = bitcast i8 addrspace(11)* %40 to i64 addrspace(11)*
  store i64 %28, i64 addrspace(11)* %41, align 8, !tbaa !23
  %42 = getelementptr inbounds i8, i8 addrspace(11)* %35, i64 24
  %43 = bitcast i8 addrspace(11)* %42 to i64 addrspace(11)*
  store i64 %3, i64 addrspace(11)* %43, align 8, !tbaa !23
  %44 = getelementptr inbounds i8, i8 addrspace(11)* %35, i64 32
  %45 = bitcast i8 addrspace(11)* %44 to i64 addrspace(11)*
  store i64 1, i64 addrspace(11)* %45, align 8, !tbaa !23
  %46 = getelementptr inbounds i8, i8 addrspace(11)* %35, i64 40
  %47 = bitcast i8 addrspace(11)* %46 to i64 addrspace(11)*
  store i64 -1, i64 addrspace(11)* %47, align 8, !tbaa !23
  %48 = and i8 %2, 1
  %49 = icmp eq i8 %48, 0
  br i1 %49, label %L141, label %L137

L137:                                             ; preds = %L100
  store i64 0, i64 addrspace(11)* %41, align 8, !tbaa !23
  br label %L141

L141:                                             ; preds = %L137, %L100
  %50 = load i64, i64 addrspace(11)* %27, align 8, !tbaa !26
  %51 = icmp sgt i64 %50, -1
  br i1 %51, label %L155, label %L147

L147:                                             ; preds = %L141
  %52 = call nonnull %jl_value_t addrspace(10)* @julia_throw_inexacterror_992(%jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628360224184 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628440672960 to %jl_value_t*) to %jl_value_t addrspace(10)*), i64 %50)
  unreachable

L155:                                             ; preds = %L141
  %53 = addrspacecast %jl_value_t addrspace(10)* %17 to %jl_value_t*
  %54 = bitcast %jl_value_t* %53 to i64*
  %55 = load i64, i64* %54, align 8, !tbaa !12, !range !25
  %56 = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %gcframe30, i64 0, i64 2
  store %jl_value_t addrspace(10)* %30, %jl_value_t addrspace(10)** %56, align 16
  %57 = call i64 inttoptr (i64 139628654841680 to i64 (i64, i32, i64)*)(i64 %55, i32 0, i64 %50)
  %58 = getelementptr inbounds [4 x %jl_value_t addrspace(10)*], [4 x %jl_value_t addrspace(10)*]* %gcframe30, i64 0, i64 1
  %59 = bitcast %jl_value_t addrspace(10)** %58 to i64*
  %60 = load i64, i64* %59, align 8, !tbaa !2
  %61 = bitcast i8* %ptls_i8 to i64*
  store i64 %60, i64* %61, align 8, !tbaa !2
  ret %jl_value_t addrspace(10)* %30
}

define internal nonnull %jl_value_t addrspace(10)* @jfptr_YY.IOBufferYY.328_994(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1 {
top:
  %3 = bitcast %jl_value_t addrspace(10)** %1 to i8 addrspace(10)**
  %4 = load i8 addrspace(10)*, i8 addrspace(10)** %3, align 8, !nonnull !15, !dereferenceable !28
  %5 = addrspacecast i8 addrspace(10)* %4 to i8 addrspace(11)*
  %6 = load i8, i8 addrspace(11)* %5, align 1
  %7 = getelementptr inbounds %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %1, i64 1
  %8 = bitcast %jl_value_t addrspace(10)** %7 to i8 addrspace(10)**
  %9 = load i8 addrspace(10)*, i8 addrspace(10)** %8, align 8, !nonnull !15, !dereferenceable !28
  %10 = addrspacecast i8 addrspace(10)* %9 to i8 addrspace(11)*
  %11 = load i8, i8 addrspace(11)* %10, align 1
  %12 = getelementptr inbounds %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %1, i64 3
  %13 = bitcast %jl_value_t addrspace(10)** %12 to i8 addrspace(10)**
  %14 = load i8 addrspace(10)*, i8 addrspace(10)** %13, align 8, !nonnull !15, !dereferenceable !28
  %15 = addrspacecast i8 addrspace(10)* %14 to i8 addrspace(11)*
  %16 = load i8, i8 addrspace(11)* %15, align 1
  %17 = getelementptr inbounds %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %1, i64 4
  %18 = bitcast %jl_value_t addrspace(10)** %17 to i64 addrspace(10)**
  %19 = load i64 addrspace(10)*, i64 addrspace(10)** %18, align 8, !nonnull !15, !dereferenceable !18, !align !18
  %20 = addrspacecast i64 addrspace(10)* %19 to i64 addrspace(11)*
  %21 = load i64, i64 addrspace(11)* %20, align 8
  %22 = getelementptr inbounds %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %1, i64 5
  %23 = bitcast %jl_value_t addrspace(10)** %22 to i64 addrspace(10)**
  %24 = load i64 addrspace(10)*, i64 addrspace(10)** %23, align 8, !nonnull !15, !dereferenceable !18, !align !18
  %25 = addrspacecast i64 addrspace(10)* %24 to i64 addrspace(11)*
  %26 = load i64, i64 addrspace(11)* %25, align 8
  %27 = getelementptr inbounds %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %1, i64 6
  %28 = load %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %27, align 8, !nonnull !15
  %29 = call nonnull %jl_value_t addrspace(10)* @julia_YY.IOBufferYY.328_993(i8 %6, i8 %11, i8 %16, i64 %21, i64 %26, %jl_value_t addrspace(10)* nonnull %28)
  ret %jl_value_t addrspace(10)* %29
}

declare nonnull %jl_value_t addrspace(10)* @j_getindex_994(%jl_value_t addrspace(10)*, i64)

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #10

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #10

declare nonnull %jl_value_t addrspace(10)* @j_throw_inexacterror_995(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)*, i64)

declare nonnull %jl_value_t addrspace(10)* @j_getindex_996(%jl_value_t addrspace(10)*, i64)

; Function Attrs: inaccessiblememonly norecurse nounwind
declare void @julia.write_barrier(%jl_value_t addrspace(10)*, ...) #11

; Function Attrs: nounwind readnone
declare %jl_value_t* @julia.pointer_from_objref(%jl_value_t addrspace(11)*) #4

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #10

; Function Attrs: noinline noreturn
define internal nonnull %jl_value_t addrspace(10)* @julia_throw_inexacterror_992(%jl_value_t addrspace(10)* nonnull, %jl_value_t addrspace(10)*, i64) #12 {
top:
  %3 = alloca [3 x %jl_value_t addrspace(10)*], align 8
  %gcframe2 = alloca [3 x %jl_value_t addrspace(10)*], align 16
  %gcframe2.sub = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %gcframe2, i64 0, i64 0
  %.sub = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %3, i64 0, i64 0
  %4 = bitcast [3 x %jl_value_t addrspace(10)*]* %gcframe2 to i8*
  call void @llvm.memset.p0i8.i32(i8* nonnull align 16 %4, i8 0, i32 24, i1 false), !tbaa !2
  %thread_ptr = call i8* asm "movq %fs:0, $0", "=r"() #15
  %ptls_i8 = getelementptr i8, i8* %thread_ptr, i64 -15712
  %5 = bitcast [3 x %jl_value_t addrspace(10)*]* %gcframe2 to i64*
  store i64 4, i64* %5, align 16, !tbaa !2
  %6 = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %gcframe2, i64 0, i64 1
  %7 = bitcast i8* %ptls_i8 to i64*
  %8 = load i64, i64* %7, align 8
  %9 = bitcast %jl_value_t addrspace(10)** %6 to i64*
  store i64 %8, i64* %9, align 8, !tbaa !2
  %10 = bitcast i8* %ptls_i8 to %jl_value_t addrspace(10)***
  store %jl_value_t addrspace(10)** %gcframe2.sub, %jl_value_t addrspace(10)*** %10, align 8
  %11 = call %jl_value_t addrspace(10)* @jl_box_int64(i64 signext %2)
  %12 = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %gcframe2, i64 0, i64 2
  store %jl_value_t addrspace(10)* %11, %jl_value_t addrspace(10)** %12, align 16
  store %jl_value_t addrspace(10)* %0, %jl_value_t addrspace(10)** %.sub, align 8
  %13 = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %3, i64 0, i64 1
  store %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628440672960 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)** %13, align 8
  %14 = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %3, i64 0, i64 2
  store %jl_value_t addrspace(10)* %11, %jl_value_t addrspace(10)** %14, align 8
  %15 = call nonnull %jl_value_t addrspace(10)* @jl_invoke(%jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628442395280 to %jl_value_t*) to %jl_value_t addrspace(10)*), %jl_value_t addrspace(10)** nonnull %.sub, i32 3, %jl_value_t addrspace(10)* addrspacecast (%jl_value_t* inttoptr (i64 139628442394768 to %jl_value_t*) to %jl_value_t addrspace(10)*))
  %16 = addrspacecast %jl_value_t addrspace(10)* %15 to %jl_value_t addrspace(12)*
  %17 = getelementptr inbounds [3 x %jl_value_t addrspace(10)*], [3 x %jl_value_t addrspace(10)*]* %gcframe2, i64 0, i64 2
  store %jl_value_t addrspace(10)* %15, %jl_value_t addrspace(10)** %17, align 16
  call void @jl_throw(%jl_value_t addrspace(12)* %16)
  unreachable
}

define internal nonnull %jl_value_t addrspace(10)* @jfptr_throw_inexacterror_993(%jl_value_t addrspace(10)*, %jl_value_t addrspace(10)**, i32) #1 {
top:
  %3 = load %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %1, align 8, !nonnull !15
  %4 = getelementptr inbounds %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %1, i64 1
  %5 = load %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %4, align 8, !nonnull !15
  %6 = getelementptr inbounds %jl_value_t addrspace(10)*, %jl_value_t addrspace(10)** %1, i64 2
  %7 = bitcast %jl_value_t addrspace(10)** %6 to i64 addrspace(10)**
  %8 = load i64 addrspace(10)*, i64 addrspace(10)** %7, align 8, !nonnull !15, !dereferenceable !18, !align !18
  %9 = addrspacecast i64 addrspace(10)* %8 to i64 addrspace(11)*
  %10 = load i64, i64 addrspace(11)* %9, align 8
  %11 = call nonnull %jl_value_t addrspace(10)* @julia_throw_inexacterror_992(%jl_value_t addrspace(10)* %3, %jl_value_t addrspace(10)* nonnull %5, i64 %10)
  unreachable
}

; Function Attrs: alwaysinline
define double @enzyme_entry(double) #13 {
entry:
  %1 = call double (i8*, ...) @__enzyme_autodiff.Float64(i8* bitcast (double (double)* @julia_besselj0_980 to i8*), metadata !"diffe_out", double %0)
  ret double %1
}

declare double @__enzyme_autodiff.Float64(i8*, ...)

; Function Attrs: inaccessiblemem_or_argmemonly
declare void @jl_gc_queue_root(%jl_value_t addrspace(10)*) #14

; Function Attrs: allocsize(1)
declare noalias nonnull %jl_value_t addrspace(10)* @jl_gc_pool_alloc(i8*, i32, i32) #7

; Function Attrs: allocsize(1)
declare noalias nonnull %jl_value_t addrspace(10)* @jl_gc_big_alloc(i8*, i64) #7

declare noalias nonnull %jl_value_t addrspace(10)** @julia.new_gc_frame(i32)

declare void @julia.push_gc_frame(%jl_value_t addrspace(10)**, i32)

declare %jl_value_t addrspace(10)** @julia.get_gc_frame_slot(%jl_value_t addrspace(10)**, i32)

declare void @julia.pop_gc_frame(%jl_value_t addrspace(10)**)

; Function Attrs: allocsize(1)
declare noalias nonnull %jl_value_t addrspace(10)* @julia.gc_alloc_bytes(i8*, i64) #7

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i1) #10

attributes #0 = { noreturn }
attributes #1 = { "thunk" }
attributes #2 = { returns_twice }
attributes #3 = { argmemonly nounwind readonly }
attributes #4 = { nounwind readnone }
attributes #5 = { nounwind readonly }
attributes #6 = { cold noreturn nounwind }
attributes #7 = { allocsize(1) }
attributes #8 = { nounwind readnone speculatable }
attributes #9 = { argmemonly norecurse nounwind readonly }
attributes #10 = { argmemonly nounwind }
attributes #11 = { inaccessiblememonly norecurse nounwind }
attributes #12 = { noinline noreturn }
attributes #13 = { alwaysinline }
attributes #14 = { inaccessiblemem_or_argmemonly }
attributes #15 = { nounwind }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 1, !"Debug Info Version", i32 3}
!2 = !{!3, !3, i64 0}
!3 = !{!"jtbaa_gcframe", !4, i64 0}
!4 = !{!"jtbaa", !5, i64 0}
!5 = !{!"jtbaa"}
!6 = !{!7, !7, i64 0}
!7 = !{!"jtbaa_tag", !8, i64 0}
!8 = !{!"jtbaa_data", !4, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"jtbaa_immut", !11, i64 0}
!11 = !{!"jtbaa_value", !8, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"jtbaa_arrayptr", !14, i64 0}
!14 = !{!"jtbaa_array", !4, i64 0}
!15 = !{}
!16 = !{!17, !17, i64 0}
!17 = !{!"jtbaa_arraybuf", !8, i64 0}
!18 = !{i64 8}
!19 = !{i64 40}
!20 = !{i64 16}
!21 = !{!11, !11, i64 0}
!22 = !{i64 4096, i64 0}
!23 = !{!24, !24, i64 0}
!24 = !{!"jtbaa_mutab", !11, i64 0}
!25 = !{i64 1, i64 0}
!26 = !{!27, !27, i64 0}
!27 = !{!"jtbaa_arraylen", !14, i64 0}
!28 = !{i64 1}

; CHECK: { double } @diffejulia_besselj0_980
