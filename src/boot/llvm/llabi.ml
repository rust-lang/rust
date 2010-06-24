(*
 * LLVM integration with the Rust runtime.
 *)

type abi = {
  crate_ty:   Llvm.lltype;
  task_ty:    Llvm.lltype;
  word_ty:    Llvm.lltype;
  rust_start: Llvm.llvalue;
};;

let declare_abi (llctx:Llvm.llcontext) (llmod:Llvm.llmodule) : abi =
  let i32 = Llvm.i32_type llctx in

  let crate_ty =
    (* TODO: other architectures besides x86 *)
    let crate_opaque_ty = Llvm.opaque_type llctx in
    let crate_tyhandle = Llvm.handle_to_type (Llvm.struct_type llctx [|
        i32;                              (* ptrdiff_t image_base_off *)
        Llvm.pointer_type crate_opaque_ty;(* uintptr_t self_addr *)
        i32;                              (* ptrdiff_t debug_abbrev_off *)
        i32;                              (* size_t debug_abbrev_sz *)
        i32;                              (* ptrdiff_t debug_info_off *)
        i32;                              (* size_t debug_info_sz *)
        i32;                              (* size_t activate_glue_off *)
        i32;                              (* size_t main_exit_task_glue_off *)
        i32;                              (* size_t unwind_glue_off *)
        i32;                              (* size_t yield_glue_off *)
        i32;                              (* int n_rust_syms *)
        i32;                              (* int n_c_syms *)
        i32                               (* int n_libs *)
      |])
    in
    Llvm.refine_type crate_opaque_ty (Llvm.type_of_handle crate_tyhandle);
    Llvm.type_of_handle crate_tyhandle
  in
  ignore (Llvm.define_type_name "rust_crate" crate_ty llmod);

  let task_ty =
    (* TODO: other architectures besides x86 *)
    Llvm.struct_type llctx [|
      i32;                    (* size_t refcnt *)
      Llvm.pointer_type i32;  (* stk_seg *stk *)
      Llvm.pointer_type i32;  (* uintptr_t runtime_sp *)
      Llvm.pointer_type i32;  (* uintptr_t rust_sp *)
      Llvm.pointer_type i32;  (* rust_rt *rt *)
      Llvm.pointer_type i32   (* rust_crate_cache *cache *)
    |]
  in
  ignore (Llvm.define_type_name "rust_task" task_ty llmod);

  let rust_start_ty =
    let task_ptr_ty = Llvm.pointer_type task_ty in
    let llnilty = Llvm.array_type (Llvm.i1_type llctx) 0 in
    let main_ty = Llvm.function_type (Llvm.void_type llctx)
      [| Llvm.pointer_type llnilty; task_ptr_ty; |]
    in
    let args_ty = Array.map Llvm.pointer_type [| main_ty; crate_ty; |] in
    let args_ty = Array.append args_ty [| i32; i32 |] in
      Llvm.function_type i32 args_ty
  in
  {
    crate_ty = crate_ty;
    task_ty = task_ty;
    word_ty = i32;
    rust_start = Llvm.declare_function "rust_start" rust_start_ty llmod
  }
;;

