(*
 * LLVM ABI-level stuff that needs to happen after modules have been
 * translated.
 *)

let finalize_module
    (llctx:Llvm.llcontext)
    (llmod:Llvm.llmodule)
    (abi:Llabi.abi)
    (asm_glue:Llasm.asm_glue)
    (exit_task_glue:Llvm.llvalue)
    (crate_ptr:Llvm.llvalue)
    : unit =
  let i32 = Llvm.i32_type llctx in

  (*
   * Count the number of Rust functions and the number of C functions by
   * simply (and crudely) testing whether each function in the module begins
   * with "_rust_".
   *)

  let (rust_fn_count, c_fn_count) =
    let count (rust_fn_count, c_fn_count) fn =
      let begins_with prefix str =
        let (str_len, prefix_len) =
          (String.length str, String.length prefix)
        in
        prefix_len <= str_len && (String.sub str 0 prefix_len) = prefix
      in
      if begins_with "_rust_" (Llvm.value_name fn) then
        (rust_fn_count + 1, c_fn_count)
      else
        (rust_fn_count, c_fn_count + 1)
    in
    Llvm.fold_left_functions count (0, 0) llmod
  in

  let crate_val =
    let crate_addr = Llvm.const_ptrtoint crate_ptr i32 in
    let glue_off glue =
      let addr = Llvm.const_ptrtoint glue i32 in
        Llvm.const_sub addr crate_addr
    in
    let activate_glue_off = glue_off asm_glue.Llasm.asm_activate_glue in
    let yield_glue_off = glue_off asm_glue.Llasm.asm_yield_glue in
    let exit_task_glue_off = glue_off exit_task_glue in

    Llvm.const_struct llctx [|
      Llvm.const_int i32 0;             (* ptrdiff_t image_base_off *)
      crate_ptr;                        (* uintptr_t self_addr *)
      Llvm.const_int i32 0;             (* ptrdiff_t debug_abbrev_off *)
      Llvm.const_int i32 0;             (* size_t debug_abbrev_sz *)
      Llvm.const_int i32 0;             (* ptrdiff_t debug_info_off *)
      Llvm.const_int i32 0;             (* size_t debug_info_sz *)
      activate_glue_off;                (* size_t activate_glue_off *)
      exit_task_glue_off;               (* size_t main_exit_task_glue_off *)
      Llvm.const_int i32 0;             (* size_t unwind_glue_off *)
      yield_glue_off;                   (* size_t yield_glue_off *)
      Llvm.const_int i32 rust_fn_count; (* int n_rust_syms *)
      Llvm.const_int i32 c_fn_count;    (* int n_c_syms *)
      Llvm.const_int i32 0              (* int n_libs *)
    |]
  in

  Llvm.set_initializer crate_val crate_ptr;

  (* Define the main function for crt0 to call. *)
  let main_fn =
    let main_ty = Llvm.function_type i32 [| i32; i32 |] in
    Llvm.define_function "main" main_ty llmod
  in
  let argc = Llvm.param main_fn 0 in
  let argv = Llvm.param main_fn 1 in
  let main_builder = Llvm.builder_at_end llctx (Llvm.entry_block main_fn) in
  let rust_main_fn =
    match Llvm.lookup_function "_rust_main" llmod with
        None -> raise (Failure "no main function found")
      | Some fn -> fn
  in
  let rust_start = abi.Llabi.rust_start in
  let rust_start_args = [| rust_main_fn; crate_ptr; argc; argv |] in
    ignore (Llvm.build_call
              rust_start rust_start_args "start_rust" main_builder);
    ignore (Llvm.build_ret (Llvm.const_int i32 0) main_builder)
;;


(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)

