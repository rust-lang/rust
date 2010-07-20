(*
 * machine-specific assembler routines.
 *)

open Common;;

type asm_glue =
    {
      asm_activate_glue : Llvm.llvalue;
      asm_yield_glue : Llvm.llvalue;
      asm_upcall_glues : Llvm.llvalue array;
    }
;;

let n_upcall_glues = 7
;;

(* x86-specific asm. *)

let x86_glue
  (llctx:Llvm.llcontext)
  (llmod:Llvm.llmodule)
  (abi:Llabi.abi)
  (sess:Session.sess)
  : asm_glue =
  let (prefix,align) =
    match sess.Session.sess_targ with
        Linux_x86_elf
      | Win32_x86_pe -> ("",4)
      | MacOS_x86_macho -> ("_", 16)
  in
  let save_callee_saves =
    ["pushl %ebp";
     "pushl %edi";
     "pushl %esi";
     "pushl %ebx";]
  in
  let restore_callee_saves =
    ["popl  %ebx";
     "popl  %esi";
     "popl  %edi";
     "popl  %ebp";]
  in
  let load_esp_from_rust_sp =
    [ Printf.sprintf "movl  %d(%%edx), %%esp"
        (Abi.task_field_rust_sp * 4)]
  in
  let load_esp_from_runtime_sp =
    [ Printf.sprintf "movl  %d(%%edx), %%esp"
        (Abi.task_field_runtime_sp * 4) ]
  in
  let store_esp_to_rust_sp     =
    [ Printf.sprintf "movl  %%esp, %d(%%edx)"
        (Abi.task_field_rust_sp * 4) ]
  in
  let store_esp_to_runtime_sp  =
    [ Printf.sprintf "movl  %%esp, %d(%%edx)"
        (Abi.task_field_runtime_sp * 4) ]
  in

  let list_init i f = (Array.to_list (Array.init i f)) in
  let list_init_concat i f = List.concat (list_init i f) in

  let glue =
    [
      ("rust_activate_glue",
       String.concat "\n\t"
         (["movl  4(%esp), %edx    # edx = rust_task"]
          @ save_callee_saves
          @ store_esp_to_runtime_sp
          @ load_esp_from_rust_sp
            (* 
             * This 'add' instruction is a bit surprising.
             * See lengthy comment in boot/be/x86.ml activate_glue.
             *)
          @ [ Printf.sprintf
                "addl  $20, %d(%%edx)"
                (Abi.task_field_rust_sp * 4) ]

          @ restore_callee_saves
          @ ["ret"]));

      ("rust_yield_glue",
       String.concat "\n\t"

         (["movl  0(%esp), %edx    # edx = rust_task"]
          @ load_esp_from_rust_sp
          @ save_callee_saves
          @ store_esp_to_rust_sp
          @ load_esp_from_runtime_sp
          @ restore_callee_saves
          @ ["ret"]))
    ]
    @ list_init n_upcall_glues
      begin
        fun i ->
          (* 
           * 0, 4, 8, 12 are callee-saves
           * 16 is retpc
           * 20 is taskptr
           * 24 is callee
           * 28 .. (7+i) * 4 are args
           *)

          ((Printf.sprintf "rust_upcall_%d" i),
           String.concat "\n\t"
             (save_callee_saves
              @ ["movl  %esp, %ebp     # ebp = rust_sp";
                 "movl  20(%esp), %edx # edx = rust_task"]
              @ store_esp_to_rust_sp
              @ load_esp_from_runtime_sp
              @ [Printf.sprintf
                   "subl  $%d, %%esp   # esp -= args" ((i+1)*4);
                 "andl  $~0xf, %esp    # align esp down";
                 "movl  %edx, (%esp)   # arg[0] = rust_task "]

              @ (list_init_concat i
                   begin
                     fun j ->
                       [ Printf.sprintf "movl  %d(%%ebp),%%edx" ((j+7)*4);
                         Printf.sprintf "movl  %%edx,%d(%%esp)" ((j+1)*4) ]
                   end)

              @ ["movl  24(%ebp), %edx # edx = callee";
                 "call  *%edx          # call *%edx";
                 "movl  20(%ebp), %edx # edx = rust_task"]
              @ load_esp_from_rust_sp
              @ restore_callee_saves
              @ ["ret"]))
      end
  in

  let _ =
    Llvm.set_module_inline_asm llmod
      begin
        String.concat "\n"
          begin
            List.map
              begin
                fun (sym,asm) ->
                  Printf.sprintf
                    "\t.globl %s%s\n\t.balign %d\n%s%s:\n\t%s"
                    prefix sym align prefix sym asm
              end
              glue
          end
      end
  in

  let decl_cdecl_fn name out_ty arg_tys =
    let ty = Llvm.function_type out_ty arg_tys in
    let fn = Llvm.declare_function name ty llmod in
      Llvm.set_function_call_conv Llvm.CallConv.c fn;
      fn
  in

  let decl_glue s =
    let task_ptr_ty = Llvm.pointer_type abi.Llabi.task_ty in
    let void_ty = Llvm.void_type llctx in
      decl_cdecl_fn s void_ty [| task_ptr_ty |]
  in

  let decl_upcall n =
    let task_ptr_ty = Llvm.pointer_type abi.Llabi.task_ty in
    let word_ty = abi.Llabi.word_ty in
    let callee_ty = word_ty in
    let args_ty =
      Array.append
        [| task_ptr_ty; callee_ty |]
        (Array.init n (fun _ -> word_ty))
    in
    let name = Printf.sprintf "rust_upcall_%d" n in
      decl_cdecl_fn name word_ty args_ty
  in
    {
      asm_activate_glue = decl_glue "rust_activate_glue";
      asm_yield_glue = decl_glue "rust_yield_glue";
      asm_upcall_glues = Array.init n_upcall_glues decl_upcall;
    }
;;

(* x64-specific asm. *)
(* arm-specific asm. *)
(* ... *)


let get_glue
  (llctx:Llvm.llcontext)
  (llmod:Llvm.llmodule)
  (abi:Llabi.abi)
  (sess:Session.sess)
  : asm_glue =
  match sess.Session.sess_targ with
      Linux_x86_elf
    | Win32_x86_pe
    | MacOS_x86_macho ->
        x86_glue llctx llmod abi sess
;;


(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
