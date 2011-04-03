
(*
 * The 'abi' structure is pretty much just a grab-bag of machine
 * dependencies and structure-layout information. Part of the latter
 * is shared with trans and semant.
 *
 * Make some attempt to factor it as time goes by.
 *)

(* Word offsets for structure fields in rust-internal.h, and elsewhere in
   compiler. *)

let rc_base_field_refcnt = 0;;

(* FIXME: this needs updating if you ever want to work on 64 bit. *)
let const_refcount = 0x7badfaceL;;

let task_field_refcnt = rc_base_field_refcnt;;
let task_field_stk = task_field_refcnt + 2;;
let task_field_runtime_sp = task_field_stk + 1;;
let task_field_rust_sp = task_field_runtime_sp + 1;;
let task_field_gc_alloc_chain = task_field_rust_sp + 1;;
let task_field_dom = task_field_gc_alloc_chain + 1;;
let n_visible_task_fields = task_field_dom + 1;;

let dom_field_interrupt_flag = 1;;

let frame_glue_fns_field_mark = 0;;
let frame_glue_fns_field_drop = 1;;
let frame_glue_fns_field_reloc = 2;;

let box_rc_field_refcnt = 0;;
let box_rc_field_body = 1;;

let box_gc_alloc_base = (-3);;
let box_gc_field_prev = (-3);;
let box_gc_field_next = (-2);;
let box_gc_field_ctrl = (-1);;
let box_gc_field_refcnt = 0;;
let box_gc_field_body = 1;;

let box_rc_header_size = 1;;
let box_gc_header_size = 4;;

let box_gc_malloc_return_adjustment = 3;;

let stk_field_valgrind_id = 0;;
let stk_field_limit = stk_field_valgrind_id + 1;;
let stk_field_data = stk_field_limit + 1;;

(* Both obj and fn are two-word "bindings": One word points to some static
 * dispatch information (vtbl, thunk, callee), and the other points to some
 * box of bound data (object-body or closure).
 *)

let binding_field_dispatch = 0;;
let binding_field_bound_data = 1;;

let obj_field_vtbl = binding_field_dispatch;;
let obj_field_box = binding_field_bound_data;;

let obj_body_elt_tydesc = 0;;
let obj_body_elt_fields = 1;;

let fn_field_code = binding_field_dispatch;;
let fn_field_box = binding_field_bound_data;;

(* NB: bound ty params come last to facilitate ignoring them on
 * closure-dropping. *)
let closure_body_elt_bound_args_tydesc = 0;;
let closure_body_elt_target = 1;;
let closure_body_elt_bound_args = 2;;
let closure_body_elt_bound_ty_params = 3;;

let tag_elt_discriminant = 0;;
let tag_elt_variant = 1;;

let general_code_alignment = 16;;

let tydesc_field_first_param = 0;;
let tydesc_field_size = 1;;
let tydesc_field_align = 2;;
let tydesc_field_take_glue = 3;;
let tydesc_field_drop_glue = 4;;
let tydesc_field_free_glue = 5;;
let tydesc_field_sever_glue = 6;;
let tydesc_field_mark_glue = 7;;
let tydesc_field_obj_drop_glue = 8;;
let tydesc_field_cmp_glue = 9;;   (* FIXME these two aren't in the *)
let tydesc_field_hash_glue = 10;; (* runtime's type_desc struct.   *)
let tydesc_field_stateflag = 11;;

let vec_elt_rc = 0;;
let vec_elt_alloc = 1;;
let vec_elt_fill = 2;;
let vec_elt_pad = 3;;
let vec_elt_data = 4;;

let calltup_elt_out_ptr = 0;;
let calltup_elt_task_ptr = 1;;
let calltup_elt_indirect_args = 2;;
let calltup_elt_ty_params = 3;;
let calltup_elt_args = 4;;
let calltup_elt_iterator_args = 5;;

let iterator_args_elt_block_fn = 0;;
let iterator_args_elt_outer_frame_ptr = 1;;

let indirect_args_elt_closure = 0;;

(* Current worst case is by vec grow glue *)
let worst_case_glue_call_args = 8;;

(* 
 * ABI tags used to inform the runtime which sort of frame to set up for new
 * spawned functions. FIXME: There is almost certainly a better abstraction to
 * use.
 *)
let abi_x86_rustboot_cdecl = 1;;
let abi_x86_rustc_fastcall = 2;;

type abi =
    {
      abi_word_sz: int64;
      abi_word_bits: Il.bits;
      abi_word_ty: Common.ty_mach;

      abi_tag: int;

      abi_has_pcrel_data: bool;
      abi_has_pcrel_code: bool;

      abi_n_hardregs: int;
      abi_str_of_hardreg: (int -> string);

      abi_emit_target_specific: (Il.emitter -> Il.quad -> unit);
      abi_constrain_vregs: (Il.quad -> (Il.vreg,Bits.t) Hashtbl.t -> unit);

      abi_emit_fn_prologue: (Il.emitter
                             -> Common.size        (* framesz *)
                             -> Common.size      (* callsz  *)
                               -> Common.nabi
                                 -> Common.fixup (* grow_task *)
                                   -> bool       (* is_obj_fn *)
                                     -> bool     (* minimal *)
                                       -> unit);

    abi_emit_fn_epilogue: (Il.emitter -> unit);

    abi_emit_fn_tail_call: (Il.emitter
                            -> int64            (* caller_callsz *)
                              -> int64          (* caller_argsz  *)
                                -> Il.code      (* callee_code   *)
                                  -> int64      (* callee_argsz  *)
                                    -> unit);

    abi_clobbers: (Il.quad -> Il.hreg list);

    abi_emit_native_call: (Il.emitter
                           -> Il.cell                 (* ret    *)
                             -> Common.nabi
                               -> Common.fixup        (* callee *)
                                 -> Il.operand array  (* args   *)
                                   -> unit);

    abi_emit_native_void_call: (Il.emitter
                                -> Common.nabi
                                  -> Common.fixup             (* callee *)
                                    -> Il.operand array       (* args   *)
                                      -> unit);

    abi_emit_native_call_in_thunk: (Il.emitter
                                    -> Il.cell option         (* ret    *)
                                      -> Common.nabi
                                        -> Il.operand         (* callee *)
                                          -> Il.operand array (* args   *)
                                            -> unit);
    abi_emit_inline_memcpy: (Il.emitter
                             -> int64           (* n_bytes   *)
                               -> Il.reg        (* dst_ptr   *)
                                 -> Il.reg      (* src_ptr   *)
                                   -> Il.reg    (* tmp_reg   *)
                                     -> bool    (* ascending *)
                                       -> unit);

    (* Global glue. *)
    abi_activate: (Il.emitter -> unit);
    abi_yield: (Il.emitter -> unit);
    abi_unwind: (Il.emitter -> Common.nabi -> Common.fixup -> unit);
    abi_gc: (Il.emitter -> unit);
    abi_get_next_pc_thunk:
      ((Il.reg                   (* output            *)
        * Common.fixup           (* thunk in objfile  *)
        * (Il.emitter -> unit))  (* fn to make thunk  *)
         option);

    abi_sp_reg: Il.reg;
    abi_fp_reg: Il.reg;
    abi_dwarf_fp_reg: int;
    abi_tp_cell: Il.cell;
    abi_implicit_args_sz: int64;
    abi_frame_base_sz: int64;
    abi_callee_saves_sz: int64;
    abi_frame_info_sz: int64;
    abi_spill_slot: (Il.spill -> Il.mem);
  }
;;

let load_fixup_addr
    (e:Il.emitter)
    (out_reg:Il.reg)
    (fix:Common.fixup)
    (rty:Il.referent_ty)
    : unit =

  let cell = Il.Reg (out_reg, Il.AddrTy rty) in
  let op = Il.ImmPtr (fix, rty) in
    Il.emit e (Il.lea cell op);
;;

let load_fixup_codeptr
    (e:Il.emitter)
    (out_reg:Il.reg)
    (fixup:Common.fixup)
    (has_pcrel_code:bool)
    (indirect:bool)
    : Il.code =
  if indirect
  then
    begin
      load_fixup_addr e out_reg fixup (Il.ScalarTy (Il.AddrTy Il.CodeTy));
      Il.CodePtr (Il.Cell (Il.Mem (Il.RegIn (out_reg, None),
                                   Il.ScalarTy (Il.AddrTy Il.CodeTy))))
    end
  else
    if has_pcrel_code
    then (Il.CodePtr (Il.ImmPtr (fixup, Il.CodeTy)))
    else
      begin
        load_fixup_addr e out_reg fixup Il.CodeTy;
        Il.CodePtr (Il.Cell (Il.Reg (out_reg, Il.AddrTy Il.CodeTy)))
      end
;;


(* 
 * Local Variables:
 * fill-column: 78; 
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'"; 
 * End:
 *)
