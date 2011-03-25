open Common;;
open Semant;;

(* A note on GC:
 * 
 * We employ -- or "will employ" when the last few pieces of it are done -- a
 * "simple" precise, mark-sweep, single-generation, per-task (thereby
 * preemptable and relatively quick) GC scheme on mutable memory.
 * 
 * - For the sake of this note, call any box of 'state' effect a gc_val.
 *
 * - gc_vals come from the same malloc as all other values but undergo
 *   different storage management.
 *
 *  - Every frame has a frame_glue_fns pointer in its fp[-1] slot, written on
 *    function-entry.
 *
 *  - gc_vals have *three* extra words at their head, not one.
 *
 *  - A pointer to a gc_val, however, points to the third of these three
 *    words. So a certain quantity of code can treat gc_vals the same way it
 *    would treat refcounted box vals.
 *
 *  - The first word at the head of a gc_val is used as a refcount, as in
 *    non-gc allocations.
 *
 *  - The (-1)st word at the head of a gc_val is a pointer to a tydesc,
 *    with the low bit of that pointer used as a mark bit.
 *
 *  - The (-2)nd word at the head of a gc_val is a linked-list pointer to the
 *    gc_val that was allocated (temporally) just before it. Following this
 *    list traces through all the currently active gc_vals in a task.
 *
 *  - The task has a gc_alloc_chain field that points to the most-recent
 *    gc_val allocated.
 *
 *  - GC glue has two phases, mark and sweep:
 * 
 *    - The mark phase walks down the frame chain, like the unwinder. It calls
 *      each frame's mark glue as it's passing through. This will mark all the
 *      reachable parts of the task's gc_vals.
 * 
 *    - The sweep phase walks down the task's gc_alloc_chain checking to see
 *      if each allocation has been marked. If marked, it has its mark-bit
 *      reset and the sweep passes it by. If unmarked, it has its tydesc
 *      free_glue called on its body, and is unlinked from the chain. The
 *      free-glue will cause the allocation to (recursively) drop all of its
 *      references and/or run dtors.
 * 
 *    - Note that there is no "special gc state" at work here; the task looks
 *      like it's running normal code that happens to not perform any gc_val
 *      allocation. Mark-bit twiddling is open-coded into all the mark
 *      functions, which know their contents; we only have to do O(frames)
 *      indirect calls to mark, the rest are static. Sweeping costs O(gc-heap)
 *      indirect calls, unfortunately, because the set of sweep functions to
 *      call is arbitrary based on allocation order.
 *)


type deref_ctrl =
    DEREF_one_box
  | DEREF_all_boxes
  | DEREF_none
;;

type mem_ctrl =
    MEM_rc_opaque
  | MEM_rc_struct
  | MEM_gc
  | MEM_interior
;;

type clone_ctrl =
    CLONE_none
  | CLONE_chan of Il.cell
  | CLONE_all of Il.cell
;;

type call_ctrl =
    CALL_direct
  | CALL_vtbl
  | CALL_indirect
;;

type for_each_ctrl =
    {
      for_each_fixup: fixup;
      for_each_depth: int;
    }
;;

let word_sz (abi:Abi.abi) : int64 =
  abi.Abi.abi_word_sz
;;

let word_n (abi:Abi.abi) (n:int) : int64 =
  Int64.mul (word_sz abi) (Int64.of_int n)
;;

let word_bits (abi:Abi.abi) : Il.bits =
  abi.Abi.abi_word_bits
;;

let word_ty_mach (abi:Abi.abi) : ty_mach =
  match word_bits abi with
      Il.Bits8 -> TY_u8
    | Il.Bits16 -> TY_u16
    | Il.Bits32 -> TY_u32
    | Il.Bits64 -> TY_u64
;;

let word_ty_signed_mach (abi:Abi.abi) : ty_mach =
  match word_bits abi with
      Il.Bits8 -> TY_i8
    | Il.Bits16 -> TY_i16
    | Il.Bits32 -> TY_i32
    | Il.Bits64 -> TY_i64
;;


let rec ty_mem_ctrl (cx:ctxt) (ty:Ast.ty) : mem_ctrl =
  match ty with
      Ast.TY_port _
    | Ast.TY_chan _
    | Ast.TY_task
    | Ast.TY_str -> MEM_rc_opaque
    | Ast.TY_vec _ ->
        if type_has_state cx ty
        then MEM_gc
        else MEM_rc_opaque
    | Ast.TY_box t ->
        if type_has_state cx t
        then MEM_gc
        else
          if type_is_structured cx t
          then MEM_rc_struct
          else MEM_rc_opaque
    | Ast.TY_mutable t
    | Ast.TY_constrained (t, _) ->
        ty_mem_ctrl cx t
    | _ ->
        MEM_interior
;;

let slot_mem_ctrl (cx:ctxt) (slot:Ast.slot) : mem_ctrl =
  match slot.Ast.slot_mode with
      Ast.MODE_alias -> MEM_interior
    | Ast.MODE_local ->
        ty_mem_ctrl cx (slot_ty slot)
;;


let iter_block_slots
    (cx:Semant.ctxt)
    (block_id:node_id)
    (fn:Ast.slot_key -> node_id -> Ast.slot -> unit)
    : unit =
  let block_slots = Hashtbl.find cx.ctxt_block_slots block_id in
    Hashtbl.iter
      begin
        fun key slot_id ->
          let slot = get_slot cx slot_id in
            fn key slot_id slot
      end
      block_slots
;;

let iter_frame_slots
    (cx:Semant.ctxt)
    (frame_id:node_id)
    (fn:Ast.slot_key -> node_id -> Ast.slot -> unit)
    : unit =
  let blocks = Hashtbl.find cx.ctxt_frame_blocks frame_id in
    List.iter (fun block -> iter_block_slots cx block fn) blocks
;;

let iter_arg_slots
    (cx:Semant.ctxt)
    (frame_id:node_id)
    (fn:Ast.slot_key -> node_id -> Ast.slot -> unit)
    : unit =
  match htab_search cx.ctxt_frame_args frame_id with
      None -> ()
    | Some ls ->
        List.iter
          begin
            fun slot_id ->
              let key = Hashtbl.find cx.ctxt_slot_keys slot_id in
              let slot = get_slot cx slot_id in
                fn key slot_id slot
          end
          ls
;;

let iter_frame_and_arg_slots
    (cx:Semant.ctxt)
    (frame_id:node_id)
    (fn:Ast.slot_key -> node_id -> Ast.slot -> unit)
    : unit =
  iter_frame_slots cx frame_id fn;
  iter_arg_slots cx frame_id fn;
;;

let next_power_of_two (x:int64) : int64 =
  let xr = ref (Int64.sub x 1L) in
    xr := Int64.logor (!xr) (Int64.shift_right_logical (!xr) 1);
    xr := Int64.logor (!xr) (Int64.shift_right_logical (!xr) 2);
    xr := Int64.logor (!xr) (Int64.shift_right_logical (!xr) 4);
    xr := Int64.logor (!xr) (Int64.shift_right_logical (!xr) 8);
    xr := Int64.logor (!xr) (Int64.shift_right_logical (!xr) 16);
    xr := Int64.logor (!xr) (Int64.shift_right_logical (!xr) 32);
    Int64.add 1L (!xr)
;;

let iter_tup_parts
    (get_element_ptr:'a -> int -> 'a)
    (dst_ptr:'a)
    (src_ptr:'a)
    (tys:Ast.ty_tup)
    (f:'a -> 'a -> Ast.ty -> unit)
    : unit =
  Array.iteri
    begin
      fun i ty ->
        f (get_element_ptr dst_ptr i)
          (get_element_ptr src_ptr i)
          ty
    end
    tys
;;

let iter_rec_parts
    (get_element_ptr:'a -> int -> 'a)
    (dst_ptr:'a)
    (src_ptr:'a)
    (entries:Ast.ty_rec)
    (f:'a -> 'a -> Ast.ty -> unit)
    : unit =
  iter_tup_parts get_element_ptr dst_ptr src_ptr
    (Array.map snd entries) f
;;


(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
