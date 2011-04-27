(* Translation *)

open Semant;;
open Common;;
open Transutil;;

let log cx = Session.log "trans"
  (should_log cx cx.ctxt_sess.Session.sess_log_trans)
  cx.ctxt_sess.Session.sess_log_out
;;

let arr_max a = (Array.length a) - 1;;

type quad_idx = int
;;

type call =
    {
      call_ctrl: call_ctrl;
      call_callee_ptr: Il.operand;
      call_callee_ty: Ast.ty;
      call_callee_ty_params: Ast.ty array;
      call_output: Il.cell;
      call_args: Ast.atom array;
      call_iterator_args: Il.operand array;
      call_indirect_args: Il.operand array;
    }
;;

let need_ty_fn ty =
  match simplified_ty ty with
      Ast.TY_fn tfn -> tfn
    | _ -> bug () "need fn"
;;

let call_output_slot call =
  (fst (need_ty_fn call.call_callee_ty)).Ast.sig_output_slot
;;


type const =
    CONST_val of int64
  | CONST_frag of Asm.frag
;;

let trans_visitor
    (cx:ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =

  let iflog thunk =
    if (should_log cx cx.ctxt_sess.Session.sess_log_trans)
    then thunk ()
    else ()
  in

  let curr_file = Stack.create () in
  let curr_stmt = Stack.create () in

  let (abi:Abi.abi) = cx.ctxt_abi in
  let (word_sz:int64) = word_sz abi in
  let (word_slot:Ast.slot) = word_slot abi in
  let (word_ty:Ast.ty) = Ast.TY_mach abi.Abi.abi_word_ty in

  let oper_str = Il.string_of_operand abi.Abi.abi_str_of_hardreg in
  let cell_str = Il.string_of_cell abi.Abi.abi_str_of_hardreg in

  let (word_bits:Il.bits) = abi.Abi.abi_word_bits in
  let (word_sty:Il.scalar_ty) = Il.ValTy word_bits in
  let (word_rty:Il.referent_ty) = Il.ScalarTy word_sty in
  let (word_ty_mach:ty_mach) =
    match word_bits with
        Il.Bits8 -> TY_u8
      | Il.Bits16 -> TY_u16
      | Il.Bits32 -> TY_u32
      | Il.Bits64 -> TY_u64
  in
  let (word_ty_signed_mach:ty_mach) =
    match word_bits with
        Il.Bits8 -> TY_i8
      | Il.Bits16 -> TY_i16
      | Il.Bits32 -> TY_i32
      | Il.Bits64 -> TY_i64
  in
  let word_n = word_n abi in
  let imm_of_ty (i:int64) (tm:ty_mach) : Il.operand =
    Il.Imm (Asm.IMM i, tm)
  in

  let imm (i:int64) : Il.operand = imm_of_ty i word_ty_mach in
  let simm (i:int64) : Il.operand = imm_of_ty i word_ty_signed_mach in
  let one = imm 1L in
  let neg_one = simm (-1L) in
  let zero = imm 0L in
  let imm_true = imm_of_ty 1L TY_u8 in
  let imm_false = imm_of_ty 0L TY_u8 in
  let zero_byte = imm_of_ty 0L TY_u8 in
  let nil_ptr = Il.Mem ((Il.Abs (Asm.IMM 0L)), Il.NilTy) in
  let wordptr_ty = Il.AddrTy (Il.ScalarTy word_sty) in

  let crate_rel fix =
    Asm.SUB (Asm.M_POS fix, Asm.M_POS cx.ctxt_crate_fixup)
  in

  let crate_rel_word fix =
    Asm.WORD (word_ty_signed_mach, crate_rel fix)
  in

  let crate_rel_imm (fix:fixup) : Il.operand =
    Il.Imm (crate_rel fix, word_ty_signed_mach)
  in

  let fixup_rel_word (base:fixup) (fix:fixup) =
    Asm.WORD (word_ty_signed_mach,
              Asm.SUB (Asm.M_POS fix, Asm.M_POS base))
  in

  let table_of_fixup_rel_fixups
      (fixup:fixup)
      (fixups:fixup array)
      : Asm.frag =
    Asm.SEQ (Array.map (fixup_rel_word fixup) fixups)
  in

  let table_of_table_rel_fixups (fixups:fixup array) : Asm.frag =
    let table_fix = new_fixup "vtbl" in
      Asm.DEF (table_fix, table_of_fixup_rel_fixups table_fix fixups)
  in

  let nabi_indirect =
      match cx.ctxt_sess.Session.sess_targ with
          Linux_x86_elf -> false
        | _ -> true
  in

  let nabi_rust =
    { nabi_indirect = nabi_indirect;
      nabi_convention = CONV_rust }
  in

  let out_mem_disp = abi.Abi.abi_frame_base_sz in
  let arg0_disp =
    Int64.add abi.Abi.abi_frame_base_sz abi.Abi.abi_implicit_args_sz
  in
  let frame_info_disp =
    Int64.neg (Int64.add
                 abi.Abi.abi_frame_info_sz
                 abi.Abi.abi_callee_saves_sz)
  in
  let frame_fns_disp = Int64.add frame_info_disp (word_n 0) in
  let frame_crate_ptr_disp = Int64.add frame_info_disp (word_n 1) in

  let fn_ty (id:node_id) : Ast.ty =
    Hashtbl.find cx.ctxt_all_item_types id
  in
  let fn_args_rty
      (id:node_id)
      (closure:Il.referent_ty option)
      : Il.referent_ty =
    let n_params =
      if defn_id_is_obj_fn_or_drop cx id
      then 0
      else n_item_ty_params cx id
    in
      call_args_referent_type cx n_params (fn_ty id) closure
  in

  let emitters = Stack.create () in
  let push_new_emitter (vregs_ok:bool) (fnid:node_id option) =
    let e = Il.new_emitter
      abi.Abi.abi_emit_target_specific
      vregs_ok fnid
    in
      Stack.push e emitters;
  in

  let push_new_emitter_with_vregs fnid = push_new_emitter true fnid in
  let push_new_emitter_without_vregs fnid = push_new_emitter false fnid in

  let pop_emitter _ = ignore (Stack.pop emitters) in
  let emitter _ = Stack.top emitters in
  let emitter_size_cache _ = (emitter()).Il.emit_size_cache in
  let flush_emitter_size_cache _ =
    Hashtbl.clear (emitter_size_cache())
  in

  let quad_categories = Hashtbl.create 0 in
  let quad_category_stack = Stack.create () in
  let in_quad_category name thunk =
    if cx.ctxt_sess.Session.sess_report_quads
    then Stack.push name quad_category_stack;
    let x = thunk() in
      if cx.ctxt_sess.Session.sess_report_quads
      then ignore (Stack.pop quad_category_stack);
      x
  in

  let credit name i =
    let c =
      htab_search_or_add quad_categories name
        (fun _ -> ref 0)
    in
      c := (!c) + i
  in

  let in_native_quad_category name thunk =
    if cx.ctxt_sess.Session.sess_report_quads
    then
      let i = (emitter()).Il.emit_pc in
      let x = thunk() in
      let j = (emitter()).Il.emit_pc in
        credit name (j-i);
        x
    else
      thunk()
  in

  let emit q =
    begin
      match q with
        Il.Jmp _ -> flush_emitter_size_cache();
        | _ -> ()
    end;
    Il.emit (emitter()) q;
    if cx.ctxt_sess.Session.sess_report_quads
    then
      begin
        let name =
          if Stack.is_empty quad_category_stack
          then "other"
          else Stack.top quad_category_stack
        in
          credit name 1
      end
  in

  let next_vreg _ = Il.next_vreg (emitter()) in
  let next_vreg_cell t = Il.next_vreg_cell (emitter()) t in
  let next_spill_cell t =
    let s = Il.next_spill (emitter()) in
    let spill_mem = Il.Spill s in
    let spill_ta = (spill_mem, Il.ScalarTy t) in
      Il.Mem spill_ta
  in
  let mark _ : quad_idx =
    flush_emitter_size_cache ();
    (emitter()).Il.emit_pc
  in
  let patch_existing (jmp:quad_idx) (targ:quad_idx) : unit =
    Il.patch_jump (emitter()) jmp targ;
    flush_emitter_size_cache ();
  in
  let patch (i:quad_idx) : unit =
    Il.patch_jump (emitter()) i (mark());
    flush_emitter_size_cache ();
    (* Insert a dead quad to ensure there's an otherwise-unused
     * jump-target here.
     *)
    emit Il.Dead
  in

  let current_fn () =
    match (emitter()).Il.emit_node with
        None -> bug () "current_fn without associated node"
      | Some id -> id
  in
  let current_fn_args_rty (closure:Il.referent_ty option) : Il.referent_ty =
    fn_args_rty (current_fn()) closure
  in
  let current_fn_callsz () = get_callsz cx (current_fn()) in

  let annotations _ =
    (emitter()).Il.emit_annotations
  in

  let annotate (str:string) =
    let e = emitter() in
      Hashtbl.add e.Il.emit_annotations e.Il.emit_pc str
  in

  let epilogue_jumps = Stack.create() in
  let simple_break_jumps = Stack.create() in (* not used for for-each *)

  let path_name (_:unit) : string =
    string_of_name (path_to_name cx.ctxt_curr_path)
  in

  let should_inline_structure_helpers _ = false in

  let based (reg:Il.reg) : Il.mem =
    Il.RegIn (reg, None)
  in

  let based_off (reg:Il.reg) (off:Asm.expr64) : Il.mem =
    Il.RegIn (reg, Some off)
  in

  let based_imm (reg:Il.reg) (imm:int64) : Il.mem =
    based_off reg (Asm.IMM imm)
  in

  let fp_imm (imm:int64) : Il.mem =
    based_imm abi.Abi.abi_fp_reg imm
  in

  let sp_imm (imm:int64) : Il.mem =
    based_imm abi.Abi.abi_sp_reg imm
  in

  let word_at (mem:Il.mem) : Il.cell =
    Il.Mem (mem, Il.ScalarTy (Il.ValTy word_bits))
  in

  let imov (dst:Il.cell) (src:Il.operand) : unit =
    emit (Il.imov dst src)
  in

  let mov (dst:Il.cell) (src:Il.operand) : unit =
    emit (Il.umov dst src)
  in

  let umul (dst:Il.cell) (a:Il.operand) (b:Il.operand) : unit =
    emit (Il.binary Il.UMUL dst a b);
  in

  let add (dst:Il.cell) (a:Il.operand) (b:Il.operand) : unit =
    emit (Il.binary Il.ADD dst a b);
  in

  let add_to (dst:Il.cell) (src:Il.operand) : unit =
    add dst (Il.Cell dst) src;
  in

  let sub (dst:Il.cell) (a:Il.operand) (b:Il.operand) : unit =
    emit (Il.binary Il.SUB dst a b);
  in

  let sub_from (dst:Il.cell) (src:Il.operand) : unit =
    sub dst (Il.Cell dst) src;
  in

  let lea (dst:Il.cell) (src:Il.mem) : unit =
    emit (Il.lea dst (Il.Cell (Il.Mem (src, Il.OpaqueTy))))
  in

  let rty_ptr_at (mem:Il.mem) (pointee_rty:Il.referent_ty) : Il.cell =
    Il.Mem (mem, Il.ScalarTy (Il.AddrTy pointee_rty))
  in

  let ptr_at (mem:Il.mem) (pointee_ty:Ast.ty) : Il.cell =
    rty_ptr_at mem (referent_type cx pointee_ty)
  in

  let need_scalar_ty (rty:Il.referent_ty) : Il.scalar_ty =
    match rty with
        Il.ScalarTy s -> s
      | _ -> bug () "expected ScalarTy"
  in

  let need_mem_cell (cell:Il.cell) : Il.typed_mem =
    match cell with
        Il.Mem a -> a
      | Il.Reg _ -> bug ()
          "expected address cell, got non-address register cell"
  in

  let need_cell (operand:Il.operand) : Il.cell =
    match operand with
        Il.Cell c -> c
      | _ -> bug () "expected cell, got operand %s"
          (Il.string_of_operand  abi.Abi.abi_str_of_hardreg operand)
  in

  let get_element_ptr =
    Il.get_element_ptr word_bits abi.Abi.abi_str_of_hardreg
  in

  let get_variant_ptr (mem_cell:Il.cell) (i:int) : Il.cell =
    match mem_cell with
        Il.Mem (mem, Il.UnionTy elts)
          when i >= 0 && i < (Array.length elts) ->
            assert ((Array.length elts) != 0);
            Il.Mem (mem, elts.(i))

      | _ -> bug () "get_variant_ptr %d on cell %s" i
          (cell_str mem_cell)
  in

  let rec ptr_cast = Il.ptr_cast

  and cell_cast = Il.cell_cast

  and curr_crate_ptr _ : Il.cell =
    word_at (fp_imm frame_crate_ptr_disp)

  and crate_rel_to_ptr (rel:Il.operand) (rty:Il.referent_ty) : Il.cell =
    (in_quad_category "crate_rel -> ptr"
       (fun _ ->
          let cell = next_vreg_cell (Il.AddrTy rty) in
          let diff = next_vreg_cell (Il.AddrTy rty) in
            mov cell (Il.Cell (curr_crate_ptr()));
            imov diff rel;
            add_to cell (Il.Cell diff);
            cell))

  (* 
   * Note: alias *requires* its cell to be in memory already, and should
   * only be used on slots you know to be memory-resident. Use 'aliasing' or 
   * 'via_memory' if you have a cell or operand you want in memory for a very
   * short period of time (the time spent by the code generated by the thunk).
   *)

  and alias (cell:Il.cell) : Il.cell =
    let mem, ty = need_mem_cell cell in
    let vreg_cell = next_vreg_cell (Il.AddrTy ty) in
      begin
        match ty with
            Il.NilTy -> ()
          | _ -> lea vreg_cell mem
      end;
      vreg_cell

  and force_to_mem (src:Il.operand) : Il.typed_mem =
    let do_spill op (t:Il.scalar_ty) =
      let spill = next_spill_cell t in
        mov spill op;
        need_mem_cell spill
    in
    match src with
        Il.Cell (Il.Mem ta) -> ta
      | Il.Cell (Il.Reg (_, t)) -> do_spill src t
      | Il.Imm (_,tm) -> do_spill src (Il.ValTy (Il.bits_of_ty_mach tm))
      | Il.ImmPtr (f, rty) ->
          do_spill
            (Il.Cell (crate_rel_to_ptr (crate_rel_imm f) rty))
            (Il.AddrTy rty)

  and force_to_reg (op:Il.operand) : Il.typed_reg =
    let do_mov op st =
      let tmp = next_vreg () in
      let regty = (tmp, st) in
        mov (Il.Reg regty) op;
        regty
    in
      match op with
          Il.Imm  (_, tm) -> do_mov op (Il.ValTy (Il.bits_of_ty_mach tm))
        | Il.ImmPtr (f, rty) ->
            do_mov
              (Il.Cell (crate_rel_to_ptr (crate_rel_imm f) rty))
              (Il.AddrTy rty)
        | Il.Cell (Il.Reg rt) -> rt
        | Il.Cell (Il.Mem (_, Il.ScalarTy st)) -> do_mov op st
        | Il.Cell (Il.Mem (_, rt)) ->
            bug () "forcing non-scalar referent of type %s to register"
              (Il.string_of_referent_ty rt)

  and via_memory (writeback:bool) (c:Il.cell) (thunk:Il.cell -> unit) : unit =
    match c with
        Il.Mem _ -> thunk c
      | Il.Reg _ ->
          let mem_c = Il.Mem (force_to_mem (Il.Cell c)) in
            thunk mem_c;
            if writeback
            then
              mov c (Il.Cell mem_c)

  and aliasing (writeback:bool) (c:Il.cell) (thunk:Il.cell -> unit) : unit =
    via_memory writeback c (fun c -> thunk (alias c))

  and pointee_type (ptr:Il.cell) : Il.referent_ty =
    match ptr with
        Il.Reg (_, (Il.AddrTy rt)) -> rt
      | Il.Mem (_, Il.ScalarTy (Il.AddrTy rt)) -> rt
      | _ ->
          bug () "taking pointee-type of non-address cell %s "
            (cell_str ptr)

  and deref (ptr:Il.cell) : Il.cell =
    let (r, st) = force_to_reg (Il.Cell ptr) in
      match st with
          Il.AddrTy rt -> Il.Mem (based r, rt)
        | _ -> bug () "dereferencing non-address cell of type %s "
            (Il.string_of_scalar_ty st)

  and deref_off (ptr:Il.cell) (off:Asm.expr64) : Il.cell =
    let (r, st) = force_to_reg (Il.Cell ptr) in
      match st with
          Il.AddrTy rt -> Il.Mem (based_off r off, rt)
        | _ -> bug () "offset-dereferencing non-address cell of type %s "
            (Il.string_of_scalar_ty st)

  and deref_imm (ptr:Il.cell) (imm:int64) : Il.cell =
    deref_off ptr (Asm.IMM imm)

  and tp_imm (imm:int64) : Il.cell =
    deref_imm abi.Abi.abi_tp_cell imm
  in


  let make_tydesc_tys n =
    Array.init n (fun _ -> Ast.TY_type)
  in

  let cell_vreg_num (vr:(int option) ref) : int =
    match !vr with
        None ->
          let v = (Il.next_vreg_num (emitter())) in
            vr := Some v;
            v
      | Some v -> v
  in

  let slot_id_referent_type (slot_id:node_id) : Il.referent_ty =
    slot_referent_type cx (get_slot cx slot_id)
  in

  let caller_args_cell (args_rty:Il.referent_ty) : Il.cell =
    Il.Mem (fp_imm out_mem_disp, args_rty)
  in

  let get_obj_box_from_calltup (args_cell:Il.cell) =
    let indirect_args =
      get_element_ptr args_cell Abi.calltup_elt_indirect_args
    in
      deref (cell_cast
               (get_element_ptr indirect_args Abi.indirect_args_elt_closure)
               (Il.ScalarTy (Il.AddrTy (obj_box_rty word_bits))))
  in

  let fp_to_args (fp:Il.cell) (args_rty:Il.referent_ty): Il.cell =
    let (reg, _) = force_to_reg (Il.Cell fp) in
    Il.Mem(based_imm reg out_mem_disp, args_rty)
  in

  let get_ty_param (ty_params:Il.cell) (param_idx:int) : Il.cell =
      get_element_ptr ty_params param_idx
  in

  let get_ty_params_of_frame
      (fnid:node_id)
      (fp:Il.reg)
      (n_ty_params:int)
      : Il.cell =

        let fn_ty = mk_simple_ty_fn [| |] in
        let fn_rty =
          call_args_referent_type cx n_ty_params fn_ty (Some Il.OpaqueTy)
        in
        let args_cell = Il.Mem (based_imm fp out_mem_disp, fn_rty) in

          if defn_id_is_obj_fn_or_drop cx fnid
          then
            (*
             * To get the typarams in an obj fn, we must go to the
             * implicit obj's captured type descriptor.
             *)
            let obj_box =
              get_obj_box_from_calltup args_cell
            in
            let obj = get_element_ptr obj_box Abi.box_rc_field_body in
            let tydesc = get_element_ptr obj Abi.obj_body_elt_tydesc in
            let ty_params_ty = Ast.TY_tup (make_tydesc_tys n_ty_params) in
            let ty_params_rty = referent_type cx ty_params_ty in
            let ty_params =
              get_element_ptr (deref tydesc) Abi.tydesc_field_first_param
            in
            let ty_params =
              cell_cast ty_params (Il.ScalarTy (Il.AddrTy ty_params_rty))
            in
              deref ty_params
          else
            (*
             * Regular function --- typarams are right in the frame calltup.
             *)
            get_element_ptr args_cell Abi.calltup_elt_ty_params
  in

  let get_args_for_current_frame _ =
    let curr_args_rty =
      current_fn_args_rty (Some Il.OpaqueTy)
    in
      caller_args_cell curr_args_rty
  in

  let get_indirect_args_for_current_frame _ =
    get_element_ptr (get_args_for_current_frame ())
      Abi.calltup_elt_indirect_args
  in

  let get_iterator_args_for_current_frame _ =
    get_element_ptr (get_args_for_current_frame ())
      Abi.calltup_elt_iterator_args
  in

  let get_closure_for_current_frame _ =
    let self_indirect_args =
      get_indirect_args_for_current_frame ()
    in
      get_element_ptr self_indirect_args
        Abi.indirect_args_elt_closure
  in

  let get_iter_block_fn_for_current_frame _ =
    let self_iterator_args =
      get_iterator_args_for_current_frame ()
    in
    let blk_fn = get_element_ptr self_iterator_args
      Abi.iterator_args_elt_block_fn
    in
      cell_cast blk_fn
        (Il.ScalarTy (Il.AddrTy Il.CodeTy))
  in

  let get_iter_outer_frame_ptr_for_current_frame _ =
    let self_iterator_args =
      get_iterator_args_for_current_frame ()
    in
      get_element_ptr self_iterator_args
        Abi.iterator_args_elt_outer_frame_ptr
  in

  (*
   * Within a for-each block, calculate the fp of an enclosing for-each block
   * or the enclosing function by chasing static links.
   *)
  let get_nth_outer_frame_ptr (diff:int) : Il.cell =
    (* All for-each block frames have the same args. *)
    let block_args_rty = current_fn_args_rty None in
    let current_fp = Il.Reg (abi.Abi.abi_fp_reg, Il.AddrTy Il.OpaqueTy) in
    let rec out (n:int) (fp:Il.cell) : Il.cell =
      if n == 0
      then fp
      else
        let args = fp_to_args fp block_args_rty in
        let iter_args = get_element_ptr args Abi.calltup_elt_iterator_args in
        let outer_fp =
          get_element_ptr iter_args Abi.iterator_args_elt_outer_frame_ptr
        in
          out (n - 1) outer_fp
    in
      out diff current_fp
  in

  let curr_stmt_depth _ =
    if (Stack.is_empty curr_stmt)
    then None
    else
      Some
        (get_stmt_depth cx (Stack.top curr_stmt))
  in

  let get_ty_params_of_current_frame _ : Il.cell =
    let fnid = current_fn() in
    let n_ty_params = n_item_ty_params cx fnid in
    let local _ =
      get_ty_params_of_frame fnid abi.Abi.abi_fp_reg n_ty_params
    in
      if Hashtbl.mem cx.ctxt_block_is_loop_body fnid
      then
        begin
          let outermost_fnid = get_loop_outermost_fn cx fnid in
            match curr_stmt_depth() with
                None -> local()
              | Some depth ->
                  iflog (fun _ ->
                           annotate "loading outermost frame ty params");
                  let (outermost_fp, _) =
                    force_to_reg (Il.Cell (get_nth_outer_frame_ptr depth))
                  in
                    get_ty_params_of_frame
                      outermost_fnid
                      outermost_fp
                      (n_item_ty_params cx outermost_fnid)
        end
      else
        local()
  in

  let get_ty_param_in_current_frame (param_idx:int) : Il.cell =
    get_ty_param (get_ty_params_of_current_frame()) param_idx
  in

  let linearize_ty_params (ty:Ast.ty) : (Ast.ty * Il.operand array) =
    let htab = Hashtbl.create 0 in
    let q = Queue.create () in
    let base = ty_fold_rebuild (fun t -> t) in
    let ty_fold_param (i, mut) =
      let param = Ast.TY_param (i, mut) in
        match htab_search htab param with
            Some p -> p
          | None ->
              let p = Ast.TY_param (Hashtbl.length htab, mut) in
                htab_put htab param p;
                Queue.add (Il.Cell (get_ty_param_in_current_frame i)) q;
                p
    in
      let fold =
        { base with
            ty_fold_param = ty_fold_param; }
      in
      let ty = fold_ty cx fold ty in
        (ty, queue_to_arr q)
  in

  let has_parametric_types_cache = Hashtbl.create 0 in
  let has_parametric_types (t:Ast.ty) : bool =
    let base = ty_fold_bool_or false in
    let ty_fold_param _ =
      true
    in
    let fold = { base with ty_fold_param = ty_fold_param } in
      htab_search_or_add
        has_parametric_types_cache t
        (fun _ -> fold_ty cx fold t)
  in

  let rec calculate_sz_full (ty_params:Il.cell) (size:size) : Il.operand =
    iflog (fun _ -> annotate
             (Printf.sprintf "calculating size %s"
                (string_of_size size)));
    let sub_sz = calculate_sz_full ty_params in
    match htab_search (emitter_size_cache()) size with
        Some op ->
          iflog (fun _ -> annotate
                   (Printf.sprintf "cached size %s is %s"
                      (string_of_size size)
                      (oper_str op)));
          op

      | _ ->
          let res =
            match size with
                SIZE_fixed i -> imm i
              | SIZE_fixup_mem_pos f -> Il.Imm (Asm.M_POS f, word_ty_mach)
              | SIZE_fixup_mem_sz f -> Il.Imm (Asm.M_SZ f, word_ty_mach)

              | SIZE_param_size i ->
                  let tydesc = deref (get_ty_param ty_params i) in
                    Il.Cell (get_element_ptr tydesc Abi.tydesc_field_size)

              | SIZE_param_align i ->
                  let tydesc = deref (get_ty_param ty_params i) in
                    Il.Cell (get_element_ptr tydesc Abi.tydesc_field_align)

              | SIZE_rt_neg a ->
                  let op_a = sub_sz a in
                  let tmp = next_vreg_cell word_sty in
                    emit (Il.unary Il.NEG tmp op_a);
                    Il.Cell tmp

              | SIZE_rt_add (a, b) ->
                  let op_a = sub_sz a in
                  let op_b = sub_sz b in
                  let tmp = next_vreg_cell word_sty in
                    add tmp op_a op_b;
                    Il.Cell tmp

              | SIZE_rt_mul (a, b) ->
                  let op_a = sub_sz a in
                  let op_b = sub_sz b in
                  let tmp = next_vreg_cell word_sty in
                    emit (Il.binary Il.UMUL tmp op_a op_b);
                    Il.Cell tmp

              | SIZE_rt_max (a, b) ->
                  let op_a = sub_sz a in
                  let op_b = sub_sz b in
                  let tmp = next_vreg_cell word_sty in
                    mov tmp op_a;
                    (* 
                     * FIXME: X86-ism going via a vreg; mem op= mem doesn't
                     * work and IL lacks sufficient brains to cope just now.
                     * Instead, should be
                     * 
                     *     emit (Il.cmp op_a op_b)
                     * 
                     * Luckily this isn't the worst hack since we already
                     * needed a tmp vreg for op_a.
                     *)
                    emit (Il.cmp (Il.Cell tmp) op_b);
                    let jmp = mark () in
                      emit (Il.jmp Il.JAE Il.CodeNone);
                      mov tmp op_b;
                      patch jmp;
                      Il.Cell tmp

              | SIZE_rt_align (align, off) ->
                  (*
                   * calculate off + pad where:
                   *
                   * pad = (align - (off mod align)) mod align
                   * 
                   * In our case it's always a power of two, 
                   * so we can just do:
                   * 
                   * mask = align-1
                   * off += mask
                   * off &= ~mask
                   *
                   *)
                  annotate "fetch alignment";
                  let op_align = sub_sz align in
                    annotate "fetch offset";
                    let op_off = sub_sz off in
                    let mask = next_vreg_cell word_sty in
                    let off = next_vreg_cell word_sty in
                      mov mask op_align;
                      sub_from mask one;
                      mov off op_off;
                      add_to off (Il.Cell mask);
                      emit (Il.unary Il.NOT mask (Il.Cell mask));
                      emit (Il.binary Il.AND
                              off (Il.Cell off) (Il.Cell mask));
                      Il.Cell off
          in
            iflog (fun _ -> annotate
                     (Printf.sprintf "calculated size %s is %s"
                        (string_of_size size)
                        (oper_str res)));
            htab_put (emitter_size_cache()) size res;
            res

  and calculate_sz c s =
    in_quad_category "size calc"
      (fun _ -> calculate_sz_full c s)

  and calculate_sz_in_current_frame (size:size) : Il.operand =
    calculate_sz (get_ty_params_of_current_frame()) size

  and callee_args_cell (tail_area:bool) (args_rty:Il.referent_ty) : Il.cell =
    if tail_area
    then
      Il.Mem (sp_off_sz (current_fn_callsz ()), args_rty)
    else
      Il.Mem (sp_imm 0L, args_rty)

  and based_sz (ty_params:Il.cell) (reg:Il.reg) (size:size) : Il.mem =
    match Il.size_to_expr64 size with
        Some e -> based_off reg e
      | None ->
             let runtime_size = calculate_sz ty_params size in
             let v = next_vreg () in
             let c = (Il.Reg (v, word_sty)) in
               mov c (Il.Cell (Il.Reg (reg, word_sty)));
               add_to c runtime_size;
               based v

  and fp_off_sz (size:size) : Il.mem =
    based_sz (get_ty_params_of_current_frame()) abi.Abi.abi_fp_reg size

  and sp_off_sz (size:size) : Il.mem =
    based_sz (get_ty_params_of_current_frame()) abi.Abi.abi_sp_reg size
  in

  let ty_sz_in_current_frame (ty:Ast.ty) : Il.operand =
    let rty = referent_type cx ty in
    let sz = Il.referent_ty_size word_bits rty in
      calculate_sz_in_current_frame sz
  in

  let ty_sz_with_ty_params
      (ty_params:Il.cell)
      (ty:Ast.ty)
      : Il.operand =
    let rty = referent_type cx ty in
    let sz = Il.referent_ty_size word_bits rty in
      calculate_sz ty_params sz
  in

  let get_element_ptr_dyn
      (ty_params:Il.cell)
      (mem_cell:Il.cell)
      (i:int)
      : Il.cell =
    match mem_cell with
        Il.Mem (mem, Il.StructTy elts)
          when i >= 0 && i < (Array.length elts) ->
            assert ((Array.length elts) != 0);
            begin
              let elt_rty = elts.(i) in
              let elt_off = Il.get_element_offset word_bits elts i in
                match elt_off with
                    SIZE_fixed fixed_off ->
                      Il.Mem (Il.mem_off_imm mem fixed_off, elt_rty)
                  | sz ->
                      let sz = calculate_sz ty_params sz in
                      let v = next_vreg word_sty in
                      let vc = Il.Reg (v, word_sty) in
                        lea vc mem;
                        add_to vc sz;
                        Il.Mem (based v, elt_rty)
            end
      | Il.Mem (_, Il.StructTy elts) ->
          bug ()
            "get_element_ptr dyn %d out of bounds (len %d) on cell %s"
            i (Array.length elts) (cell_str mem_cell)
      | _ ->
          bug () "get_element_ptr_dyn %d on cell %s" i (cell_str mem_cell)
  in

  let get_element_ptr_dyn_in_current_frame
      (mem_cell:Il.cell)
      (i:int)
      : Il.cell =
    get_element_ptr_dyn (get_ty_params_of_current_frame()) mem_cell i
  in

  let deref_off_sz
      (ty_params:Il.cell)
      (ptr:Il.cell)
      (size:size)
      : Il.cell =
    match Il.size_to_expr64 size with
        Some e -> deref_off ptr e
      | None ->
          let (r,_) = force_to_reg (Il.Cell ptr) in
          let mem = based_sz ty_params r size in
            Il.Mem (mem, (pointee_type ptr))
  in


  let cell_of_block_slot
      ?access_depth:(access_depth=curr_stmt_depth())
      (slot_id:node_id)
      : Il.cell =

    let referent_type = slot_id_referent_type slot_id in

    let local_access off =
      Il.Mem (fp_off_sz off, referent_type)
    in

    let outer_access off slot_depth depth =
      let _ = assert (slot_depth < depth) in
        let _ =
          iflog
            begin
              fun _ ->
                let k =
                  Hashtbl.find cx.ctxt_slot_keys slot_id
                in
                  annotate (Printf.sprintf
                              "access outer frame slot #%d = %s"
                              (int_of_node slot_id)
                              (Fmt.fmt_to_str Ast.fmt_slot_key k))
            end
        in
        let diff = depth - slot_depth in
        let _ = annotate "get outer frame pointer" in
        let fp = get_nth_outer_frame_ptr diff in
        let _ = annotate "calculate size" in
        let p =
          based_sz (get_ty_params_of_current_frame())
            (fst (force_to_reg (Il.Cell fp))) off
        in
          Il.Mem (p, referent_type)
    in

      match htab_search cx.ctxt_slot_vregs slot_id with
          Some vr ->
            begin
              match referent_type with
                  Il.ScalarTy st -> Il.Reg (Il.Vreg (cell_vreg_num vr), st)
                | Il.NilTy -> nil_ptr
                | Il.StructTy _ -> bugi cx slot_id
                    "cannot treat structured referent as single operand"
                | Il.UnionTy _ -> bugi cx slot_id
                    "cannot treat union referent as single operand"
                | Il.ParamTy _ -> bugi cx slot_id
                    "cannot treat parametric referent as single operand"
                | Il.OpaqueTy -> bugi cx slot_id
                    "cannot treat opaque referent as single operand"
                | Il.CodeTy ->  bugi cx slot_id
                    "cannot treat code referent as single operand"
            end
        | None ->
            begin
              match htab_search cx.ctxt_slot_offsets slot_id with
                  None -> bugi cx slot_id
                    "slot assigned to neither vreg nor offset"
                | Some off ->
                    if slot_is_obj_state cx slot_id
                    then
                      begin
                        let state_arg = get_closure_for_current_frame () in
                        let (slot_mem, _) =
                          need_mem_cell (deref_off_sz
                                           (get_ty_params_of_current_frame())
                                           state_arg off)
                        in
                          Il.Mem (slot_mem, referent_type)
                      end
                    else
                      match access_depth with
                          None -> local_access off
                        | Some depth ->
                            let slot_depth = get_slot_depth cx slot_id in
                              if slot_depth <> depth
                              then
                                outer_access off slot_depth depth
                              else
                                local_access off
            end
  in

  let binop_to_jmpop (binop:Ast.binop) : Il.jmpop =
    match binop with
        Ast.BINOP_eq -> Il.JE
      | Ast.BINOP_ne -> Il.JNE
      | Ast.BINOP_lt -> Il.JL
      | Ast.BINOP_le -> Il.JLE
      | Ast.BINOP_ge -> Il.JGE
      | Ast.BINOP_gt -> Il.JG
      | _ -> bug () "Unhandled binop in binop_to_jmpop"
  in

  let get_vtbl_entry_idx (table_ptr:Il.cell) (i:int) : Il.cell =
    (* Vtbls are encoded as tables of table-relative displacements. *)
    let (table_mem, _) = need_mem_cell (deref table_ptr) in
    let disp = Il.Cell (word_at (Il.mem_off_imm table_mem (word_n i))) in
    let ptr_cell = next_vreg_cell (Il.AddrTy Il.CodeTy) in
      mov ptr_cell (Il.Cell table_ptr);
      add_to ptr_cell disp;
      ptr_cell
  in

  let get_vtbl_entry
      (obj_cell:Il.cell)
      (obj_ty:Ast.ty_obj)
      (id:Ast.ident)
      : (Il.cell * Ast.ty_fn) =
    let (_, fns) = obj_ty in
    let sorted_idents = sorted_htab_keys fns in
    let i = arr_idx sorted_idents id in
    let fn_ty = Hashtbl.find fns id in
    let table_ptr = get_element_ptr obj_cell Abi.obj_field_vtbl in
      (get_vtbl_entry_idx table_ptr i, fn_ty)
  in

  let rec trans_slot_lval_ext
      (initializing:bool)
      (base_ty:Ast.ty)
      (cell:Il.cell)
      (comp:Ast.lval_component)
      : (Il.cell * Ast.ty) =

    let bounds_checked_access at ty =
      let atop = trans_atom at in
      let unit_sz = ty_sz_in_current_frame ty in
      let idx = next_vreg_cell word_sty in
        mov idx atop;
        emit (Il.binary Il.UMUL idx (Il.Cell idx) unit_sz);
        let elt_mem = trans_bounds_check (deref cell) (Il.Cell idx) in
          (Il.Mem (elt_mem, referent_type cx ty), ty)
    in
      (* 
       * All lval components aside from explicit-deref just auto-deref
       * through all boxes to find their indexable referent.
       *)
    let base_ty = strip_mutable_or_constrained_ty base_ty in
    let (cell, base_ty) =
      if comp = Ast.COMP_deref
      then (cell, base_ty)
      else deref_ty DEREF_all_boxes initializing cell base_ty
    in

    match (base_ty, comp) with
        (Ast.TY_rec entries,
         Ast.COMP_named (Ast.COMP_ident id)) ->
          let i = arr_idx (Array.map fst entries) id in
            (get_element_ptr_dyn_in_current_frame cell i, snd entries.(i))

      | (Ast.TY_tup entries,
         Ast.COMP_named (Ast.COMP_idx i)) ->
          (get_element_ptr_dyn_in_current_frame cell i, entries.(i))

      | (Ast.TY_vec ty,
         Ast.COMP_atom at) ->
          bounds_checked_access at ty

      | (Ast.TY_str,
         Ast.COMP_atom at) ->
          bounds_checked_access at (Ast.TY_mach TY_u8)

      | (Ast.TY_obj obj_ty,
         Ast.COMP_named (Ast.COMP_ident id)) ->
          let (cell, fn_ty) = get_vtbl_entry cell obj_ty id in
            (cell, (Ast.TY_fn fn_ty))

      | (Ast.TY_box _, Ast.COMP_deref) ->
          deref_ty DEREF_one_box initializing cell base_ty

      | _ -> bug () "unhandled form of lval_ext in trans_slot_lval_ext"

  (* 
   * vec: operand holding ptr to vec.
   * mul_idx: index value * unit size.
   * return: ptr to element.
   *)
  and trans_bounds_check (vec:Il.cell) (mul_idx:Il.operand) : Il.mem =
    let (len:Il.cell) = get_element_ptr vec Abi.vec_elt_fill in
    let (data:Il.cell) = get_element_ptr vec Abi.vec_elt_data in
    let (base:Il.cell) = next_vreg_cell Il.voidptr_t in
    let (elt_reg:Il.reg) = next_vreg () in
    let (elt:Il.cell) = Il.Reg (elt_reg, Il.voidptr_t) in
    let (diff:Il.cell) = next_vreg_cell word_sty in
      annotate "bounds check";
      lea base (fst (need_mem_cell data));
      add elt (Il.Cell base) mul_idx;
      emit (Il.binary Il.SUB diff (Il.Cell elt) (Il.Cell base));
      let jmp = trans_compare_simple Il.JB (Il.Cell diff) (Il.Cell len) in
        trans_cond_fail "bounds check" jmp;
        based elt_reg

  and trans_const_atom
      (atom:Ast.atom)
      : (Ast.ty * const) =
    match atom with
        Ast.ATOM_literal lit ->
          begin
            match lit.node with
                Ast.LIT_nil -> (Ast.TY_nil, CONST_val 0L)
              | Ast.LIT_bool false -> (Ast.TY_bool, CONST_val 0L)
              | Ast.LIT_bool true -> (Ast.TY_bool, CONST_val 1L)
              | Ast.LIT_char c -> (Ast.TY_char, CONST_val (Int64.of_int c))
              | Ast.LIT_int i -> (Ast.TY_int, CONST_val i)
              | Ast.LIT_uint i -> (Ast.TY_uint, CONST_val i)
              | Ast.LIT_mach_int (m, i) -> (Ast.TY_mach m, CONST_val i)
          end

      | Ast.ATOM_lval lv ->
          trans_const_lval lv

      | Ast.ATOM_pexp _ ->
          unimpl None "constant-folding pexp atom"

  and trans_const_expr
      (expr:Ast.expr)
      : (Ast.ty * const) =
    match expr with
        Ast.EXPR_atom at -> trans_const_atom at

      | Ast.EXPR_binary (_, a, b) ->
          let _ = trans_const_atom a in
          let _ = trans_const_atom b in
            unimpl None "constant-folding binary expr"

      | Ast.EXPR_unary (_, x) ->
          let _ = trans_const_atom x in
            unimpl None "constant-folding unary expr"

  and trans_const_lval
      (lv:Ast.lval)
      : (Ast.ty * const) =
    assert (lval_base_is_item cx lv);
    let item = lval_item cx lv in
      match item.node.Ast.decl_item with
          Ast.MOD_ITEM_const (_, Some e) -> trans_const_expr e

        | Ast.MOD_ITEM_tag (hdr, _, i) when Array.length hdr = 0 ->
            (lval_ty cx lv,
             CONST_frag (Asm.WORD (word_ty_mach,
                                   Asm.IMM (Int64.of_int i))))

        | _ -> bug ()
            "trans_const_lval called on unsupported item lval '%a'"
              Ast.sprintf_lval lv

  and trans_lval_item
      (lv:Ast.lval)
      : (Il.cell * Ast.ty) =
    assert (lval_base_is_item cx lv);
    match trans_const_lval lv with

        (ty, CONST_val v) ->
          let r tm = Il.Reg (force_to_reg (imm_of_ty v tm)) in
          let f tm = (r tm, ty) in
            begin
              match ty with
                  Ast.TY_mach tm -> f tm
                | Ast.TY_uint -> f word_ty_mach
                | Ast.TY_int -> f word_ty_signed_mach
                | Ast.TY_bool -> f TY_u8
                | Ast.TY_char -> f TY_u32
                | Ast.TY_nil -> (nil_ptr, ty)

                | _ -> bug ()
                    "trans_lval_item on %a: unexpected type %a"
                      Ast.sprintf_lval lv Ast.sprintf_ty ty
            end

      | (ty, CONST_frag f) ->
          let item = lval_item cx lv in
          let ptr =
            crate_rel_to_ptr
              (trans_crate_rel_data_operand
                 (DATA_const item.id)
                 (fun _ -> f))
              (referent_type cx ty)
          in
            (deref ptr, ty)

  and trans_lval_full
      (initializing:bool)
      (lv:Ast.lval)
      : (Il.cell * Ast.ty) =

    let rec trans_slot_lval_full (initializing:bool) (outermost:bool) lv =
      let (cell, ty) =
        match lv with
            Ast.LVAL_ext (base, comp) ->
              let (base_cell, base_ty) =
                trans_slot_lval_full initializing false base
              in
                trans_slot_lval_ext initializing base_ty base_cell comp

          | Ast.LVAL_base _ ->
              let sloti = lval_base_to_slot cx lv in
              let cell = cell_of_block_slot sloti.id in
              let ty = slot_ty sloti.node in
              let cell = deref_slot initializing cell sloti.node in
                (cell, ty)
      in
      let (cell, ty) =
        if outermost
        then
          let id = lval_base_id lv in
          let dctrl =
            (* If this fails, type didn't visit the lval, and we
             * don't know whether to auto-deref the entire lval.
             * Crashing here is best. Compiler bug.
             *)
            match htab_search cx.ctxt_auto_deref_lval id with
                None ->
                  bugi cx id
                    "Lval without auto-deref info; bad typecheck?"
              | Some true -> DEREF_all_boxes
              | Some false -> DEREF_none
          in
            deref_ty dctrl initializing cell ty
        else (cell, ty)
      in
        iflog
          begin
            fun _ ->
              annotate
                (Printf.sprintf "lval %a = %s"
                   Ast.sprintf_lval lv
                   (cell_str cell))
          end;
        (cell, ty)

    in
      if lval_base_is_slot cx lv
      then trans_slot_lval_full initializing true lv
      else
        if initializing
        then err None "init item"
        else trans_lval_item lv

  and trans_lval_maybe_init
      (initializing:bool)
      (lv:Ast.lval)
      : (Il.cell * Ast.ty) =
    in_quad_category "lval"
      (fun _ -> trans_lval_full initializing lv)

  and trans_lval_init (lv:Ast.lval) : (Il.cell * Ast.ty) =
    trans_lval_maybe_init true lv

  and trans_lval (lv:Ast.lval) : (Il.cell * Ast.ty) =
    trans_lval_maybe_init false lv

  and trans_callee
      (flv:Ast.lval)
      : (Il.operand * Ast.ty) =
    (* direct call to item *)
    let fty = Hashtbl.find cx.ctxt_all_lval_types (lval_base_id flv) in
      if lval_base_is_item cx flv then
        let fn_item = lval_item cx flv in
        let fn_ptr = code_fixup_to_ptr_operand (get_fn_fixup cx fn_item.id) in
          (fn_ptr, fty)
      else
        (* indirect call to computed slot *)
        let (cell, _) = trans_lval flv in
          (Il.Cell cell, fty)

  and align x =
    Asm.ALIGN_FILE (16, Asm.ALIGN_MEM(16, x))

  and trans_crate_rel_data_operand
      (d:data)
      (thunk:unit -> Asm.frag)
      : Il.operand =
    let (fix, _) =
      htab_search_or_add cx.ctxt_data d
        begin
          fun _ ->
            let fix = new_fixup "data item" in
            let frag = align (Asm.DEF (fix, thunk())) in
              (fix, frag)
        end
    in
      crate_rel_imm fix

  and trans_crate_rel_data_frag (d:data) (thunk:unit -> Asm.frag) : Asm.frag =
    let (fix, _) =
      htab_search_or_add cx.ctxt_data d
        begin
          fun _ ->
            let fix = new_fixup "data item" in
            let frag = align (Asm.DEF (fix, thunk())) in
              (fix, frag)
        end
    in
      crate_rel_word fix

  and trans_crate_rel_static_string_operand (s:string) : Il.operand =
    trans_crate_rel_data_operand (DATA_str s) (fun _ -> Asm.ZSTRING s)

  and trans_crate_rel_static_string_frag (s:string) : Asm.frag =
    trans_crate_rel_data_frag (DATA_str s) (fun _ -> Asm.ZSTRING s)

  and trans_static_string (s:string) : Il.operand =
    Il.Cell (crate_rel_to_ptr
               (trans_crate_rel_static_string_operand s)
               (referent_type cx Ast.TY_str))

  and get_static_tydesc
      (idopt:node_id option)
      (t:Ast.ty)
      (sz:int64)
      (align:int64)
      (force_stateful:bool)
      : Il.operand =
    trans_crate_rel_data_operand
      (DATA_tydesc t)
      begin
        fun _ ->
          let tydesc_fixup = new_fixup "tydesc" in
          let fix fixup =
            fixup_rel_word tydesc_fixup fixup
          in
          let is_stateful =
            if (force_stateful || type_has_state cx t) then 1L else 0L
          in
            iflog
              (fun _ ->
                 log cx "tydesc for %a has sz=%Ld, align=%Ld, is_stateful=%Ld"
                   Ast.sprintf_ty t sz align is_stateful);
            Asm.DEF
              (tydesc_fixup,
               Asm.SEQ
                 [|
                   Asm.WORD (word_ty_mach, Asm.IMM 0L);
                   Asm.WORD (word_ty_mach, Asm.IMM sz);
                   Asm.WORD (word_ty_mach, Asm.IMM align);
                   fix (get_take_glue t);
                   fix (get_drop_glue t);
                   begin
                     match ty_mem_ctrl cx t with
                         MEM_interior ->
                           Asm.WORD (word_ty_mach, Asm.IMM 0L);
                       | _ ->
                           fix (get_free_glue t (type_has_state cx t));
                   end;

                   if cx.ctxt_sess.Session.sess_minimal
                   then Asm.WORD (word_ty_mach, Asm.IMM 0L)
                   else fix (get_sever_glue t);

                   if cx.ctxt_sess.Session.sess_minimal
                   then Asm.WORD (word_ty_mach, Asm.IMM 0L)
                   else fix (get_mark_glue t);

                   (* Include any obj-dtor, if this is an obj and has one. *)
                   begin
                     match idopt with
                         None -> Asm.WORD (word_ty_mach, Asm.IMM 0L);
                       | Some oid ->
                           begin
                             let g = GLUE_obj_drop oid in
                               match htab_search cx.ctxt_glue_code g with
                                   Some code -> fix code.code_fixup
                                 | None ->
                                     Asm.WORD (word_ty_mach, Asm.IMM 0L);
                           end
                   end;
                   Asm.WORD (word_ty_mach, Asm.IMM is_stateful);
                 |])
      end

  and get_obj_vtbl (id:node_id) : Il.operand =
    let obj =
      match Hashtbl.find cx.ctxt_all_defns id with
          DEFN_item { Ast.decl_item = Ast.MOD_ITEM_obj obj;
                      Ast.decl_params = _} -> obj
        | _ -> bug () "Trans.get_obj_vtbl on non-obj referent"
    in
      trans_crate_rel_data_operand (DATA_obj_vtbl id)
        begin
          fun _ ->
            iflog (fun _ -> log cx "emitting %d-entry obj vtbl for %s"
                     (Hashtbl.length obj.Ast.obj_fns) (path_name()));
            table_of_table_rel_fixups
              (Array.map
                 begin
                   fun k ->
                     let fn = Hashtbl.find obj.Ast.obj_fns k in
                       get_fn_fixup cx fn.id
                 end
                 (sorted_htab_keys obj.Ast.obj_fns))
        end

  and copy_loop
      (dst:Il.cell)
      (src:Il.cell)
      (sz:Il.operand)
      (elt_sz:Il.operand)
      (elt_copy:Il.cell -> Il.cell -> unit)
      : unit =
    let eltp_sty = Il.AddrTy (Il.ScalarTy (Il.ValTy Il.Bits8)) in
    let dptr = next_vreg_cell eltp_sty in
    let sptr = next_vreg_cell eltp_sty in
    let dlim = next_vreg_cell eltp_sty in
      lea dptr (fst (need_mem_cell dst));
      lea sptr (fst (need_mem_cell src));
      mov dlim (Il.Cell dptr);
      add_to dlim sz;

      let fwd_jmp = mark () in
        emit (Il.jmp Il.JMP Il.CodeNone);
        let back_jmp_targ = mark () in

          elt_copy dptr sptr;

          add_to dptr elt_sz;
          add_to sptr elt_sz;

          patch fwd_jmp;
          let back_jmp =
            trans_compare_simple Il.JB (Il.Cell dptr) (Il.Cell dlim)
          in
            List.iter
              (fun j -> patch_existing j back_jmp_targ) back_jmp;


  and trans_copy_forward_args (args_rty:Il.referent_ty) : unit =
    let caller_args_cell = caller_args_cell args_rty in
    let callee_args_cell = callee_args_cell false args_rty in
    let nbytes = Il.referent_ty_size word_bits args_rty in
    let nbytes = calculate_sz_in_current_frame nbytes in
      copy_loop callee_args_cell caller_args_cell nbytes one
        begin
          fun dptr sptr ->
            mov (deref dptr) (Il.Cell (deref sptr))
        end


  and get_forwarding_obj_fn
      (ident:Ast.ident)
      (caller:Ast.ty_obj)
      (callee:Ast.ty_obj)
      : fixup =
    (* Forwarding "glue" is not glue in the normal sense of being called with
     * only Abi.worst_case_glue_call_args args; the functions are full-fleged
     * obj fns like any other, and they perform a full call to the target
     * obj. We just use the glue facility here to store the forwarding
     * operators somewhere.
     *)
    let g = GLUE_forward (ident, caller, callee) in
    let fix = new_fixup (glue_str cx g) in
    let fty = Hashtbl.find (snd caller) ident in
    let self_args_rty =
      call_args_referent_type cx 0
        (Ast.TY_fn fty) (Some (obj_box_rty word_bits))
    in
    let callsz = Il.referent_ty_size word_bits self_args_rty in
    let spill = new_fixup "forwarding fn spill" in
      trans_glue_frame_entry callsz spill true;
      let all_self_args_cell = caller_args_cell self_args_rty in
      let self_indirect_args_cell =
        get_element_ptr all_self_args_cell Abi.calltup_elt_indirect_args
      in
        (*
         * Note: this is wrong. This assumes our closure is a vtbl,
         * when in fact it is a pointer to a refcounted malloc slab
         * containing an obj.
         *)
      let closure_cell =
        deref (get_element_ptr self_indirect_args_cell
                 Abi.indirect_args_elt_closure)
      in

      let (callee_fn_cell, _) =
        get_vtbl_entry closure_cell callee ident
      in
        iflog (fun _ -> annotate "copy args forward to callee");
        trans_copy_forward_args self_args_rty;

        iflog (fun _ -> annotate "call through to callee");
        (* FIXME (issue #80): use a tail-call here. *)
        call_code (code_of_cell callee_fn_cell);
        trans_glue_frame_exit fix spill g;
        fix


  and get_forwarding_vtbl
      (caller:Ast.ty_obj)
      (callee:Ast.ty_obj)
      : Il.operand =
    trans_crate_rel_data_operand (DATA_forwarding_vtbl (caller,callee))
      begin
        fun _ ->
          let (_,fns) = caller in
          iflog (fun _ -> log cx "emitting %d-entry obj forwarding vtbl"
                   (Hashtbl.length fns));
            table_of_table_rel_fixups
              (Array.map
                 begin
                   fun k ->
                     get_forwarding_obj_fn k caller callee
                 end
                 (sorted_htab_keys fns))
        end

  and drop_existing_if_not_init init cell ty =
    if not init
    then drop_ty_in_current_frame cell ty

  and trans_new_str
      (initializing:bool)
      (dst:Ast.lval)
      (s:string)
      (id:node_id)
      : unit =
    (* Include null byte. *)
    let init_sz = Int64.of_int ((String.length s) + 1) in
    let (dst_cell, dst_ty) = trans_lval_maybe_init initializing dst in
      drop_existing_if_not_init initializing dst_cell dst_ty;
      let ptr =
        crate_rel_to_ptr
          (trans_crate_rel_data_operand
             (DATA_const id)
             (fun _ ->
               Asm.SEQ
                 [|
                   Asm.WORD (word_ty_signed_mach,
                             Asm.IMM Abi.const_refcount);
                   Asm.WORD (word_ty_mach, Asm.IMM init_sz);
                   Asm.WORD (word_ty_mach, Asm.IMM init_sz);
                   Asm.WORD (word_ty_mach, Asm.IMM 0L);
                   Asm.ZSTRING s
                 |]))
          (referent_type cx Ast.TY_str)
      in
        mov dst_cell (Il.Cell ptr);

  and trans_lit (lit:Ast.lit) : Il.operand =
    match lit with
        Ast.LIT_nil -> Il.Cell (nil_ptr)
      | Ast.LIT_bool false -> imm_false
      | Ast.LIT_bool true -> imm_true
      | Ast.LIT_char c -> imm_of_ty (Int64.of_int c) TY_u32
      | Ast.LIT_int i -> simm i
      | Ast.LIT_uint i -> imm i
      | Ast.LIT_mach_int (m, n) -> imm_of_ty n m

  and trans_atom (atom:Ast.atom) : Il.operand =
    iflog
      begin
        fun _ ->
          annotate (Fmt.fmt_to_str Ast.fmt_atom atom)
      end;
    match atom with
        Ast.ATOM_lval lv ->
          let (cell, ty) = trans_lval lv in
            Il.Cell (fst (deref_ty DEREF_none false cell ty))

      | Ast.ATOM_literal lit -> trans_lit lit.node
      | Ast.ATOM_pexp _ -> bug () "Trans.trans_atom on ATOM_pexp"


  and fixup_to_ptr_operand
      (imm_ok:bool)
      (fix:fixup)
      (referent_ty:Il.referent_ty)
      : Il.operand =
    if imm_ok
    then Il.ImmPtr (fix, referent_ty)
    else Il.Cell (crate_rel_to_ptr (crate_rel_imm fix) referent_ty)

  and code_fixup_to_ptr_operand (fix:fixup) : Il.operand =
    fixup_to_ptr_operand abi.Abi.abi_has_pcrel_code fix Il.CodeTy

  (* A pointer-valued op may be of the form ImmPtr, which carries its
   * target fixup, "constant-propagated" through trans so that
   * pc-relative addressing can make use of it whenever
   * appropriate. Reify_ptr exists for cases when you are about to
   * store an ImmPtr into a memory cell or other place beyond which the
   * compiler will cease to know about its identity; at this point you
   * should decay it to a crate-relative displacement and
   * (computationally) add it to the crate base value, before working
   * with it.
   * 
   * This helps you obey the IL type-system prohibition against
   * 'mov'-ing an ImmPtr to a cell. If you forget to call this
   * in the right places, you will get code-generation failures.
   *)
  and reify_ptr (op:Il.operand) : Il.operand =
    match op with
        Il.ImmPtr (fix, rty) ->
          Il.Cell (crate_rel_to_ptr (crate_rel_imm fix) rty)
      | _ -> op

  and annotate_quads (name:string) : unit =
    let e = emitter() in
    let quads = emitted_quads e in
    let annotations = annotations() in
      log cx "emitted quads for %s:" name;
      for i = 0 to arr_max quads
      do
        if Hashtbl.mem annotations i
        then
          List.iter
            (fun a -> log cx "// %s" a)
            (List.rev (Hashtbl.find_all annotations i));
        log cx "[%6d]\t%s" i
          (Il.string_of_quad
             abi.Abi.abi_str_of_hardreg quads.(i));
      done


  and write_frame_info_ptrs (fnid:node_id option) =
    let frame_fns =
      match fnid with
          None -> zero
        | Some _ when cx.ctxt_sess.Session.sess_minimal -> zero
        | Some fnid -> get_frame_glue_fns fnid
    in
    let crate_ptr_reg = next_vreg () in
    let crate_ptr_cell = Il.Reg (crate_ptr_reg, (Il.AddrTy Il.OpaqueTy)) in
      iflog (fun _ -> annotate "write frame-info pointers");
      Abi.load_fixup_addr (emitter())
        crate_ptr_reg cx.ctxt_crate_fixup Il.OpaqueTy;
      mov (word_at (fp_imm frame_crate_ptr_disp)) (Il.Cell (crate_ptr_cell));
      imov (word_at (fp_imm frame_fns_disp)) frame_fns

  and check_interrupt_flag _ =
    if cx.ctxt_sess.Session.sess_minimal
    then ()
    else
      begin
        let dom = next_vreg_cell wordptr_ty in
        let flag = next_vreg_cell word_sty in
          mov dom (Il.Cell (tp_imm (word_n Abi.task_field_dom)));
          mov flag (Il.Cell (deref_imm dom
                               (word_n Abi.dom_field_interrupt_flag)));
          let null_jmp = null_check flag in
            trans_yield ();
            patch null_jmp
      end

  and trans_glue_frame_entry
      (callsz:size)
      (spill:fixup)
      (user_level:bool)
      : unit =
    (*
     * The user_level flag is true to indicate that this is glue that is
     * not called via the push/pop mechanism of trans_call_glue, and thereby
     * may (legitimately) have callsz exceeding Abi.worst_case_glue_call_args.
     *
     * Assert that the callsz is indeed no bigger than our abi's purported
     * worst-case glue args.  Moreover, the callsz should be static for non-
     * user-level glue, so we can rely on (force_sz callsz) as a preliminary
     * assertion as well.
     *)
    if not user_level
    then assert ((Int64.compare
                    (force_sz callsz)
                    (Int64.mul
                       word_sz
                       (Int64.of_int Abi.worst_case_glue_call_args))) <= 0);

    let framesz = SIZE_fixup_mem_sz spill in
      push_new_emitter_with_vregs None;
      iflog (fun _ -> annotate "prologue");
      in_native_quad_category "prologue"
        (fun _ ->
           abi.Abi.abi_emit_fn_prologue (emitter())
             framesz callsz nabi_rust (upcall_fixup "upcall_grow_task")
             false cx.ctxt_sess.Session.sess_minimal);
      (in_quad_category "prologue"
         (fun _ -> write_frame_info_ptrs None));
      (* FIXME: not clear why, but checking interrupt in glue context
       * causes many.rs to crash when run on a sufficiently large number
       * of tasks; possibly a weird interaction with growing? *)
      (* check_interrupt_flag (); *)
      iflog (fun _ -> annotate "finished prologue");

  and emitted_quads e =
    Array.sub e.Il.emit_quads 0 e.Il.emit_pc

  and capture_emitted_glue (fix:fixup) (spill:fixup) (g:glue) : unit =
    let e = emitter() in
      iflog (fun _ -> annotate_quads (glue_str cx g));
      let code = { code_fixup = fix;
                   code_quads = emitted_quads e;
                   code_vregs_and_spill = Some (Il.num_vregs e, spill); }
      in
        htab_put cx.ctxt_glue_code g code

  and trans_glue_frame_exit (fix:fixup) (spill:fixup) (g:glue) : unit =
    iflog (fun _ -> annotate "epilogue");
    abi.Abi.abi_emit_fn_epilogue (emitter());
    capture_emitted_glue fix spill g;
    pop_emitter ()

  and emit_exit_task_glue (fix:fixup) (g:glue) : unit =
    let name = glue_str cx g in
    let spill = new_fixup (name ^ " spill") in
      push_new_emitter_with_vregs None;
      (* 
       * We return-to-here in a synthetic frame we did not build; our job is
       * merely to call upcall_exit.
       *)
      iflog (fun _ -> annotate "assume 'exited' state");
      trans_void_upcall "upcall_exit" [| |];
      capture_emitted_glue fix spill g;
      pop_emitter ()

  and get_exit_task_glue _ : fixup =
    let g = GLUE_exit_task in
      match htab_search cx.ctxt_glue_code g with
          Some code -> code.code_fixup
        | None ->
            let fix = cx.ctxt_exit_task_fixup in
              emit_exit_task_glue fix g;
              fix

  (* FIXME (issue #2): this should eventually use tail calling logic *)

  and emit_fn_thunk_glue
      (n_ty_params:int)
      (arg_slots:Ast.slot array)
      (arg_bound_flags:bool array)
      (fix:fixup)
      (g:glue)
      : unit =

    let extract_slots want_bound =
      arr_filter_some
        (arr_map2
           (fun slot bound ->
              if bound = want_bound then Some slot else None)
           arg_slots
           arg_bound_flags)
    in
    let bound_slots = extract_slots true in
    let unbound_slots = extract_slots false in

    let (self_ty:Ast.ty) = mk_simple_ty_fn unbound_slots in
    let (callee_ty:Ast.ty) = mk_simple_ty_fn arg_slots in

    let self_box_rty = closure_box_rty cx n_ty_params bound_slots in

    let self_args_rty =
      call_args_referent_type cx 0 self_ty (Some self_box_rty)
    in

    let callee_args_rty =
      call_args_referent_type cx n_ty_params callee_ty (Some Il.OpaqueTy)
    in

    let callsz = Il.referent_ty_size word_bits callee_args_rty in
    let spill = new_fixup "bind glue spill" in
      trans_glue_frame_entry callsz spill true;

      let all_self_args_cell = caller_args_cell self_args_rty in

      let self_indirect_args_cell =
        get_element_ptr all_self_args_cell Abi.calltup_elt_indirect_args
      in

      let box_cell =
        deref (get_element_ptr self_indirect_args_cell
                 Abi.indirect_args_elt_closure)
      in

      let closure_cell =
        get_element_ptr box_cell Abi.box_rc_field_body
      in

      let closure_target_cell =
        get_element_ptr closure_cell Abi.closure_body_elt_target
      in

      let closure_target_code_cell =
        get_element_ptr closure_target_cell Abi.fn_field_code
      in

      let self_args_cell =
        get_element_ptr all_self_args_cell Abi.calltup_elt_args
      in

      let self_ty_params_cell =
        get_element_ptr all_self_args_cell Abi.calltup_elt_ty_params
      in

        merge_bound_args
          n_ty_params
          self_args_rty callee_args_rty
          arg_slots arg_bound_flags;
        iflog (fun _ -> annotate "call through to closure target fn");

        call_code (code_of_cell closure_target_code_cell);

        (* Drop the args we were passed. *)
        Array.iteri
          (fun i slot ->
             let cell = get_element_ptr self_args_cell i in
               drop_slot self_ty_params_cell cell slot)
          unbound_slots;

        trans_glue_frame_exit fix spill g


  and get_fn_thunk_glue
      (bind_id:node_id)
      (n_ty_params:int)
      (arg_slots:Ast.slot array)
      (arg_bound_flags:bool array)
      : fixup =
    let g = GLUE_fn_thunk bind_id in
      match htab_search cx.ctxt_glue_code g with
          Some code -> code.code_fixup
        | None ->
            let fix = new_fixup (glue_str cx g) in
              emit_fn_thunk_glue n_ty_params arg_slots arg_bound_flags fix g;
              fix


  (* 
   * Mem-glue functions are either 'mark', 'drop' or 'free', they take
   * one pointer arg and return nothing.
   *)

  and trans_mem_glue_frame_entry (n_outgoing_args:int) (spill:fixup) : unit =
    let callsz = SIZE_fixed (word_n n_outgoing_args) in
      trans_glue_frame_entry callsz spill false

  and get_mem_glue_full (g:glue) (inner:Il.mem -> unit) : fixup =
    match htab_search cx.ctxt_glue_code g with
        Some code -> code.code_fixup
      | None ->
          begin
            let name = glue_str cx g in
            let fix = new_fixup name in
              (* 
               * Put a temporary code entry in the table to handle
               * recursive emit calls during the generation of the glue
               * function.
               *)
            let tmp_code = { code_fixup = fix;
                             code_quads = [| |];
                             code_vregs_and_spill = None; } in
            let spill = new_fixup (name ^ " spill") in
              htab_put cx.ctxt_glue_code g tmp_code;
              log cx "emitting glue: %s" name;
              trans_mem_glue_frame_entry Abi.worst_case_glue_call_args spill;
              let (arg:Il.mem) = fp_imm arg0_disp in
                inner arg;
                Hashtbl.remove cx.ctxt_glue_code g;
                trans_glue_frame_exit fix spill g;
                fix
          end

  and get_mem_glue g i =
    in_quad_category "mem glue" (fun _ -> get_mem_glue_full g i)

  and get_typed_mem_glue
      (g:glue)
      (fty:Ast.ty)
      (inner:Il.cell -> Il.cell -> unit)
      : fixup =
      get_mem_glue g
        begin
          fun _ ->
            let n_ty_params = 0 in
            let calltup_rty =
              call_args_referent_type cx n_ty_params fty None
            in
            let calltup_cell = caller_args_cell calltup_rty in
            let out_cell =
              get_element_ptr calltup_cell Abi.calltup_elt_out_ptr
            in
            let args_cell =
              get_element_ptr calltup_cell Abi.calltup_elt_args
            in
              begin
                match Il.cell_referent_ty args_cell with
                    Il.StructTy az ->
                      assert ((Array.length az)
                              <= Abi.worst_case_glue_call_args);
                  | _ -> bug () "unexpected cell referent ty in glue args"
              end;
              inner out_cell args_cell
        end

  and trace_str b s =
    if b
    then
      begin
        let static = trans_static_string s in
          trans_void_upcall "upcall_trace_str" [| static |]
      end

  and trace_word b w =
    if b
    then
      trans_void_upcall "upcall_trace_word" [| Il.Cell w |]

  and ty_params_covering (t:Ast.ty) : Ast.slot =
    let n_ty_params = n_used_type_params cx t in
    let params = make_tydesc_tys n_ty_params in
      alias_slot (Ast.TY_tup params)

  and get_drop_glue
      (ty:Ast.ty)
      : fixup =
    let ty = get_genericized_ty ty in
    let g = GLUE_drop ty in
    let inner _ (args:Il.cell) =
      let ty_params = deref (get_element_ptr args 0) in
      let cell = get_element_ptr args 1 in
        note_drop_step ty "in drop-glue, dropping";
        trace_word cx.ctxt_sess.Session.sess_trace_drop cell;
        drop_ty_full true ty_params (deref cell) ty;
        note_drop_step ty "drop-glue complete";
    in
    let ty_params_ptr = ty_params_covering ty in
    let fty = mk_simple_ty_fn [| ty_params_ptr; alias_slot ty |] in
      get_typed_mem_glue g fty inner


  and get_free_glue
      (ty:Ast.ty)
      (is_gc:bool)
      : fixup =
    let ty = get_genericized_ty ty in
    let g = GLUE_free ty in
    let inner _ (args:Il.cell) =
      (* Free-glue assumes it's called with a pointer to a box allocation with
       * normal box layout. It's just a way to move drop+free out of leaf
       * code.
       *)
      let ty_params = deref (get_element_ptr args 0) in
      let cell = get_element_ptr args 1 in
        free_ty is_gc ty_params ty cell
    in
    let ty_params_ptr = ty_params_covering ty in
    let fty = mk_simple_ty_fn [| ty_params_ptr; local_slot ty |] in
      get_typed_mem_glue g fty inner


  and get_sever_glue
      (ty:Ast.ty)
      : fixup =
    let ty = get_genericized_ty ty in
    let g = GLUE_sever ty in
    let inner _ (args:Il.cell) =
      let ty_params = deref (get_element_ptr args 0) in
      let cell = get_element_ptr args 1 in
        note_gc_step ty "in sever-glue, severing";
        sever_ty ty_params (deref cell) ty;
        note_gc_step ty "in sever-glue complete";
    in
    let ty_params_ptr = ty_params_covering ty in
    let fty = mk_simple_ty_fn [| ty_params_ptr; alias_slot ty |] in
      get_typed_mem_glue g fty inner


  and get_mark_glue
      (ty:Ast.ty)
      : fixup =
    let ty = get_genericized_ty ty in
    let g = GLUE_mark ty in
    let inner _ (args:Il.cell) =
      let ty_params = deref (get_element_ptr args 0) in
      let cell = get_element_ptr args 1 in
        note_gc_step ty "in mark-glue, marking";
        mark_ty ty_params (deref cell) ty;
        note_gc_step ty "mark-glue complete";
    in
    let ty_params_ptr = ty_params_covering ty in
    let fty = mk_simple_ty_fn [| ty_params_ptr; alias_slot ty |] in
      get_typed_mem_glue g fty inner


  and get_clone_glue
      (ty:Ast.ty)
      : fixup =
    let ty = get_genericized_ty ty in
    let g = GLUE_clone ty in
    let inner (out_ptr:Il.cell) (args:Il.cell) =
      let dst = deref out_ptr in
      let ty_params = deref (get_element_ptr args 0) in
      let src = deref (get_element_ptr args 1) in
      let clone_task = get_element_ptr args 2 in
        clone_ty ty_params clone_task dst src ty
    in
    let ty_params_ptr = ty_params_covering ty in
    let fty =
      mk_ty_fn
        (local_slot ty)        (* dst *)
        [|
          ty_params_ptr;
          alias_slot ty;       (* src *)
          word_slot            (* clone-task *)
        |]
    in
      get_typed_mem_glue g fty inner


  and get_take_glue
      (ty:Ast.ty)
      : fixup =
    let ty = get_genericized_ty ty in
    let arg_ty_params_alias = 0 in
    let arg_src_alias = 1 in

    let g = GLUE_take ty in
    let inner (_:Il.cell) (args:Il.cell) =
      let ty_params = deref (get_element_ptr args arg_ty_params_alias) in
      let src = deref (get_element_ptr args arg_src_alias) in
        trans_take_ty true ty_params src ty;
    in
    let ty_params_ptr = ty_params_covering ty in
    let fty =
      mk_ty_fn
        (local_slot ty)
        [| ty_params_ptr; alias_slot ty; word_slot |]
    in
      get_typed_mem_glue g fty inner

  and get_cmp_glue ty =
    let ty = get_genericized_ty ty in
    let arg_ty_params_alias = 0 in
    let arg_lhs_alias = 1 in
    let arg_rhs_alias = 2 in
    let g = GLUE_cmp ty in
    let inner (out_ptr:Il.cell) (args:Il.cell) =
      let dst = deref out_ptr in
      let ty_params = deref (get_element_ptr args arg_ty_params_alias) in
      let lhs = deref (get_element_ptr args arg_lhs_alias) in
      let rhs = deref (get_element_ptr args arg_rhs_alias) in
      let early_finish_jmps = Queue.create () in
      let cmp_part lhs rhs ty =
        let tmp = trans_cmp ~ty_params ~ty (Il.Cell lhs) (Il.Cell rhs) in
        let keep_going_jmps =
          trans_compare_simple Il.JE tmp zero
        in
          mov dst tmp;
          Queue.add (mark()) early_finish_jmps;
          emit (Il.jmp Il.JMP Il.CodeNone);
          List.iter patch keep_going_jmps
      in
        mov dst zero;
        iter_ty_parts_full ty_params lhs rhs ty cmp_part;
        Queue.iter patch early_finish_jmps;
    in
    let ty_params_ptr = ty_params_covering ty in
    let fty =
      mk_ty_fn
        (local_slot Ast.TY_int)
        [| ty_params_ptr; alias_slot ty; alias_slot ty |]
    in
      get_typed_mem_glue g fty inner

  (*
   * Vector-growth glue takes the following arguments:
   *
   *   0. (Implicit) task ptr
   *   1. Pointer to the typarams of the caller's frame (possibly required to
   *      be passed to element's copy glue).
   *   2. Pointer to the tydesc of the vec, so that we can tell if it's gc
   *      mem, and have a tydesc to pass to malloc if we're allocating anew.
   *   3. Pointer to tydesc of the vec's stored element type, so that elements
   *      can be copied to a newly alloc'ed vec if one must be created.
   *   4. Alias to vec that needs to grow (i.e. ptr to ptr to rust_vec).
   *   5. Number of bytes of growth requested
   *)
  and emit_vec_grow_glue
      (fix:fixup)
      (g:glue)
      : unit =
    let arg_typarams_ptr = 0 in
    let arg_vec_tydesc_ptr = 1 in
    let arg_elt_tydesc_ptr = 2 in
    let arg_vec_alias = 3 in
    let arg_nbytes = 4 in

    let name = glue_str cx g in
      log cx "emitting glue: %s" name;

      let fn_ty =
        mk_simple_ty_fn
          [| ty_params_covering Ast.TY_int;      (* an OK lie *)
             local_slot Ast.TY_type;
             local_slot Ast.TY_type;
             alias_slot (Ast.TY_vec Ast.TY_int); (* an OK lie *)
             local_slot Ast.TY_uint |]
      in

      let args_rty = call_args_referent_type cx 0 fn_ty None in

      let callsz = Il.referent_ty_size word_bits args_rty in
      let spill = new_fixup (name ^ " spill") in
        trans_glue_frame_entry callsz spill false;

        let args_cell =
          get_element_ptr (caller_args_cell args_rty) Abi.calltup_elt_args
        in

        let vec_alias_cell = get_element_ptr args_cell arg_vec_alias in
        let vec_cell = deref vec_alias_cell in
        let nbytes_cell = get_element_ptr args_cell arg_nbytes in
        let vec_td_ptr_cell = get_element_ptr args_cell arg_vec_tydesc_ptr in
        let elt_td_ptr_cell = get_element_ptr args_cell arg_elt_tydesc_ptr in
        let ty_params_cell =
          deref (get_element_ptr args_cell arg_typarams_ptr)
        in

        let need_copy_cell = next_vreg_cell word_sty in
        let new_vec_cell = next_vreg_cell (vec_sty word_bits) in

          aliasing true need_copy_cell
            begin
              fun need_copy_alias_cell ->
                trans_upcall "upcall_vec_grow"
                  new_vec_cell
                  [| Il.Cell vec_cell;
                     Il.Cell nbytes_cell;
                     Il.Cell need_copy_alias_cell;
                     Il.Cell vec_td_ptr_cell; |]
            end;

          let no_copy_jmps =
            trans_compare_simple Il.JE (Il.Cell need_copy_cell) zero
          in

            let dst_vec = deref new_vec_cell in
            let src_vec = deref vec_cell in

            let fill =
              get_element_ptr_dyn ty_params_cell src_vec Abi.vec_elt_fill
            in
            let elt_sz =
              get_element_ptr (deref elt_td_ptr_cell) Abi.tydesc_field_size
            in

            let dst_buf =
              get_element_ptr_dyn ty_params_cell dst_vec Abi.vec_elt_data
            in
            let src_buf =
              get_element_ptr_dyn ty_params_cell src_vec Abi.vec_elt_data
            in

            let ty_params_ptr =
              get_tydesc_params ty_params_cell elt_td_ptr_cell
            in

              (* Take all *)
              copy_loop dst_buf src_buf (Il.Cell fill) (Il.Cell elt_sz)
                begin
                  fun _ sptr ->
                    trans_call_dynamic_glue
                      elt_td_ptr_cell
                      Abi.tydesc_field_take_glue
                      None
                      [| ty_params_ptr; sptr |]
                      None;
                end;

              (* Memcpy all *)
              copy_loop dst_buf src_buf (Il.Cell fill) one
                begin
                  fun dptr sptr ->
                    mov (deref dptr) (Il.Cell (deref sptr))
                end;

              (* Set the new vec's fill to the original vec's fill *)
              let dst_fill = get_element_ptr dst_vec Abi.vec_elt_fill in
              let v = next_vreg_cell word_sty in
                mov v (Il.Cell fill);
                mov dst_fill (Il.Cell v);

                List.iter patch no_copy_jmps;

                mov vec_cell (Il.Cell new_vec_cell);

                trans_glue_frame_exit fix spill g


  and get_vec_grow_glue _
      : fixup =
    let g = GLUE_vec_grow in
    match htab_search cx.ctxt_glue_code g with
        Some code -> code.code_fixup
      | None ->
          begin
            let fix = new_fixup (glue_str cx g) in
              emit_vec_grow_glue fix g;
              fix
          end

  (* Glue functions use mostly the same calling convention as ordinary
   * functions.
   * 
   * Each glue function expects its own particular arguments, which are
   * usually aliases-- ie, caller doesn't transfer ownership to the
   * glue. And nothing is represented in terms of AST nodes. So we
   * don't do lvals-and-atoms here.
   *)

  and trans_call_glue
      (code:Il.code)
      (dst:Il.cell option)
      (args:Il.cell array)
      (clo:Il.cell option)
      : unit =
    let inner dst cloptr =
        for i = ((Array.length args) - 1) downto 0
        do
          emit (Il.Push (Il.Cell args.(i)))
        done;
        emit (Il.Push cloptr);
        emit (Il.Push (Il.Cell abi.Abi.abi_tp_cell));
        emit (Il.Push dst);
        call_code code;
        add_to (Il.Reg (abi.Abi.abi_sp_reg, word_sty))
          (imm (Int64.mul word_sz (Int64.of_int (3 + (Array.length args)))));
    in
    let cloptr =
      match clo with
          None -> zero
        | Some cloptr -> Il.Cell cloptr
    in
      match dst with
          None -> inner zero cloptr
        | Some dst ->
            aliasing true dst (fun dst -> inner (Il.Cell dst) cloptr)

  and trans_call_static_glue
      (callee:Il.operand)
      (dst:Il.cell option)
      (args:Il.cell array)
      (clo:Il.cell option)
      : unit =
    trans_call_glue (code_of_operand callee) dst args clo

  and trans_call_dynamic_glue
      (tydesc:Il.cell)
      (idx:int)
      (dst:Il.cell option)
      (args:Il.cell array)
      (clo:Il.cell option)
      : unit =
    let fptr = get_vtbl_entry_idx tydesc idx in
      trans_call_glue (code_of_operand (Il.Cell fptr)) dst args clo

  and trans_call_simple_static_glue
      (fix:fixup)
      (ty_params:Il.cell)
      (args:Il.cell array)
      (clo:Il.cell option)
      : unit =
    trans_call_static_glue
      (code_fixup_to_ptr_operand fix)
      None
      (Array.append [| alias ty_params |] args)
      clo

  and get_tydesc_params
      (outer_ty_params:Il.cell)
      (td:Il.cell)
      : Il.cell =
    let first_param =
      get_element_ptr (deref td) Abi.tydesc_field_first_param
    in
    let res = next_vreg_cell Il.voidptr_t in
      mov res (Il.Cell (alias outer_ty_params));
      emit (Il.cmp (Il.Cell first_param) zero);
      let no_param_jmp = mark() in
        emit (Il.jmp Il.JE Il.CodeNone);
        mov res (Il.Cell first_param);
        patch no_param_jmp;
        res

  and trans_call_simple_dynamic_glue
      (ty_param:int)
      (vtbl_idx:int)
      (ty_params:Il.cell)
      (args:Il.cell array)
      (clo:Il.cell option)
      : unit =
    iflog (fun _ ->
             annotate (Printf.sprintf "calling tydesc[%d].glue[%d]"
                         ty_param vtbl_idx));
    let td = get_ty_param ty_params ty_param in
    let ty_params_ptr = get_tydesc_params ty_params td in
      trans_call_dynamic_glue
        td
        vtbl_idx
        None
        (Array.append [| ty_params_ptr |] args)
        clo

  (*
   * NB: there are 2 categories of comparisons:
   *
   *   - Those called 'compare' that take a jmpop and return a jump list
   *     that the caller should patch.
   *
   *   - Those called 'cmp' that return a number, -1/0/1, indicating the
   *     relative order of lhs and rhs.
   *
   * While in theory compare could be built out of cmp, on real machines
   * we are forced to build cmp out of compare.
   *)


  (* 
   * [trans_cmp] returns the result-code of a three-value comparison, 
   * which is an operand representing the ordering of lhs and rhs. -1 means
   * less than, 0 means equal, 1 means greater-than.
   *
   * We assume that the LHS and RHS of the comparison have the same type, an
   * invariant that the typechecker enforces.
   *)
  and trans_cmp
      ~ty_params:(ty_params:Il.cell)
      ~ty:(ty:Ast.ty)
      (lhs:Il.operand)
      (rhs:Il.operand)
      : Il.operand =
    let ty = strip_mutable_or_constrained_ty ty in
    let (result:Il.cell) = next_vreg_cell (Il.ValTy Il.Bits32) in
      begin
        match ty with

            Ast.TY_bool
          | Ast.TY_mach _
          | Ast.TY_int
          | Ast.TY_uint
          | Ast.TY_char ->
              let cjmp =
                if type_is_unsigned_2s_complement ty
                then Il.JB
                else Il.JL
              in
                (* Start with assumption lhs < rhs *)
                imov result neg_one;
                let lhs_lt_rhs_jmps =
                  trans_compare ~ty_params ~cjmp ~ty lhs rhs
                in
                  (* ... disproven, so assume lhs > rhs *)
                  mov result one;
                  let rhs_lt_lhs_jmps =
                    trans_compare ~ty_params ~cjmp ~ty rhs lhs
                  in
                    (* ... disproven, must be lhs == rhs *)
                    mov result zero;
                    List.iter patch lhs_lt_rhs_jmps;
                    List.iter patch rhs_lt_lhs_jmps;

          | Ast.TY_obj _ ->
              let lhs = need_cell lhs in
              let rhs = need_cell rhs in
              let lhs_binding = get_element_ptr lhs Abi.obj_field_box in
              let rhs_binding = get_element_ptr rhs Abi.obj_field_box in
              let lhs_box, rhs_box = deref lhs_binding, deref rhs_binding in
              let lhs_obj = get_element_ptr lhs_box Abi.box_rc_field_body in
              let rhs_obj = get_element_ptr rhs_box Abi.box_rc_field_body in
              let td = get_element_ptr lhs_obj Abi.obj_body_elt_tydesc in
              let lhs_body =
                get_element_ptr lhs_obj Abi.obj_body_elt_fields
              in
              let rhs_body =
                get_element_ptr rhs_obj Abi.obj_body_elt_fields
              in
              let ty_params_ptr = get_tydesc_params ty_params td in
                trans_call_dynamic_glue
                  td Abi.tydesc_field_cmp_glue
                  (Some result)
                  [| ty_params_ptr; alias lhs_body; alias rhs_body |]
                  None

          | Ast.TY_param (i, _) ->
              let lhs = need_cell lhs in
              let rhs = need_cell rhs in
              let td = get_ty_param ty_params i in
              let ty_params_ptr = get_tydesc_params ty_params td in
                trans_call_dynamic_glue
                  td Abi.tydesc_field_cmp_glue
                  (Some result)
                  [| ty_params_ptr; alias lhs; alias rhs |]
                  None

          | Ast.TY_vec _
          | Ast.TY_str ->
              (* FIXME: temporary until we get sequence-compares working. *)
              mov result zero;

          | _ ->
              let lhs = need_cell lhs in
              let rhs = need_cell rhs in
                trans_call_static_glue
                  (code_fixup_to_ptr_operand (get_cmp_glue ty))
                  (Some result)
                  [| alias ty_params; alias lhs; alias rhs |]
                  None
      end;
      Il.Cell result


  (*
   * [trans_compare_simple] returns a set of jump addresses, which the
   * caller patches to the destination. Only use this function if you are sure
   * that the LHS and RHS have the same type and that both will fit in a
   * machine register; otherwise, use [trans_compare] instead.
   *)
  and trans_compare_simple
      (cjmp:Il.jmpop)
      (lhs:Il.operand)
      (rhs:Il.operand)
      : quad_idx list =
    emit (Il.cmp (Il.Cell (Il.Reg (force_to_reg lhs))) rhs);
    let jmp = mark() in
      emit (Il.jmp cjmp Il.CodeNone);
      [ jmp ]

  (*
   * [trans_compare] returns a set of jump addresses, which the
   * caller patches to the destination.
   *)
  and trans_compare
      ?ty_params:(ty_params=get_ty_params_of_current_frame())
      ~cjmp:(cjmp:Il.jmpop)
      ~ty:(ty:Ast.ty)
      (lhs:Il.operand)
      (rhs:Il.operand)
      : quad_idx list =
      match ty with
          Ast.TY_bool
        | Ast.TY_mach _
        | Ast.TY_int
        | Ast.TY_uint
        | Ast.TY_char ->
            trans_compare_simple cjmp lhs rhs

        | _ ->
            let result =
              trans_cmp ~ty_params ~ty lhs rhs
            in
              emit (Il.cmp result zero);
              let jmp = mark() in
                emit (Il.jmp cjmp Il.CodeNone);
                [ jmp ]


  and trans_cond (invert:bool) (expr:Ast.expr) : quad_idx list =
    let anno _ =
      iflog
        begin
          fun _ ->
            annotate ((Fmt.fmt_to_str Ast.fmt_expr expr) ^
                        ": cond, finale")
        end
    in
      match expr with
          Ast.EXPR_binary (binop, a, b) ->
            let lhs = trans_atom a in
            let rhs = trans_atom b in
            let cjmp = binop_to_jmpop binop in
            let cjmp =
              if invert then
                match cjmp with
                    Il.JE -> Il.JNE
                  | Il.JNE -> Il.JE
                  | Il.JL -> Il.JGE
                  | Il.JLE -> Il.JG
                  | Il.JGE -> Il.JL
                  | Il.JG -> Il.JLE
                  | _ -> bug () "Unhandled inverse binop in trans_cond"
              else
                cjmp
            in
              anno ();
              let ty = atom_type cx a in
                trans_compare ~cjmp ~ty lhs rhs

      | _ ->
          let bool_operand = trans_expr expr in
            anno ();
            trans_compare_simple Il.JNE bool_operand
              (if invert then imm_true else imm_false)

  and trans_binop (binop:Ast.binop) : Il.binop =
    match binop with
        Ast.BINOP_or -> Il.OR
      | Ast.BINOP_and -> Il.AND
      | Ast.BINOP_xor -> Il.XOR

      | Ast.BINOP_lsl -> Il.LSL
      | Ast.BINOP_lsr -> Il.LSR
      | Ast.BINOP_asr -> Il.ASR

      | Ast.BINOP_add -> Il.ADD
      | Ast.BINOP_sub -> Il.SUB

      (* FIXME (issue #57):
       * switch on type of operands, IMUL/IDIV/IMOD etc.
       *)
      | Ast.BINOP_mul -> Il.UMUL
      | Ast.BINOP_div -> Il.UDIV
      | Ast.BINOP_mod -> Il.UMOD
      | _ -> bug () "bad binop to Trans.trans_binop"

  and trans_binary
      (binop:Ast.binop)
      (lhs:Il.operand)
      (rhs:Il.operand) : Il.operand =
    let arith op =
      let bits = Il.operand_bits word_bits lhs in
      let dst = Il.Reg (Il.next_vreg (emitter()), Il.ValTy bits) in
        emit (Il.binary op dst lhs rhs);
        Il.Cell dst
    in
    match binop with
        Ast.BINOP_or | Ast.BINOP_and | Ast.BINOP_xor
      | Ast.BINOP_lsl | Ast.BINOP_lsr | Ast.BINOP_asr
      | Ast.BINOP_add | Ast.BINOP_sub
      (* FIXME (issue #57):
       * switch on type of operands, IMUL/IDIV/IMOD etc.
       *)
      | Ast.BINOP_mul | Ast.BINOP_div | Ast.BINOP_mod ->
          arith (trans_binop binop)

      | _ -> let dst = Il.Reg (Il.next_vreg (emitter()), Il.ValTy Il.Bits8) in
          mov dst imm_true;
          let jmps = trans_compare_simple (binop_to_jmpop binop) lhs rhs in
            mov dst imm_false;
            List.iter patch jmps;
            Il.Cell dst


  and trans_expr (expr:Ast.expr) : Il.operand =

    let anno _ =
      iflog
        begin
          fun _ ->
            annotate ((Fmt.fmt_to_str Ast.fmt_expr expr) ^
                        ": plain exit, finale")
        end
    in
      match expr with
          Ast.EXPR_binary (binop, a, b) ->
            if not (is_prim_type (simplified_ty (atom_type cx a))) ||
                not (is_prim_type (simplified_ty (atom_type cx b))) then
              unimpl None "application of binary operator %a to operands of \
                type %s and %s"
                Ast.sprintf_binop binop
                (pretty_ty_str cx (Ast.sprintf_ty ()) (atom_type cx a))
                (pretty_ty_str cx (Ast.sprintf_ty ()) (atom_type cx b));
            trans_binary binop (trans_atom a) (trans_atom b)

        | Ast.EXPR_unary (unop, a) ->
            assert (is_prim_type (simplified_ty (atom_type cx a)));
            let src = trans_atom a in
            let bits = Il.operand_bits word_bits src in
            let dst = Il.Reg (Il.next_vreg (emitter()), Il.ValTy bits) in
            let op = match unop with
                Ast.UNOP_not
              | Ast.UNOP_bitnot -> Il.NOT
              | Ast.UNOP_neg -> Il.NEG
              | Ast.UNOP_cast t ->
                  let t = Hashtbl.find cx.ctxt_all_cast_types t.id in
                  let at = atom_type cx a in
                  let (t, at) = (simplified_ty t, simplified_ty at) in
                    if (type_is_2s_complement at) &&
                      (type_is_2s_complement t)
                    then
                      if type_is_unsigned_2s_complement t
                      then Il.UMOV
                      else Il.IMOV
                    else
                      err None "unsupported cast operator"
            in
              anno ();
              emit (Il.unary op dst src);
              (* Insist the bool domain being 0x0 and 0x1 *)
              if unop = Ast.UNOP_not
              then trans_binary Ast.BINOP_and (Il.Cell dst) one
              else Il.Cell dst

        | Ast.EXPR_atom a ->
            trans_atom a

  and drop_slot_by_id (depth:int) (slot_id:node_id) : unit =
    let slot = get_slot cx slot_id in
    let k = Hashtbl.find cx.ctxt_slot_keys slot_id in
      iflog (fun _ ->
               annotate
                 (Printf.sprintf
                    "drop_slot %d = %s "
                    (int_of_node slot_id)
                    (Fmt.fmt_to_str Ast.fmt_slot_key k)));
      drop_slot_in_current_frame
        (cell_of_block_slot
           ~access_depth:(Some depth) slot_id) slot

  and drop_slots_after_block bid : unit =
    let depth = Hashtbl.find cx.ctxt_block_loop_depths bid in
    match htab_search cx.ctxt_post_block_slot_drops bid with
        None -> ()
      | Some slots -> List.iter (drop_slot_by_id depth) slots

  and trans_block (block:Ast.block) : unit =
    flush_emitter_size_cache();
    trace_str cx.ctxt_sess.Session.sess_trace_block
      "entering block";
    emit (Il.Enter (Hashtbl.find cx.ctxt_block_fixups block.id));
    Array.iter trans_stmt block.node;
    drop_slots_after_block block.id;
    trace_str cx.ctxt_sess.Session.sess_trace_block
      "exiting block";
    emit Il.Leave;
    trace_str cx.ctxt_sess.Session.sess_trace_block
      "exited block";

  and upcall_fixup (name:string) : fixup =
    Semant.require_native cx REQUIRED_LIB_rustrt name;

  and trans_upcall
      (name:string)
      (ret:Il.cell)
      (args:Il.operand array)
      : unit =
    in_native_quad_category "upcall"
      (fun _ ->
         abi.Abi.abi_emit_native_call (emitter())
           ret nabi_rust (upcall_fixup name) args)

  and trans_void_upcall
      (name:string)
      (args:Il.operand array)
      : unit =
    in_native_quad_category "upcall"
      (fun _ ->
         abi.Abi.abi_emit_native_void_call (emitter())
           nabi_rust (upcall_fixup name) args);

  and trans_log_int lev (a:Ast.atom) : unit =
    trans_void_upcall "upcall_log_int_rustboot" [| simm (Int64.of_int lev);
                                                   trans_atom a |]

  and trans_log_str lev (a:Ast.atom) : unit =
    trans_void_upcall "upcall_log_str_rustboot" [| simm (Int64.of_int lev);
                                                   trans_atom a |]

  and trans_spawn
      ((*initializing*)_:bool)
      (dst:Ast.lval)
      (domain:Ast.domain)
      (name:string)
      (fn_lval:Ast.lval)
      (args:Ast.atom array)
      : unit =
    let (task_cell, _) = trans_lval_init dst in
    let runtime_name = trans_static_string name in
    let (fptr_operand, fn_ty) = trans_callee fn_lval in
    (*let fn_ty_params = [| |] in*)
    let _ =
      (* FIXME (issue #82): handle indirect-spawns (clone closure). *)
      if not (lval_is_direct_fn cx fn_lval)
      then bug () "unhandled indirect-spawn"
    in
    let args_rty = call_args_referent_type cx 0 fn_ty None in
    let fptr_operand = reify_ptr fptr_operand in
    let exit_task_glue_fixup = get_exit_task_glue () in
    let callsz =
      calculate_sz_in_current_frame (Il.referent_ty_size word_bits args_rty)
    in
    let exit_task_glue_fptr =
      code_fixup_to_ptr_operand exit_task_glue_fixup
    in
    let exit_task_glue_fptr = reify_ptr exit_task_glue_fptr in

      iflog (fun _ -> annotate "spawn task: copy args");

      let new_task = next_vreg_cell Il.voidptr_t in
      let call = { call_ctrl = CALL_indirect;
                   call_callee_ptr = fptr_operand;
                   call_callee_ty = fn_ty;
                   call_callee_ty_params = [| |];
                   call_output = task_cell;
                   call_args = args;
                   call_iterator_args = [| |];
                   call_indirect_args = [| |] }
      in
        match domain with
            Ast.DOMAIN_thread ->
              begin
                trans_upcall "upcall_new_thread" new_task [| runtime_name |];
                copy_fn_args false true (CLONE_all new_task) call;
                trans_upcall "upcall_start_thread" task_cell
                  [|
                    Il.Cell new_task;
                    exit_task_glue_fptr;
                     (imm (Int64.of_int abi.Abi.abi_tag));
                    fptr_operand;
                    callsz
                  |];
            end
         | _ ->
             begin
                 trans_upcall "upcall_new_task" new_task [| runtime_name |];
                 copy_fn_args false true (CLONE_chan new_task) call;
                 trans_upcall "upcall_start_task" task_cell
                   [|
                     Il.Cell new_task;
                     exit_task_glue_fptr;
                     (imm (Int64.of_int abi.Abi.abi_tag));
                     fptr_operand;
                     callsz
                   |];
             end;
      ()

  and get_curr_span _ =
      if Stack.is_empty curr_stmt
      then ("<none>", 0, 0)
      else
        let stmt_id = Stack.top curr_stmt in
          match (Session.get_span cx.ctxt_sess stmt_id) with
              None -> ("<none>", 0, 0)
            | Some sp -> sp.lo

  and trans_cond_fail (str:string) (fwd_jmps:quad_idx list) : unit =
    let (filename, line, _) = get_curr_span () in
      iflog (fun _ -> annotate ("condition-fail: " ^ str));
      trans_void_upcall "upcall_fail"
        [|
          trans_static_string str;
          trans_static_string filename;
          imm (Int64.of_int line)
        |];
      List.iter patch fwd_jmps

  and trans_check_expr (id:node_id) (e:Ast.expr) : unit =
    match simplified_ty (expr_type cx e) with
        Ast.TY_bool ->
          let fwd_jmps = trans_cond false e in
            trans_cond_fail (Fmt.fmt_to_str Ast.fmt_expr e) fwd_jmps
      | _ -> bugi cx id "check expr on non-bool"

  and trans_malloc
      (dst:Il.cell)
      (nbytes:Il.operand)
      (gc_ctrl_word:Il.operand)
      : unit =
    trans_upcall "upcall_malloc" dst [| nbytes; gc_ctrl_word |]

  and trans_free (src:Il.cell) (is_gc:bool) : unit =
    let is_gc = if is_gc then 1L else 0L in
      trans_void_upcall "upcall_free" [| Il.Cell src; imm is_gc |]

  and trans_yield () : unit =
    trans_void_upcall "upcall_yield" [| |];

  and trans_fail () : unit =
    let (filename, line, _) = get_curr_span () in
      trans_void_upcall "upcall_fail"
        [|
          trans_static_string "explicit failure";
          trans_static_string filename;
          imm (Int64.of_int line)
        |];

  and trans_join (task:Ast.lval) : unit =
    trans_void_upcall "upcall_join" [| trans_atom (Ast.ATOM_lval task) |]

  and trans_send (chan:Ast.lval) (src:Ast.lval) : unit =
    let (src_cell, src_ty) = trans_lval src in
      begin
        match (ty_mem_ctrl cx src_ty) with
          | MEM_rc_opaque
          | MEM_rc_struct
          | MEM_gc ->
              iflog (fun _ -> annotate "incr_refcount of src obj");
              incr_refcount src_cell;
          | _ -> ()
      end;
      aliasing false src_cell
        begin
          fun src_alias ->
            trans_void_upcall "upcall_send"
              [| trans_atom (Ast.ATOM_lval chan);
                 Il.Cell src_alias |];
        end

  and trans_recv (initializing:bool) (dst:Ast.lval) (chan:Ast.lval) : unit =
    let (dstcell, _) = trans_lval_maybe_init initializing dst in
      aliasing true dstcell
        begin
          fun dst_alias ->
            trans_void_upcall "upcall_recv"
              [| Il.Cell dst_alias;
                 trans_atom (Ast.ATOM_lval chan) |];
        end

  and trans_new_port (initializing:bool) (dst:Ast.lval) : unit =
    let (dst_cell, dst_ty) = trans_lval_maybe_init initializing dst in
    let unit_ty = match dst_ty with
        Ast.TY_port t -> t
      | _ -> bug () "init dst of port-init has non-port type"
    in
    let unit_sz = ty_sz cx unit_ty in
      drop_existing_if_not_init initializing dst_cell dst_ty;
      trans_upcall "upcall_new_port" dst_cell [| imm unit_sz |]

  and trans_del_port (port:Il.cell) : unit =
    trans_void_upcall "upcall_del_port" [| Il.Cell port |]

  and trans_new_chan
      (initializing:bool)
      (dst:Ast.lval)
      (port:Ast.lval)
      : unit =
    let (dst_cell, dst_ty) = trans_lval_maybe_init initializing dst in
      drop_existing_if_not_init initializing dst_cell dst_ty;
      trans_upcall "upcall_new_chan" dst_cell
        [| trans_atom (Ast.ATOM_lval port) |]

  and trans_del_chan (chan:Il.cell) : unit =
    trans_void_upcall "upcall_del_chan" [| Il.Cell chan |]

  and trans_kill_task (task:Il.cell) : unit =
    trans_void_upcall "upcall_kill" [| Il.Cell task |]

  (*
   * A vec is implicitly boxed: every slot vec[T] is 1 word and
   * points to a refcounted structure. That structure has 4 words with
   * defined meaning at the beginning; data follows the header.
   *
   *   word 0: refcount or gc control word
   *   word 1: allocated size of data
   *   word 2: initialised size of data
   *   word 3: padding word to hit even multiple of 16
   *   word 4...N: data
   * 
   * This 4-word prefix is shared with strings, we factor the common
   * part out for reuse in string code.
   *)

  and trans_new_vec
      (initializing:bool)
      (dst:Ast.lval)
      (atoms:Ast.atom array)
      : unit =
    let (dst_cell, dst_ty) = trans_lval_maybe_init initializing dst in
    let gc_ctrl =
      if (ty_mem_ctrl cx dst_ty) = MEM_gc
      then Il.Cell (get_tydesc None dst_ty)
      else zero
    in
    let unit_ty = match dst_ty with
        Ast.TY_vec t -> t
      | _ -> bug () "init dst of vec-init has non-vec type"
    in
    let fill = next_vreg_cell word_sty in
    let unit_sz = ty_sz_in_current_frame unit_ty in
      drop_existing_if_not_init initializing dst_cell dst_ty;
      umul fill unit_sz (imm (Int64.of_int (Array.length atoms)));
      trans_upcall "upcall_new_vec" dst_cell [| Il.Cell fill; gc_ctrl |];
      let vec = deref dst_cell in
        let body_mem =
          fst (need_mem_cell
                 (get_element_ptr_dyn_in_current_frame
                    vec Abi.vec_elt_data))
        in
        let unit_rty = referent_type cx unit_ty in
        let body_rty = Il.StructTy (Array.map (fun _ -> unit_rty) atoms) in
        let body = Il.Mem (body_mem, body_rty) in
          Array.iteri
            begin
              fun i atom ->
                let cell = get_element_ptr_dyn_in_current_frame body i in
                  trans_init_ty_from_atom cell unit_ty atom
            end
            atoms;
            mov (get_element_ptr vec Abi.vec_elt_fill) (Il.Cell fill);


  and trans_new_box
      (initializing:bool)
      (dst:Ast.lval)
      (src:Ast.atom)
      : unit =
    let src_op = trans_atom src in
    let src_cell = Il.Mem (force_to_mem src_op) in
    let src_ty = simplified_ty (atom_type cx src) in
    let (dst_cell, dst_ty) = trans_lval_maybe_init initializing dst in
    let _ =
      drop_existing_if_not_init initializing dst_cell dst_ty
    in
    let dst_ty = strip_mutable_or_constrained_ty dst_ty in
    let (dst_cell, dst_ty) =
      deref_ty DEREF_one_box true dst_cell dst_ty
    in
    let _ =
      iflog (fun _ -> log cx "init_box: dst ty %a, src ty %a"
               Ast.sprintf_ty dst_ty Ast.sprintf_ty src_ty)
    in
    let _ = assert (dst_ty = src_ty) in
      trans_copy_ty (get_ty_params_of_current_frame()) true
        dst_cell dst_ty src_cell src_ty


  and get_dynamic_tydesc
      (idopt:node_id option)
      (t:Ast.ty)
      (force_stateful:bool)
      : Il.cell =
    let td = next_vreg_cell Il.voidptr_t in
    let root_desc =
      Il.Cell (crate_rel_to_ptr
                 (get_static_tydesc idopt t 0L 0L force_stateful)
                 (tydesc_rty word_bits))
    in
    let (t, param_descs) = linearize_ty_params t in
    let descs = Array.append [| root_desc |] param_descs in
    let n = Array.length descs in
    let rty = referent_type cx t in
    let (size_sz, align_sz) = Il.referent_ty_layout word_bits rty in
    let size = calculate_sz_in_current_frame size_sz in
    let align = calculate_sz_in_current_frame align_sz in
    let descs_ptr = next_vreg_cell Il.voidptr_t in
      if (Array.length descs) > 0
      then
        (* FIXME (issue #83): this relies on knowledge that spills are
         * contiguous.
         *)
        let spills =
          Array.map (fun _ -> next_spill_cell Il.voidptr_t) descs
        in
          Array.iteri (fun i t -> mov spills.(n-(i+1)) t) descs;
          lea descs_ptr (fst (need_mem_cell spills.(n-1)))
      else
        mov descs_ptr zero;
      trans_upcall "upcall_get_type_desc" td
        [| Il.Cell (curr_crate_ptr());
           size; align; imm (Int64.of_int n);
           Il.Cell descs_ptr |];
      td

  and get_tydesc (idopt:node_id option) (ty:Ast.ty) : Il.cell =
    iflog (fun _ -> log cx "getting tydesc for %a" Ast.sprintf_ty ty);
    let (ty', mut) = simplified_ty_innermost_was_mutable ty in
    match ty' with
        Ast.TY_param (idx, _) ->
          (get_ty_param_in_current_frame idx)
      | t when has_parametric_types t ->
          (get_dynamic_tydesc idopt t mut)
      | _ ->
          (crate_rel_to_ptr (get_static_tydesc idopt ty
                               (ty_sz cx ty)
                               (ty_align cx ty)
                               mut)
             (tydesc_rty word_bits))

  and box_rc_cell (cell:Il.cell) : Il.cell =
    get_element_ptr (deref cell) Abi.box_rc_field_refcnt

  and box_allocation_size
      (ty:Ast.ty)
      : Il.operand =
    let header_sz =
      match ty_mem_ctrl cx ty with
          MEM_gc
        | MEM_rc_opaque
        | MEM_rc_struct -> word_n Abi.box_rc_header_size
        | MEM_interior -> bug () "box_allocation_size of MEM_interior"
    in
    let ty = simplified_ty ty in
    let refty_sz =
      Il.referent_ty_size abi.Abi.abi_word_bits (referent_type cx ty)
    in
      match refty_sz with
          SIZE_fixed _ -> imm (Int64.add (ty_sz cx ty) header_sz)
        | _ ->
            let ty_params = get_ty_params_of_current_frame() in
            let refty_sz = calculate_sz ty_params refty_sz in
            let v = next_vreg word_sty in
            let vc = Il.Reg (v, word_sty) in
              mov vc refty_sz;
              add_to vc (imm header_sz);
              Il.Cell vc;

  and iter_tag_parts
      (ty_params:Il.cell)
      (dst_cell:Il.cell)
      (src_cell:Il.cell)
      (ttag:Ast.ty_tag)
      (f:Il.cell -> Il.cell -> Ast.ty -> unit)
      : unit =
    let src_tag = get_element_ptr src_cell Abi.tag_elt_discriminant in
    let dst_tag = get_element_ptr dst_cell Abi.tag_elt_discriminant in
    let src_union =
      get_element_ptr_dyn ty_params src_cell Abi.tag_elt_variant
    in
    let dst_union =
      get_element_ptr_dyn ty_params dst_cell Abi.tag_elt_variant
    in
    let tmp = next_vreg_cell word_sty in
    let n = get_n_tag_tups cx ttag in
      f dst_tag src_tag word_ty;
      mov tmp (Il.Cell src_tag);
      for i = 0 to n-1
      do
        let ttup = get_nth_tag_tup cx ttag i in
          if Array.length ttup <> 0
          then
            begin
              let pc0 = (emitter()).Il.emit_pc in
              (iflog (fun _ ->
                        annotate (Printf.sprintf "tag case #%i" i)));
              let jmps =
                trans_compare_simple Il.JNE
                  (Il.Cell tmp) (imm (Int64.of_int i))
              in
              let ttup = get_nth_tag_tup cx ttag i in
              let pc1 = (emitter()).Il.emit_pc in
                iter_tup_parts
                  (get_element_ptr_dyn ty_params)
                  (get_variant_ptr dst_union i)
                  (get_variant_ptr src_union i)
                  ttup f;

                (* Hack: if this variant is all dead code, blank it out. *)
                if pc1 = (emitter()).Il.emit_pc
                then
                  begin
                    for j = pc0 to (pc1-1)
                    do (emitter()).Il.emit_quads.(j) <- Il.deadq done
                  end;
                List.iter patch jmps;
            end
      done;

  and seq_unit_ty (seq:Ast.ty) : Ast.ty =
    match simplified_ty seq with
        Ast.TY_vec t -> t
      | Ast.TY_str -> Ast.TY_mach TY_u8
      | _ -> bug () "seq_unit_ty of non-vec, non-str type"


  and iter_seq_parts
      (ty_params:Il.cell)
      (dst_cell:Il.cell)
      (src_cell:Il.cell)
      (unit_ty:Ast.ty)
      (trailing_null:bool)
      (f:Il.cell -> Il.cell -> Ast.ty -> unit)
      : unit =

    let unit_sz = ty_sz_with_ty_params ty_params unit_ty in
    let _ = unit_ty in
      (* 
       * Unlike most of the iter_ty_parts helpers; this one allocates a
       * vreg and so has to be aware of when it's iterating over 2
       * sequences of cells or just 1.
       *)
      check_box_rty src_cell;
      check_box_rty dst_cell;
      if dst_cell = src_cell
      then
        begin
          let src_cell = deref src_cell in
          let data =
            get_element_ptr_dyn ty_params src_cell Abi.vec_elt_data
          in
          let len = get_element_ptr src_cell Abi.vec_elt_fill in
          let ptr = next_vreg_cell Il.voidptr_t in
          let lim = next_vreg_cell Il.voidptr_t in
            lea lim (fst (need_mem_cell data));
            mov ptr (Il.Cell lim);
            add_to lim (Il.Cell len);

            if trailing_null
            then sub_from lim (imm 1L);

            let back_jmp_target = mark () in
            let fwd_jmps =
              trans_compare_simple Il.JAE (Il.Cell ptr) (Il.Cell lim)
            in
            let unit_cell =
              deref (ptr_cast ptr (referent_type cx unit_ty))
            in
              f unit_cell unit_cell unit_ty;
              add_to ptr unit_sz;
              check_interrupt_flag ();
              emit (Il.jmp Il.JMP (Il.CodeLabel back_jmp_target));
              List.iter patch fwd_jmps;
        end
      else
        begin
          unimpl None "seq iter with src != dst."
        end


  and iter_ty_parts_full
      (ty_params:Il.cell)
      (dst_cell:Il.cell)
      (src_cell:Il.cell)
      (ty:Ast.ty)
      (f:Il.cell -> Il.cell -> Ast.ty -> unit)
      : unit =
    (* 
     * FIXME: this will require some reworking if we support
     * rec, tag or tup slots that fit in a vreg. It requires 
     * addrs presently.
     *)

    match strip_mutable_or_constrained_ty ty with
        Ast.TY_rec entries ->
          iter_rec_parts
            (get_element_ptr_dyn ty_params) dst_cell src_cell
            entries f

      | Ast.TY_tup tys ->
          iter_tup_parts
            (get_element_ptr_dyn ty_params) dst_cell src_cell
            tys f

      | Ast.TY_tag tag ->
          iter_tag_parts ty_params dst_cell src_cell tag f

      | Ast.TY_fn _
      | Ast.TY_obj _ -> bug () "Attempting to iterate over fn/pred/obj slots"

      | Ast.TY_vec _ ->
          let unit_ty = seq_unit_ty ty in
            iter_seq_parts ty_params dst_cell src_cell unit_ty false f

      | Ast.TY_str ->
          let unit_ty = seq_unit_ty ty in
            iter_seq_parts ty_params dst_cell src_cell unit_ty true f

      | _ -> ()

  (* 
   * This just calls iter_ty_parts_full with your cell as both src and
   * dst, with an adaptor function that discards the dst parts of the
   * parallel traversal and and calls your provided function on the
   * passed-in src parts.
   *)
  and iter_ty_parts
      (ty_params:Il.cell)
      (cell:Il.cell)
      (ty:Ast.ty)
      (f:Il.cell -> Ast.ty -> unit)
      : unit =
    iter_ty_parts_full ty_params cell cell ty
      (fun _ src_cell ty -> f src_cell ty)

  and drop_ty tp c t =
    (in_quad_category "drop" (fun _ -> drop_ty_normal tp c t))

  and drop_ty_normal tp c t =
    drop_ty_full false tp c t

  and drop_ty_full
      (force_inline:bool)
      (ty_params:Il.cell)
      (cell:Il.cell)
      (ty:Ast.ty)
      : unit =

    let mctrl = ty_mem_ctrl cx ty in
    let ty = strip_mutable_or_constrained_ty ty in
    let call_out_of_line _ =
      trans_call_simple_static_glue (get_drop_glue ty)
        ty_params [| alias cell |] None;
    in

      match ty with

          Ast.TY_fn _
        | Ast.TY_obj _ when force_inline ->
            note_drop_step ty "drop_ty: obj/fn path";
            let box_ptr =
              get_element_ptr cell Abi.binding_field_bound_data
            in
            let _ = check_box_rty box_ptr in
            let null_jmp = null_check box_ptr in
            let rc_jmps = drop_refcount_and_cmp box_ptr in
            let box = deref box_ptr in
            let body = get_element_ptr box Abi.box_rc_field_body in
            let tydesc = get_element_ptr body Abi.obj_body_elt_tydesc in
            let fields =
              match ty with
                  Ast.TY_fn _ ->
                    get_element_ptr body Abi.closure_body_elt_bound_args
                | _ ->
                    get_element_ptr body Abi.obj_body_elt_fields
            in
            let ty_params = get_tydesc_params ty_params tydesc in
              begin
                match ty with
                    Ast.TY_obj _ ->
                      let dtor =
                        get_element_ptr (deref tydesc)
                          Abi.tydesc_field_obj_drop_glue
                      in
                      let null_dtor_jmp = null_check dtor in
                        (* Call any dtor, if present. *)
                        note_drop_step ty "drop_ty: calling obj/fn dtor";
                        trans_call_dynamic_glue
                          tydesc
                          Abi.tydesc_field_obj_drop_glue
                          None
                          [| box_ptr |]
                          (Some box_ptr);
                        patch null_dtor_jmp;
                  | _ -> ()
              end;
              (* Drop the fields. *)
              note_drop_step ty "drop_ty: dropping obj/fn fields";
              trans_call_dynamic_glue
                tydesc
                Abi.tydesc_field_drop_glue
                None
                [| ty_params; alias fields |]
                None;
              (* FIXME: this will fail if the user has lied about the
               * state-ness of their obj. We need to store state-ness in the
               * captured tydesc, and use that.  *)
              note_drop_step ty "drop_ty: freeing obj/fn body";
              trans_free box_ptr (type_has_state cx ty);
              mov box_ptr zero;
              List.iter patch rc_jmps;
              patch null_jmp;
              note_drop_step ty "drop_ty: done obj path";

        | Ast.TY_fn _
        | Ast.TY_obj _ ->
            call_out_of_line()

        | Ast.TY_param (i, _) ->
          note_drop_step ty "drop_ty: parametric-ty path";
          aliasing false cell
            begin
              fun cell ->
                trans_call_simple_dynamic_glue
                  i
                  Abi.tydesc_field_drop_glue
                  ty_params
                  [| cell |]
                  None
            end;
          note_drop_step ty "drop_ty: done parametric-ty path";

        | _ ->
            match mctrl with
                MEM_gc
              | MEM_rc_opaque
              | MEM_rc_struct when force_inline ->

                  note_drop_step ty "drop_ty: box-drop path";

                  let _ = check_box_rty cell in
                  let null_jmp = null_check cell in
                  let js = drop_refcount_and_cmp cell in

                    (* FIXME (issue #25): check to see that the box has
                     * further box members; if it doesn't we can elide the
                     * call to the glue function.  *)

                    trans_call_simple_static_glue
                      (get_free_glue ty (mctrl = MEM_gc))
                      ty_params
                      [| cell |]
                      None;

                    (* Null the slot out to prevent double-free if the frame
                     * unwinds.  *)
                    mov cell zero;
                    List.iter patch js;
                    patch null_jmp;
                    note_drop_step ty "drop_ty: done box-drop path";

              | MEM_gc
              | MEM_rc_opaque
              | MEM_rc_struct ->
                  call_out_of_line()

              | MEM_interior
                  when type_points_to_heap cx ty ||
                    (n_used_type_params cx ty > 0) ->
                  begin
                    note_drop_step ty
                      "drop_ty possibly-heap-referencing path";
                    if force_inline || should_inline_structure_helpers ty
                    then iter_ty_parts ty_params cell ty
                      (drop_ty ty_params)
                    else
                      call_out_of_line();

                    note_drop_step ty
                      "drop_ty: done possibly-heap-referencing path";
                  end

              | MEM_interior ->
                  note_drop_step ty "drop_ty: no-op simple-interior path";
                  (* Interior allocation of all-interior value not caught
                   * above: nothing to do.  *)
                  ()

  and sever_ty
      (ty_params:Il.cell)
      (cell:Il.cell)
      (ty:Ast.ty)
      : unit =
    let _ = note_gc_step ty "severing" in
    let sever_box c =
      let _ = check_box_rty c in
      let null_jmp = null_check c in
      let rc = box_rc_cell c in
      let _ = note_gc_step ty "severing GC cell" in
        emit (Il.binary Il.SUB rc (Il.Cell rc) one);
        mov c zero;
        patch null_jmp
    in
    let ty = strip_mutable_or_constrained_ty ty in

      match ty with
          Ast.TY_fn _
        | Ast.TY_obj _ ->
            if type_has_state cx ty
            then
              let binding =
                get_element_ptr cell Abi.binding_field_bound_data
              in
                sever_box binding;

        | _ ->
            match ty_mem_ctrl cx ty with
                MEM_gc ->
                  sever_box cell

              | MEM_interior when type_points_to_heap cx ty ->
                  iter_ty_parts ty_params cell ty (sever_ty ty_params)

              | _ -> ()
                  (* No need to follow links / call glue; severing is
                     shallow. *)

  and clone_ty
      (ty_params:Il.cell)
      (clone_task:Il.cell)
      (dst:Il.cell)
      (src:Il.cell)
      (ty:Ast.ty)
      : unit =
    let ty = strip_mutable_or_constrained_ty ty in

      match ty with
          Ast.TY_chan _ ->
            trans_upcall "upcall_clone_chan" dst
              [| (Il.Cell clone_task); (Il.Cell src) |]
        | Ast.TY_task
        | Ast.TY_port _
        | _ when type_has_state cx ty
            -> bug () "cloning state type"
        | _ when i64_le (ty_sz cx ty) word_sz
            -> mov dst (Il.Cell src)
        | Ast.TY_fn _
        | Ast.TY_obj _ -> ()
        | Ast.TY_box ty ->
            let glue_fix = get_clone_glue ty in
              trans_call_static_glue
                (code_fixup_to_ptr_operand glue_fix)
                (Some dst)
                [| alias ty_params; src; clone_task |] None
        | _ ->
            iter_ty_parts_full ty_params dst src ty
              (clone_ty ty_params clone_task)

  and unfold_opaque_cell (c:Il.cell) (ty:Ast.ty) : Il.cell =
    match Il.cell_referent_ty c with
        Il.ScalarTy (Il.AddrTy _) ->
          begin
            match strip_mutable_or_constrained_ty ty with
                Ast.TY_box boxed ->
                  Il.ptr_cast c
                    (Il.StructTy [| word_rty; referent_type cx boxed |])
              | _ -> c
          end
      | _ -> c

  and free_ty
      (is_gc:bool)
      (ty_params:Il.cell)
      (ty:Ast.ty)
      (cell:Il.cell)
      : unit =
    check_box_rty cell;
    let cell = unfold_opaque_cell cell ty in
    check_box_rty cell;
    note_drop_step ty "in free-ty";
    begin
    match strip_mutable_or_constrained_ty ty with
        Ast.TY_port _ -> trans_del_port cell
      | Ast.TY_chan _ -> trans_del_chan cell
      | Ast.TY_task -> trans_kill_task cell
      | Ast.TY_str -> trans_free cell false
      | Ast.TY_vec s ->
          iter_seq_parts ty_params cell cell s false
             (fun _ src ty -> drop_ty ty_params src ty);
             trans_free cell is_gc

      | Ast.TY_box body_ty ->
          note_drop_step ty "in free-ty, dropping structured body";
          let (body_mem, _) =
            need_mem_cell
              (get_element_ptr_dyn ty_params (deref cell)
                 Abi.box_rc_field_body)
          in
          let vr = next_vreg_cell Il.voidptr_t in
            lea vr body_mem;
            trace_word cx.ctxt_sess.Session.sess_trace_drop vr;
            trans_call_simple_static_glue
              (get_drop_glue body_ty)
              ty_params
              [| vr |]
              None;
            note_drop_step ty "in free-ty, calling free";
            trans_free cell is_gc;

      | t -> bug () "freeing unexpected type: %a" Ast.sprintf_ty t
    end;
    note_drop_step ty "free-ty done";

  and mark_slot
      (ty_params:Il.cell)
      (cell:Il.cell)
      (slot:Ast.slot)
      : unit =
    (* Marking goes straight through aliases. Reachable means reachable. *)
    mark_ty ty_params (deref_slot false cell slot) (slot_ty slot)

  and mark_ty
      (ty_params:Il.cell)
      (cell:Il.cell)
      (ty:Ast.ty)
      : unit =
    let ty = strip_mutable_or_constrained_ty ty in
      match ty_mem_ctrl cx ty with
          MEM_gc ->
            let tmp = next_vreg_cell Il.voidptr_t in
              trans_upcall "upcall_mark" tmp [| Il.Cell cell |];
              let marked_jump =
                trans_compare_simple Il.JE (Il.Cell tmp) zero
              in
                (* Iterate over box parts marking outgoing links. *)
              let (body_mem, _) =
                need_mem_cell
                  (get_element_ptr_dyn ty_params (deref cell)
                     Abi.box_gc_field_body)
              in
                lea tmp body_mem;
                trans_call_simple_static_glue
                  (get_mark_glue ty)
                  ty_params
                  [| tmp |]
                  None;
                List.iter patch marked_jump;

        | MEM_interior when type_is_structured cx ty ->
            (iflog (fun _ ->
                      annotate ("mark interior memory " ^
                                  (Fmt.fmt_to_str Ast.fmt_ty ty))));
            let (mem, _) = need_mem_cell cell in
            let tmp = next_vreg_cell Il.voidptr_t in
              lea tmp mem;
              trans_call_simple_static_glue
                (get_mark_glue ty)
                ty_params
                [| tmp |]
                None

        | _ -> ()

  and check_box_rty cell =
    match cell with
        Il.Reg (_, Il.AddrTy (Il.StructTy fields))
      | Il.Mem (_, Il.ScalarTy (Il.AddrTy (Il.StructTy fields)))
          when (((Array.length fields) > 0) && (fields.(0) = word_rty)) -> ()
      | _ -> bug ()
          "expected plausibly-box cell, got %s"
            (Il.string_of_referent_ty (Il.cell_referent_ty cell))

  and drop_slot_in_current_frame
      (cell:Il.cell)
      (slot:Ast.slot)
      : unit =
      check_and_flush_chan cell slot;
      drop_slot (get_ty_params_of_current_frame()) cell
        { slot with
            Ast.slot_ty = Some (strip_mutable_or_constrained_ty
                                  (slot_ty slot)) }

  and drop_ty_in_current_frame
      (cell:Il.cell)
      (ty:Ast.ty)
      : unit =
    drop_ty (get_ty_params_of_current_frame()) cell
      (strip_mutable_or_constrained_ty ty)

  (* Returns a mark for a jmp that must be patched to the continuation of
   * the null case (i.e. fall-through means not null).
   *)
  and null_check (cell:Il.cell) : quad_idx =
    in_quad_category "null check"
      begin
        fun _ ->
          emit (Il.cmp (Il.Cell cell) zero);
          let j = mark() in
            emit (Il.jmp Il.JE Il.CodeNone);
            j
      end

  (* Returns a mark for a jmp that must be patched to the continuation of
   * the non-zero refcount case (i.e. fall-through means zero refcount).
   *)
  and drop_refcount_and_cmp (boxed:Il.cell) : quad_idx list =
    in_quad_category "refcount"
      begin
        fun _ ->
          iflog (fun _ -> annotate "drop refcount and maybe free");
          let rc = box_rc_cell boxed in
            if cx.ctxt_sess.Session.sess_trace_gc ||
              cx.ctxt_sess.Session.sess_trace_drop
            then
              begin
                trace_str true "refcount--";
                trace_word true boxed;
                trace_word true rc
              end;
            emit (Il.cmp (Il.Cell rc) (simm Abi.const_refcount));
            let j0 = mark() in
              emit (Il.jmp Il.JE Il.CodeNone);
              emit (Il.binary Il.SUB rc (Il.Cell rc) one);
              emit (Il.cmp (Il.Cell rc) zero);
              let j1 = mark () in
                emit (Il.jmp Il.JNE Il.CodeNone);
                [j0; j1]
      end

  and incr_refcount (boxed:Il.cell) : unit =
    in_quad_category "refcount"
      begin
        fun _ ->
          let rc = box_rc_cell boxed in
            if cx.ctxt_sess.Session.sess_trace_gc ||
              cx.ctxt_sess.Session.sess_trace_drop
            then
              begin
                trace_str true "refcount++";
                trace_word true boxed;
                trace_word true rc
              end;
            emit (Il.cmp (Il.Cell rc) (simm Abi.const_refcount));
            let j = mark() in
              emit (Il.jmp Il.JE Il.CodeNone);
              add_to rc one;
              patch j;
      end

  and drop_slot
      (ty_params:Il.cell)
      (cell:Il.cell)
      (slot:Ast.slot)
      : unit =
    match slot.Ast.slot_mode with
        Ast.MODE_alias -> ()
          (* Aliases are always free to drop. *)
      | Ast.MODE_local ->
          drop_ty ty_params cell (slot_ty slot)

  and note_drop_step ty step =
    if (should_log cx (cx.ctxt_sess.Session.sess_trace_drop ||
                         cx.ctxt_sess.Session.sess_log_trans))
    then
      let mctrl_str =
        match ty_mem_ctrl cx ty with
            MEM_gc -> "MEM_gc"
          | MEM_rc_struct -> "MEM_rc_struct"
          | MEM_rc_opaque -> "MEM_rc_opaque"
          | MEM_interior -> "MEM_interior"
      in
      let tystr = Fmt.fmt_to_str Ast.fmt_ty ty in
      let str = step ^ " " ^ mctrl_str ^ " " ^ tystr in
        begin
          annotate str;
          trace_str cx.ctxt_sess.Session.sess_trace_drop str
        end

  and note_gc_step ty step =
    if (should_log cx (cx.ctxt_sess.Session.sess_trace_gc ||
                         cx.ctxt_sess.Session.sess_log_trans))
    then
      let mctrl_str =
        match ty_mem_ctrl cx ty with
            MEM_gc -> "MEM_gc"
          | MEM_rc_struct -> "MEM_rc_struct"
          | MEM_rc_opaque -> "MEM_rc_opaque"
          | MEM_interior -> "MEM_interior"
      in
      let tystr = Fmt.fmt_to_str Ast.fmt_ty ty in
      let str = step ^ " " ^ mctrl_str ^ " " ^ tystr in
        begin
          annotate str;
          trace_str cx.ctxt_sess.Session.sess_trace_gc str
        end

  (* Returns the offset of the slot-body in the initialized allocation. *)
  and init_box (cell:Il.cell) (ty:Ast.ty) : unit =
    let mctrl = ty_mem_ctrl cx ty in
      match mctrl with
          MEM_gc
        | MEM_rc_opaque
        | MEM_rc_struct ->
            let ctrl =
              if mctrl = MEM_gc
              then Il.Cell (get_tydesc None ty)
              else zero
            in
              iflog (fun _ -> annotate "init box: malloc");
              let sz = box_allocation_size ty in
                trans_malloc cell sz ctrl;
                iflog (fun _ -> annotate "init box: load refcount");
                let rc = box_rc_cell cell in
                  mov rc one

      | MEM_interior -> bug () "init_box of MEM_interior"

  and deref_ty
      (dctrl:deref_ctrl)
      (initializing:bool)
      (cell:Il.cell)
      (ty:Ast.ty)
      : (Il.cell * Ast.ty) =
    match (ty, dctrl) with

      | (Ast.TY_mutable ty, _)
      | (Ast.TY_constrained (ty, _), _) ->
          deref_ty dctrl initializing cell ty

      | (Ast.TY_box ty', DEREF_one_box)
      | (Ast.TY_box ty', DEREF_all_boxes) ->
          check_box_rty cell;
          let cell = unfold_opaque_cell cell ty in
          check_box_rty cell;
          if initializing
          then init_box cell ty;
          let cell =
            get_element_ptr_dyn_in_current_frame
              (deref cell)
              (Abi.box_rc_field_body)
          in
          let inner_dctrl =
            if dctrl = DEREF_one_box
            then DEREF_none
            else DEREF_all_boxes
          in
            (* Possibly deref recursively. *)
            deref_ty inner_dctrl initializing cell ty'

      | _ -> (cell, ty)


  and deref_slot
      (initializing:bool)
      (cell:Il.cell)
      (slot:Ast.slot)
      : Il.cell =
    match slot.Ast.slot_mode with
        Ast.MODE_local ->
          cell

      | Ast.MODE_alias  ->
          if initializing
          then cell
          else deref cell

  and trans_copy_tup
      (ty_params:Il.cell)
      (initializing:bool)
      (dst:Il.cell)
      (src:Il.cell)
      (tys:Ast.ty_tup)
      : unit =
    Array.iteri
      begin
        fun i ty ->
          let sub_dst_cell = get_element_ptr_dyn ty_params dst i in
          let sub_src_cell = get_element_ptr_dyn ty_params src i in
            trans_copy_ty
              ty_params
              initializing
              sub_dst_cell ty
              sub_src_cell ty
      end
      tys

  and trans_take_ty
      (force_inline:bool)
      (ty_params:Il.cell)
      (v:Il.cell)
      (ty:Ast.ty)
      : unit =
    let ty = strip_mutable_or_constrained_ty ty in
    match ty_mem_ctrl cx ty with
        MEM_rc_opaque | MEM_gc | MEM_rc_struct -> incr_refcount v
      | _ ->
          begin
            match ty with
                Ast.TY_fn _
              | Ast.TY_obj _ ->
                  let binding =
                    get_element_ptr v Abi.binding_field_bound_data
                  in
                  let null_jmp = null_check binding in
                    incr_refcount binding;
                    patch null_jmp

              | Ast.TY_param (i, _) ->
                  aliasing false v
                    begin
                      fun v ->
                        let td = get_ty_param ty_params i in
                        let ty_params_ptr = get_tydesc_params ty_params td in
                          trans_call_dynamic_glue
                            td Abi.tydesc_field_take_glue
                            None
                            [| ty_params_ptr; v; |]
                            None
                    end

              | Ast.TY_rec _
              | Ast.TY_tag _
              | Ast.TY_tup _ ->
                  if force_inline
                  then
                    iter_ty_parts ty_params v ty
                      (trans_take_ty force_inline ty_params)
                  else
                    trans_call_static_glue
                      (code_fixup_to_ptr_operand (get_take_glue ty))
                      None
                      [| alias ty_params; alias v; |]
                      None

              | _ -> ()
          end

  and trans_copy_ty
      (ty_params:Il.cell)
      (initializing:bool)
      (dst:Il.cell) (dst_ty:Ast.ty)
      (src:Il.cell) (src_ty:Ast.ty)
      : unit =
    let anno (weight:string) : unit =
      iflog
        begin
          fun _ ->
            annotate
              (Printf.sprintf "%sweight copy: %a <- %a"
                 weight
                 Ast.sprintf_ty dst_ty
                 Ast.sprintf_ty src_ty)
        end;
    in
      iflog
        begin
          fun _ ->
            log cx "trans_copy_ty";
            log cx "   dst ty %a, src ty %a"
              Ast.sprintf_ty dst_ty Ast.sprintf_ty src_ty;
            log cx "   dst cell %s, src cell %s"
              (cell_str dst) (cell_str src);
        end;
      assert (simplified_ty src_ty = simplified_ty dst_ty);
      in_quad_category "copy"
        begin
          fun _ ->
            match (ty_mem_ctrl cx src_ty, ty_mem_ctrl cx dst_ty) with
                (MEM_rc_opaque, MEM_rc_opaque)
              | (MEM_gc, MEM_gc)
              | (MEM_rc_struct, MEM_rc_struct) ->
                  (* Lightweight copy: twiddle refcounts, move pointer. *)
                  anno "refcounted light";
                  incr_refcount src;
                  if not initializing
                  then
                    drop_ty ty_params dst dst_ty;
                  mov dst (Il.Cell src)

              | _ ->
                  (* Heavyweight copy: duplicate 1 level of the referent. *)
                  anno "heavy";
                  trans_copy_ty_heavy ty_params initializing
                    dst dst_ty src src_ty
        end

  (* NB: heavyweight copying here does not mean "producing a deep
   * clone of the entire data tree rooted at the src operand". It means
   * "replicating a single level of the tree".
   * 
   * There is no general-recursion entailed in performing a heavy
   * copy. There is only "one level" to each heavy copy call.
   * 
   * In other words, this is a lightweight copy:
   * 
   *    [dstptr]  <-copy-  [srcptr]
   *         \              |
   *          \             |
   *        [some record.rc++]
   *             |
   *           [some other record]
   * 
   * Whereas this is a heavyweight copy:
   * 
   *    [dstptr]  <-copy-  [srcptr]
   *       |                  |
   *       |                  |
   *  [some record]       [some record]
   *             |          |
   *           [some other record]
   * 
   *)

  and trans_copy_ty_heavy
      (ty_params:Il.cell)
      (initializing:bool)
      (dst:Il.cell) (dst_ty:Ast.ty)
      (src:Il.cell) (src_ty:Ast.ty)
      : unit =
    let src_ty = strip_mutable_or_constrained_ty src_ty in
    let dst_ty = strip_mutable_or_constrained_ty dst_ty in

      iflog
        begin
          fun _ ->
            log cx "trans_copy_ty_heavy";
            log cx "   dst ty %a, src ty %a"
              Ast.sprintf_ty dst_ty Ast.sprintf_ty src_ty;
            log cx "   dst cell %s, src cell %s"
              (cell_str dst) (cell_str src);
        end;

      assert (src_ty = dst_ty);

      iflog (fun _ ->
               annotate ("heavy copy: slot preparation"));

      let (dst, ty) = deref_ty DEREF_none initializing dst dst_ty in
      let (src, _) = deref_ty DEREF_none false src src_ty in
        assert (ty = dst_ty);
        match ty with
            Ast.TY_nil
          | Ast.TY_bool
          | Ast.TY_mach _
          | Ast.TY_int
          | Ast.TY_uint
          | Ast.TY_native _
          | Ast.TY_type
          | Ast.TY_char ->
              iflog
                (fun _ -> annotate
                   (Printf.sprintf "copy_ty: simple mov (%Ld byte scalar)"
                      (ty_sz cx ty)));
              mov dst (Il.Cell src)

          | _ ->
              trans_take_ty false ty_params src ty;
              if not initializing
              then drop_ty ty_params dst ty;
              let sz = ty_sz_with_ty_params ty_params ty in
                copy_loop dst src sz (imm 1L)
                  (fun dptr sptr ->
                     mov (deref dptr) (Il.Cell (deref sptr)))


  and trans_copy
      (initializing:bool)
      (dst:Ast.lval)
      (src:Ast.expr)
      : unit =
    let (dst_cell, dst_ty) = trans_lval_maybe_init initializing dst in
    let dst_ty = strip_mutable_or_constrained_ty dst_ty in
    let rec can_append t =
      match t with
          Ast.TY_vec _
        | Ast.TY_str -> true
        | Ast.TY_box t when can_append t -> true
        | _ -> false
    in
      match (dst_ty, src) with
          (t,
           Ast.EXPR_binary (Ast.BINOP_add,
                            Ast.ATOM_lval a, Ast.ATOM_lval b))
            when can_append t ->
            (*
             * Translate str or vec
             * 
             *   s = a + b
             * 
             * as
             * 
             *   s = a;
             *   s += b;
             *)
            let (a_cell, a_ty) = trans_lval a in
            let (b_cell, b_ty) = trans_lval b in
              trans_copy_ty
                (get_ty_params_of_current_frame())
                initializing dst_cell dst_ty
                a_cell a_ty;
              trans_vec_append dst_cell dst_ty
                (Il.Cell b_cell) b_ty


        | (Ast.TY_obj caller_obj_ty,
           Ast.EXPR_unary (Ast.UNOP_cast t, a)) ->
            let src_ty = atom_type cx a in
            let _ = assert (not (is_prim_type (src_ty))) in
              begin
                let t = Hashtbl.find cx.ctxt_all_cast_types t.id in
                let _ = assert (t = (Ast.TY_obj caller_obj_ty)) in
                let callee_obj_ty =
                  match atom_type cx a with
                      Ast.TY_obj t -> t
                    | _ -> bug () "obj cast from non-obj type"
                in
                let src_cell = need_cell (trans_atom a) in

                (* FIXME (issue #84): this is wrong. It treats the underlying
                 * obj-state as the same as the callee and simply substitutes
                 * the forwarding vtbl, which would be great if it had any way
                 * convey the callee vtbl to the forwarding functions. But it
                 * doesn't. Instead, we have to malloc a fresh 3-word
                 * refcounted obj to hold the callee's vtbl+state pair, copy
                 * that in as the state here.  *)
                let _ =
                  trans_copy_ty (get_ty_params_of_current_frame())
                    initializing
                    dst_cell dst_ty
                    src_cell src_ty
                in
                let caller_vtbl_oper =
                  get_forwarding_vtbl caller_obj_ty callee_obj_ty
                in
                let (caller_obj, _) =
                  deref_ty DEREF_none initializing dst_cell dst_ty
                in
                let caller_vtbl =
                  get_element_ptr caller_obj Abi.obj_field_vtbl
                in
                  mov caller_vtbl caller_vtbl_oper
              end

        | (_, Ast.EXPR_binary _)
        | (_, Ast.EXPR_unary _)
        | (_, Ast.EXPR_atom (Ast.ATOM_literal _)) ->
            (*
             * Translations of these expr types yield vregs,
             * so copy is just MOV into the lval.
             *)
            let src_operand = trans_expr src in
              mov
                (fst (deref_ty DEREF_none false dst_cell dst_ty))
                src_operand

        | (_, Ast.EXPR_atom (Ast.ATOM_lval src_lval)) ->
            if lval_is_direct_fn cx src_lval then
              trans_init_direct_fn dst_cell src_lval
            else
              (* Possibly-large structure copying *)
              let (src_cell, src_ty) = trans_lval src_lval in
                trans_copy_ty
                  (get_ty_params_of_current_frame())
                  initializing
                  dst_cell dst_ty
                  src_cell src_ty

        | (_, Ast.EXPR_atom (Ast.ATOM_pexp _)) ->
            bug () "Trans.trans_copy on ATOM_pexp"


  and trans_init_direct_fn
      (dst_cell:Il.cell)
      (flv:Ast.lval)
      : unit =
    let item = lval_item cx flv in
    let fix = Hashtbl.find cx.ctxt_fn_fixups item.id in

    let dst_pair_code_cell =
      get_element_ptr dst_cell Abi.fn_field_code
    in

    let dst_pair_box_cell =
      get_element_ptr dst_cell Abi.fn_field_box
    in
      mov dst_pair_code_cell (reify_ptr (Il.ImmPtr (fix, Il.CodeTy)));
      mov dst_pair_box_cell zero


  and trans_init_structural_from_atoms
      (dst:Il.cell)
      (dst_tys:Ast.ty array)
      (atoms:Ast.atom array)
      : unit =
    Array.iteri
      begin
        fun i atom ->
          trans_init_ty_from_atom
            (get_element_ptr_dyn_in_current_frame dst i)
            dst_tys.(i) atom
      end
      atoms

  and trans_new_rec_update
      (dst:Il.cell)
      (dst_tys:Ast.ty array)
      (trec:Ast.ty_rec)
      (atab:(Ast.ident * Ast.mutability * Ast.atom) array)
      (base:Ast.lval)
      : unit =
    Array.iteri
      begin
        fun i (fml_ident, _) ->
          let fml_entry _ (act_ident, _, atom) =
            if act_ident = fml_ident then Some atom else None
          in
          let dst_ty = dst_tys.(i) in
            match arr_search atab fml_entry with
                Some atom ->
                  trans_init_ty_from_atom
                    (get_element_ptr_dyn_in_current_frame dst i)
                    dst_ty atom
              | None ->
                  let (src, _) = trans_lval base in
                    trans_copy_ty
                      (get_ty_params_of_current_frame()) true
                      (get_element_ptr_dyn_in_current_frame dst i) dst_ty
                      (get_element_ptr_dyn_in_current_frame src i) dst_ty
      end
      trec

  and trans_init_ty_from_atom
      (dst:Il.cell) (ty:Ast.ty) (atom:Ast.atom)
      : unit =
    let src = Il.Mem (force_to_mem (trans_atom atom)) in
      trans_copy_ty (get_ty_params_of_current_frame())
       true dst ty src ty

  and trans_init_slot_from_cell
      (ty_params:Il.cell)
      (clone:clone_ctrl)
      (dst:Il.cell) (dst_slot:Ast.slot)
      (src:Il.cell) (src_ty:Ast.ty)
      : unit =
    let dst_ty = slot_ty dst_slot in
    let _ =
      iflog (fun _ ->
               log cx "trans_init_slot_from_cell";
               log cx "   dst slot %a, src ty %a"
                 Ast.sprintf_slot dst_slot Ast.sprintf_ty src_ty;
               log cx "   dst cell %s, src cell %s"
                 (cell_str dst) (cell_str src))
    in
      match (dst_slot.Ast.slot_mode, clone) with
          (Ast.MODE_alias, CLONE_none) ->
            mov dst (Il.Cell (alias (Il.Mem (need_mem_cell src))))

        | (Ast.MODE_local, CLONE_none) ->
            trans_copy_ty
              ty_params true
              dst dst_ty src src_ty

        | (Ast.MODE_alias, _) ->
            bug () "attempting to clone into alias slot"

        | (_, CLONE_chan clone_task) ->
            let clone =
              if (type_contains_chan cx src_ty)
              then CLONE_all clone_task
              else CLONE_none
            in
              (* Feed back with massaged args. *)
              trans_init_slot_from_cell ty_params
                clone dst dst_slot src src_ty

        | (_, CLONE_all clone_task) ->
            clone_ty ty_params clone_task dst src src_ty


  and trans_init_slot_from_atom
      (clone:clone_ctrl)
      (dst:Il.cell) (dst_slot:Ast.slot)
      (src_atom:Ast.atom)
      : unit =
    let _ =
      iflog (fun _ ->
               log cx "trans_init_slot_from_atom";
               log cx "   dst slot %a, src ty %a"
                 Ast.sprintf_slot dst_slot
                 Ast.sprintf_ty (atom_type cx src_atom);
               log cx "   dst cell %s"
                 (cell_str dst))
    in
    match (dst_slot.Ast.slot_mode, clone, src_atom) with
        (Ast.MODE_alias, CLONE_none,
         Ast.ATOM_literal _) ->
          (* Aliasing a literal is a bit weird since nobody
           * else will ever see it, but it seems harmless.
           *)
          let src = trans_atom src_atom in
            mov dst (Il.Cell (alias (Il.Mem (force_to_mem src))))

      | (Ast.MODE_alias, CLONE_chan _, _)
      | (Ast.MODE_alias, CLONE_all _, _) ->
          bug () "attempting to clone into alias slot"
      | _ ->
          let src = Il.Mem (force_to_mem (trans_atom src_atom)) in
            begin
              log cx "   forced-to-mem src cell %s" (cell_str src);
              trans_init_slot_from_cell
                (get_ty_params_of_current_frame())
                clone dst dst_slot src (atom_type cx src_atom)
            end


  and trans_be_fn
      (cx:ctxt)
      (dst_cell:Il.cell)
      (flv:Ast.lval)
      (ty_params:Ast.ty array)
      (args:Ast.atom array)
      : unit =
    let (ptr, fn_ty) = trans_callee flv in
    let cc = call_ctrl flv in
    let call = { call_ctrl = cc;
                 call_callee_ptr = ptr;
                 call_callee_ty = fn_ty;
                 call_callee_ty_params = ty_params;
                 call_output = dst_cell;
                 call_args = args;
                 call_iterator_args = call_iterator_args None;
                 call_indirect_args = call_indirect_args flv cc }
    in
      (* FIXME (issue #85): true if caller is object fn *)
    let caller_is_closure = false in
      log cx "trans_be_fn: %s call to lval %a"
        (call_ctrl_string cc) Ast.sprintf_lval flv;
      trans_be (fun () -> Ast.sprintf_lval () flv) caller_is_closure call

  and trans_prepare_fn_call
      (initializing:bool)
      (cx:ctxt)
      (dst_cell:Il.cell)
      (flv:Ast.lval)
      (ty_params:Ast.ty array)
      (fco:for_each_ctrl option)
      (args:Ast.atom array)
      : Il.operand =
    let (ptr, fn_ty) = trans_callee flv in
    let cc = call_ctrl flv in
    let call = { call_ctrl = cc;
                 call_callee_ptr = ptr;
                 call_callee_ty = fn_ty;
                 call_callee_ty_params = ty_params;
                 call_output = dst_cell;
                 call_args = args;
                 call_iterator_args = call_iterator_args fco;
                 call_indirect_args = call_indirect_args flv cc }
    in
      iflog
        begin
          fun _ ->
            log cx "trans_prepare_fn_call: %s call to lval %a"
              (call_ctrl_string cc) Ast.sprintf_lval flv;
            log cx "lval type: %a" Ast.sprintf_ty fn_ty;
            Array.iteri (fun i t -> log cx "ty param %d = %a"
                           i Ast.sprintf_ty t)
              ty_params;
        end;
      trans_prepare_call initializing (fun () -> Ast.sprintf_lval () flv) call

  and trans_call_pred_and_check
      (constr:Ast.constr)
      (flv:Ast.lval)
      (args:Ast.atom array)
      : unit =
    let (ptr, fn_ty) = trans_callee flv in
    let dst_cell = Il.Mem (force_to_mem imm_false) in
    let call = { call_ctrl = call_ctrl flv;
                 call_callee_ptr = ptr;
                 call_callee_ty = fn_ty;
                 call_callee_ty_params = [| |];
                 call_output = dst_cell;
                 call_args = args;
                 call_iterator_args = [| |];
                 call_indirect_args = [| |] }
    in
      iflog (fun _ -> annotate "predicate call");
      let fn_ptr =
        trans_prepare_call true (fun _ -> Ast.sprintf_lval () flv) call
      in
        call_code (code_of_operand fn_ptr);
        iflog (fun _ -> annotate "predicate check/fail");
        let jmp = trans_compare_simple Il.JE (Il.Cell dst_cell) imm_true in
        let errstr = Printf.sprintf "predicate check: %a"
          Ast.sprintf_constr constr
        in
          trans_cond_fail errstr jmp

  and trans_init_closure
      (closure_cell:Il.cell)
      (target_fn_ptr:Il.operand)
      (target_binding_ptr:Il.operand)
      (ty_params:Ast.ty array)
      (bound_arg_slots:Ast.slot array)
      (bound_args:Ast.atom array)
      : unit =

    let rc_cell = get_element_ptr closure_cell Abi.box_rc_field_refcnt in
    let body_cell = get_element_ptr closure_cell Abi.box_rc_field_body in
    let targ_cell = get_element_ptr body_cell Abi.closure_body_elt_target in
    let bound_args_tydesc_cell =
      get_element_ptr body_cell Abi.closure_body_elt_bound_args_tydesc
    in
    let bound_ty_params_cell =
      get_element_ptr body_cell Abi.closure_body_elt_bound_ty_params
    in
    let args_cell =
      get_element_ptr body_cell Abi.closure_body_elt_bound_args
    in

    iflog (fun _ -> annotate "init closure refcount");
    mov rc_cell one;

    iflog (fun _ -> annotate "set closure bound-args tydesc ptr");
    mov bound_args_tydesc_cell
      (Il.Cell (get_tydesc None
                  (Ast.TY_tup (Array.map slot_ty bound_arg_slots))));


    iflog (fun _ -> annotate "set closure target code ptr");
    mov
      (get_element_ptr targ_cell Abi.fn_field_code)
      (reify_ptr target_fn_ptr);

    iflog (fun _ -> annotate "set closure target closure ptr");
    mov
      (get_element_ptr targ_cell Abi.fn_field_box)
      (reify_ptr target_binding_ptr);

    iflog (fun _ -> annotate "set closure bound tydescs");
    Array.iteri
      begin
        fun i ty ->
          mov
            (get_element_ptr bound_ty_params_cell i)
            (Il.Cell (get_tydesc None ty))
      end
      ty_params;

    iflog (fun _ -> annotate "set closure bound args");
    copy_bound_args args_cell bound_arg_slots bound_args

  and trans_bind_fn
      (initializing:bool)
      (cc:call_ctrl)
      (bind_id:node_id)
      (dst:Ast.lval)
      (flv:Ast.lval)
      (fn_sig:Ast.ty_sig)
      (args:Ast.atom option array)
      : unit =
    let (dst_cell, _) = trans_lval_maybe_init initializing dst in
    let (target_ptr, _) = trans_callee flv in
    let ty_params =
      match htab_search cx.ctxt_call_lval_params (lval_base_id flv) with
          Some params -> params
        | None -> [| |]
    in
    let n_ty_params = Array.length ty_params in
    let arg_bound_flags = Array.map bool_of_option args in
    let arg_slots =
      arr_map2
        (fun arg_slot bound_flag ->
           if bound_flag then Some arg_slot else None)
        fn_sig.Ast.sig_input_slots
        arg_bound_flags
    in
    let bound_arg_slots = arr_filter_some arg_slots in
    let bound_args = arr_filter_some args in
    let thunk_fixup =
      get_fn_thunk_glue bind_id n_ty_params
        fn_sig.Ast.sig_input_slots arg_bound_flags
    in
    let target_code_ptr = callee_code_ptr target_ptr cc in
    let target_box_ptr = callee_box_ptr flv cc in
    let closure_box_rty = closure_box_rty cx n_ty_params bound_arg_slots in
    let closure_box_sz =
      calculate_sz_in_current_frame
        (Il.referent_ty_size word_bits closure_box_rty)
    in
    let pair_code_cell = get_element_ptr dst_cell Abi.fn_field_code in
    let pair_box_cell =
      cell_cast
        (get_element_ptr dst_cell Abi.fn_field_box)
        (Il.ScalarTy (Il.AddrTy (closure_box_rty)))
    in
      iflog (fun _ -> annotate "assign thunk-ptr to code field of pair");
      mov pair_code_cell (reify_ptr (Il.ImmPtr (thunk_fixup, Il.CodeTy)));
      iflog (fun _ ->
               annotate "heap-allocate closure to binding slot of pair");
      trans_malloc pair_box_cell closure_box_sz zero;
      trans_init_closure
        (deref pair_box_cell)
        target_code_ptr
        target_box_ptr
        ty_params
        bound_arg_slots
        bound_args


  and trans_arg0 (arg_cell:Il.cell) (initializing:bool) (call:call) : unit =
    (* Emit arg0 of any call: the output slot. *)
    iflog (fun _ -> annotate "fn-call arg 0: output slot");
    if not initializing
    then
      drop_slot
        (get_ty_params_of_current_frame())
        call.call_output
        (call_output_slot call);
    (* We always get to the same state here: the output slot is uninitialized.
     * We then do something that's illegal to do in the language, but legal
     * here: alias the uninitialized memory. We are ok doing this because the
     * call will fill it in before anyone else observes it. That's the
     * point.
     *)
    mov arg_cell (Il.Cell (alias call.call_output));

  and trans_arg1 (arg_cell:Il.cell) : unit =
    (* Emit arg1 of any call: the task pointer. *)
    iflog (fun _ -> annotate "fn-call arg 1: task pointer");
    trans_init_slot_from_cell
      (get_ty_params_of_current_frame())
      CLONE_none
      arg_cell word_slot
      abi.Abi.abi_tp_cell word_ty

  and trans_argN
      (clone:clone_ctrl)
      (arg_cell:Il.cell)
      (arg_slot:Ast.slot)
      (arg:Ast.atom)
      : unit =
    iflog (fun _ -> log cx "trans_argN: arg slot %a, arg atom %a"
             Ast.sprintf_slot arg_slot Ast.sprintf_atom arg);
    trans_init_slot_from_atom clone arg_cell arg_slot arg

  and code_of_cell (cell:Il.cell) : Il.code =
    match cell with
        Il.Mem (_, Il.ScalarTy (Il.AddrTy Il.CodeTy))
      | Il.Reg (_, Il.AddrTy Il.CodeTy) -> Il.CodePtr (Il.Cell cell)
      | _ ->
          bug () "expected code-pointer cell, found %s"
            (cell_str cell)

  and code_of_operand (operand:Il.operand) : Il.code =
    match operand with
        Il.Cell c -> code_of_cell c
      | Il.ImmPtr (_, Il.CodeTy) -> Il.CodePtr operand
      | _ ->
          bug () "expected code-pointer operand, got %s"
            (oper_str operand)

  and ty_arg_slots (ty:Ast.ty) : Ast.slot array =
    match simplified_ty ty with
        Ast.TY_fn (tsig, _) -> tsig.Ast.sig_input_slots
      | _ -> bug () "Trans.ty_arg_slots on non-callable type: %a"
          Ast.sprintf_ty ty

  and copy_fn_args
      (tail_area:bool)
      (initializing_arg0:bool)
      (clone:clone_ctrl)
      (call:call)
      : unit =

    let n_ty_params = Array.length call.call_callee_ty_params in
    let all_callee_args_rty =
      let clo =
        if call.call_ctrl = CALL_direct
        then None
        else (Some Il.OpaqueTy)
      in
        call_args_referent_type cx n_ty_params call.call_callee_ty clo
    in
    let all_callee_args_cell =
      callee_args_cell tail_area all_callee_args_rty
    in

    let _ = iflog (fun _ -> annotate
                     (Printf.sprintf
                        "copying fn args to %d-ty-param call with rty: %s\n"
                        n_ty_params (Il.string_of_referent_ty
                                       all_callee_args_rty)))
    in
    let callee_arg_slots = ty_arg_slots call.call_callee_ty in
    let callee_output_cell =
      get_element_ptr all_callee_args_cell Abi.calltup_elt_out_ptr
    in
    let callee_task_cell =
      get_element_ptr all_callee_args_cell Abi.calltup_elt_task_ptr
    in
    let callee_indirect_args =
      get_element_ptr all_callee_args_cell Abi.calltup_elt_indirect_args
    in
    let callee_ty_params =
      get_element_ptr all_callee_args_cell Abi.calltup_elt_ty_params
    in
    let callee_args =
      get_element_ptr_dyn_in_current_frame
        all_callee_args_cell Abi.calltup_elt_args
    in
    let callee_iterator_args =
      get_element_ptr_dyn_in_current_frame
        all_callee_args_cell Abi.calltup_elt_iterator_args
    in

    let n_args = Array.length call.call_args in
    let n_iterators = Array.length call.call_iterator_args in
    let n_indirects = Array.length call.call_indirect_args in

      Array.iteri
        begin
          fun i arg_atom ->
            iflog (fun _ ->
                     annotate
                       (Printf.sprintf "fn-call arg %d of %d (+ %d indirect)"
                          i n_args n_indirects));
            trans_argN
              clone
              (get_element_ptr_dyn_in_current_frame callee_args i)
              callee_arg_slots.(i)
              arg_atom
        end
        call.call_args;

      Array.iteri
        begin
          fun i iterator_arg_operand ->
            iflog (fun _ ->
                     annotate (Printf.sprintf "fn-call iterator-arg %d of %d"
                                 i n_iterators));
            mov
              (get_element_ptr_dyn_in_current_frame callee_iterator_args i)
              iterator_arg_operand
        end
        call.call_iterator_args;

      Array.iteri
        begin
          fun i indirect_arg_operand ->
            iflog (fun _ ->
                     annotate (Printf.sprintf "fn-call indirect-arg %d of %d"
                                 i n_indirects));
            mov
              (get_element_ptr_dyn_in_current_frame callee_indirect_args i)
              indirect_arg_operand
        end
        call.call_indirect_args;

      Array.iteri
        begin
          fun i ty_param ->
            iflog (fun _ ->
                     annotate
                       (Printf.sprintf "fn-call ty param %d of %d"
                          i n_ty_params));
            trans_init_slot_from_cell
              (get_ty_params_of_current_frame())
              CLONE_none
              (get_element_ptr callee_ty_params i) word_slot
              (get_tydesc None ty_param) word_ty
        end
        call.call_callee_ty_params;

      trans_arg1 callee_task_cell;

      trans_arg0 callee_output_cell initializing_arg0 call



  and call_code (code:Il.code) : unit =
    let vr = next_vreg_cell Il.voidptr_t in
      emit (Il.call vr code);


  and copy_bound_args
      (dst_cell:Il.cell)
      (bound_arg_slots:Ast.slot array)
      (bound_args:Ast.atom array)
      : unit =
    let n_slots = Array.length bound_arg_slots in
      Array.iteri
        begin
          fun i slot ->
            iflog (fun _ ->
                     annotate (Printf.sprintf
                                 "copy bound arg %d of %d" i n_slots));
            trans_argN CLONE_none
              (get_element_ptr dst_cell i)
              slot bound_args.(i)
        end
        bound_arg_slots

  and merge_bound_args
      (n_ty_params:int)
      (all_self_args_rty:Il.referent_ty)
      (all_callee_args_rty:Il.referent_ty)
      (arg_slots:Ast.slot array)
      (arg_bound_flags:bool array)
      : unit =
    begin
      (* 
       * NB: 'all_*_args', both self and callee, are always 5-tuples: 
       * 
       *    [out_ptr, task_ptr, indirect_args, ty_params, [args]]
       * 
       * The first few bindings here just destructure those via GEP.
       * 
       *)
      let all_self_args_cell = caller_args_cell all_self_args_rty in
      let all_callee_args_cell = callee_args_cell false all_callee_args_rty in

      let self_args_cell =
        get_element_ptr all_self_args_cell Abi.calltup_elt_args
      in
      let self_indirect_args_cell =
        get_element_ptr all_self_args_cell Abi.calltup_elt_indirect_args
      in
      let closure_box_cell =
        deref (get_element_ptr self_indirect_args_cell
                 Abi.indirect_args_elt_closure)
      in
      let closure_cell =
        get_element_ptr closure_box_cell Abi.box_rc_field_body
      in
      let closure_args_cell =
        get_element_ptr closure_cell Abi.closure_body_elt_bound_args
      in
      let closure_ty_params_cell =
        get_element_ptr closure_cell Abi.closure_body_elt_bound_ty_params
      in
      let callee_ty_params_cell =
        get_element_ptr all_callee_args_cell Abi.calltup_elt_ty_params
      in
      let callee_args_cell =
        get_element_ptr_dyn closure_ty_params_cell
          all_callee_args_cell Abi.calltup_elt_args
      in

      let n_args = Array.length arg_bound_flags in
      let bound_i = ref 0 in
      let unbound_i = ref 0 in

        iflog (fun _ -> annotate "copy out-ptr");
        mov
          (get_element_ptr all_callee_args_cell Abi.calltup_elt_out_ptr)
          (Il.Cell (get_element_ptr all_self_args_cell
                      Abi.calltup_elt_out_ptr));

        iflog (fun _ -> annotate "copy task-ptr");
        mov
          (get_element_ptr all_callee_args_cell Abi.calltup_elt_task_ptr)
          (Il.Cell (get_element_ptr all_self_args_cell
                      Abi.calltup_elt_task_ptr));

        iflog (fun _ -> annotate "copy ty-params");
        for ty_i = 0 to (n_ty_params - 1) do
          mov
            (get_element_ptr callee_ty_params_cell ty_i)
            (Il.Cell (get_element_ptr closure_ty_params_cell ty_i))
        done;

        iflog (fun _ -> annotate "copy args");
        for arg_i = 0 to (n_args - 1) do
          let dst_cell = get_element_ptr callee_args_cell arg_i in
          let slot = arg_slots.(arg_i) in
          let is_bound = arg_bound_flags.(arg_i) in
          let src_cell =
            if is_bound then
              begin
                iflog (fun _ -> annotate
                         (Printf.sprintf
                            "extract bound arg %d as actual arg %d"
                            !bound_i arg_i));
                get_element_ptr closure_args_cell (!bound_i)
              end
            else
              begin
                iflog (fun _ -> annotate
                         (Printf.sprintf
                            "extract unbound arg %d as actual arg %d"
                            !unbound_i arg_i));
                get_element_ptr self_args_cell (!unbound_i);
              end
          in
            iflog (fun _ -> annotate
                     (Printf.sprintf
                        "copy into actual-arg %d" arg_i));
            trans_init_slot_from_cell
              closure_ty_params_cell CLONE_none
              dst_cell slot
              (deref_slot false src_cell slot) (slot_ty slot);
            incr (if is_bound then bound_i else unbound_i);
        done;
        assert ((!bound_i + !unbound_i) == n_args)
    end


  and callee_code_ptr
      (fptr:Il.operand)
      (cc:call_ctrl)
      : Il.operand =
    match cc with
        CALL_direct
      | CALL_vtbl -> fptr
      | CALL_indirect ->
          (* fptr is a pair [code*, box*] *)
          let pair_cell = need_cell (reify_ptr fptr) in
            Il.Cell (get_element_ptr pair_cell Abi.fn_field_code)

  and callee_box_ptr
      (pair_lval:Ast.lval)
      (cc:call_ctrl)
      : Il.operand =
    if cc = CALL_direct
    then zero
    else
      let (pair_cell, ty) = trans_lval pair_lval in
      let (pair_cell, _) =
        if cc = CALL_vtbl
          (* |pair_lval| here is the obj to whose vtbl we're dispatching.
           * Said obj might have been auto-deref'ed for the method call,
           * so we have to be sure to do the same here.
           *)
        then deref_ty DEREF_all_boxes false pair_cell ty
        else (pair_cell, ty)
      in
        Il.Cell (get_element_ptr pair_cell Abi.binding_field_bound_data)

  and call_ctrl flv : call_ctrl =
    if lval_is_static cx flv
    then CALL_direct
    else
      if lval_is_obj_vtbl cx flv
      then CALL_vtbl
      else CALL_indirect

  and call_ctrl_string cc =
    match cc with
        CALL_direct -> "direct"
      | CALL_indirect -> "indirect"
      | CALL_vtbl -> "vtbl"

  and call_iterator_args
      (fco:for_each_ctrl option)
      : Il.operand array =
    match fco with
        None -> [| |]
      | Some fco ->
          begin
            iflog (fun _ -> annotate "calculate iterator args");
            [| reify_ptr (code_fixup_to_ptr_operand fco.for_each_fixup);
               Il.Cell (Il.Reg (abi.Abi.abi_fp_reg, Il.voidptr_t)); |]
          end

  and call_indirect_args
      (flv:Ast.lval)
      (cc:call_ctrl)
      : Il.operand array =
      begin
        match cc with
            CALL_direct -> [| |]
          | CALL_indirect -> [| callee_box_ptr flv cc |]
          | CALL_vtbl ->
              begin
                match flv with
                    (* FIXME (issue #84): will need to pass both words of obj
                     * if we add a 'self' value for self-dispatch within
                     * objs. Also to support forwarding-functions / 'as'.
                     *)
                    Ast.LVAL_ext (base, _) -> [| callee_box_ptr base cc |]
                  | _ ->
                      bug (lval_base_id flv)
                        "call_indirect_args on obj-fn without base obj"
              end
      end

  and trans_be
      (logname:(unit -> string))
      (caller_is_closure:bool)
      (call:call)
      : unit =
    let callee_fptr = callee_code_ptr call.call_callee_ptr call.call_ctrl in
    let callee_code = code_of_operand callee_fptr in
    let callee_args_rty =
      call_args_referent_type cx 0 call.call_callee_ty
        (if call.call_ctrl = CALL_direct then None else (Some Il.OpaqueTy))
    in
    let callee_argsz =
      force_sz (Il.referent_ty_size word_bits callee_args_rty)
    in
    let closure_rty =
      if caller_is_closure
      then Some Il.OpaqueTy
      else None
    in
    let caller_args_rty = current_fn_args_rty closure_rty in
    let caller_argsz =
      force_sz (Il.referent_ty_size word_bits caller_args_rty)
    in
      iflog (fun _ -> annotate
               (Printf.sprintf "copy args for tail call to %s" (logname ())));
      copy_fn_args true true CLONE_none call;
      drop_slots_at_curr_stmt();
      iflog (fun _ -> annotate "drop args");
      iter_arg_slots cx (current_fn()) callee_drop_slot;
      abi.Abi.abi_emit_fn_tail_call (emitter())
        (force_sz (current_fn_callsz()))
        caller_argsz callee_code callee_argsz;

  and trans_prepare_call
      (initializing:bool)
      (logname:(unit -> string))
      (call:call)
      : Il.operand =

    let callee_fptr = callee_code_ptr call.call_callee_ptr call.call_ctrl in
      iflog (fun _ -> annotate
               (Printf.sprintf "copy args for call to %s" (logname ())));
      copy_fn_args false initializing CLONE_none call;
      iflog (fun _ -> annotate (Printf.sprintf "call %s" (logname ())));
      callee_fptr

  and callee_drop_slot
      (k:Ast.slot_key)
      (slot_id:node_id)
      (slot:Ast.slot)
      : unit =
    iflog (fun _ ->
             annotate (Printf.sprintf "callee_drop_slot %d = %s "
                         (int_of_node slot_id)
                         (Fmt.fmt_to_str Ast.fmt_slot_key k)));
    drop_slot_in_current_frame (cell_of_block_slot slot_id) slot


  and trans_alt_tag (at:Ast.stmt_alt_tag) : unit =

    let trans_arm arm : quad_idx =
      let (pat, block) = arm.node in

      (* Translates the pattern and returns a pair where the first
          component is a list of the addresses of the branch
          instructions that are taken if the match fails,
          and the second component is a thunk that, when invoked,
          emits initialization code for the variables bound in this pattern.

          trans_pat can't just emit the initialization code itself, because
          then, pattern-bound variables could be taken without ever being 
          dropped if a nested pattern fails partway through (because the 
          drop code is part of the action for the pattern).
       *)
      let rec trans_pat
          (pat:Ast.pat)
          (src_cell:Il.cell)
          (src_ty:Ast.ty)
          : (quad_idx list *
             (unit -> unit)) =

        match pat with
            Ast.PAT_lit lit ->
              (trans_compare_simple Il.JNE (trans_lit lit) (Il.Cell src_cell),
               fun _ -> ())

          | Ast.PAT_tag (lval, pats) ->
              let tag_ident =
                match lval with
                    Ast.LVAL_ext (_, (Ast.COMP_named (Ast.COMP_ident id)))
                  | Ast.LVAL_ext (_, (Ast.COMP_named (Ast.COMP_app (id, _))))
                  | Ast.LVAL_base { node = Ast.BASE_ident id; id = _ }
                  | Ast.LVAL_base { node = Ast.BASE_app (id, _); id = _ } ->
                      id
                  | _ -> bug cx "expected lval ending in ident"
              in 
              let ttag =
                match strip_mutable_or_constrained_ty src_ty with
                    Ast.TY_tag ttag -> ttag
                  | _ -> bug cx "expected tag type"
              in
              let tinfo = Hashtbl.find cx.ctxt_all_tag_info ttag.Ast.tag_id in
              let (i,_,_) =
                Hashtbl.find tinfo.tag_idents tag_ident
              in
              let ttup = get_nth_tag_tup cx ttag i in

              let tag_cell:Il.cell =
                get_element_ptr src_cell Abi.tag_elt_discriminant
              in
              let union_cell =
                get_element_ptr_dyn_in_current_frame
                  src_cell
                  Abi.tag_elt_variant
              in

              let next_jumps =
                trans_compare_simple Il.JNE
                  (Il.Cell tag_cell) (imm (Int64.of_int i))
              in

              let tup_cell:Il.cell = get_variant_ptr union_cell i in

              let trans_elem_pat i elem_pat :
                    (quad_idx list * (unit -> unit)) =
                let elem_cell =
                  get_element_ptr_dyn_in_current_frame tup_cell i
                in
                let elem_ty = ttup.(i) in
                  trans_pat elem_pat elem_cell elem_ty
              in

              let elem_jumps =
                (Array.to_list (Array.mapi trans_elem_pat pats)) in
              let (elem_jump_addrs, ks) = List.split elem_jumps
              in
                (next_jumps @ (List.concat elem_jump_addrs),
                 (* Compose all the var-initialization thunks together
                    to make one thunk that initializes all the vars *)
                 List.fold_left (fun g f -> (fun x -> f (g x)))
                   (fun _ -> ()) ks)

          | Ast.PAT_slot (dst, _) ->
              let dst_slot = get_slot cx dst.id in
              let dst_cell = cell_of_block_slot dst.id in
              (* irrefutable *)
                ([], (fun () -> (* init the slot later, inside the block,
                                   once we know we had a match *)
                                  trans_init_slot_from_cell 
                                  (get_ty_params_of_current_frame())
                                  CLONE_none dst_cell dst_slot
                                  src_cell src_ty))
          | Ast.PAT_wild -> ([], fun _ -> ())     (* irrefutable *)
      in

      let (lval_cell, lval_ty) = trans_lval at.Ast.alt_tag_lval in
      let (next_jumps, prologue) = trans_pat pat lval_cell lval_ty in
        prologue (); (* binds any pattern-bound variables *)
        trans_block block;
        let last_jump = mark() in
          emit (Il.jmp Il.JMP Il.CodeNone);
          List.iter patch next_jumps;
          last_jump
    in
    let last_jumps = Array.map trans_arm at.Ast.alt_tag_arms in
      if not (arr_exists
                (fun _ arm -> (fst arm.node) = Ast.PAT_wild)
                at.Ast.alt_tag_arms)
      then
        trans_cond_fail "non-exhaustive match failure"
          (Array.to_list last_jumps)
      else
        Array.iter patch last_jumps

  (* If we're about to drop a channel, synthesize an upcall_flush_chan.
   * TODO: This should rather appear in a chan dtor when chans become
   * objects. *)
  and check_and_flush_chan
    (cell:Il.cell)
    (slot:Ast.slot)
      : unit =
      let ty = strip_mutable_or_constrained_ty (slot_ty slot) in
      match simplified_ty ty with
          Ast.TY_chan _ ->
                annotate "check_and_flush_chan, flush_chan";
                let rc = box_rc_cell cell in
                  emit (Il.cmp (Il.Cell rc) one);
                let jump = mark () in
                  emit (Il.jmp Il.JNE Il.CodeNone);
                  trans_void_upcall "upcall_flush_chan" [| Il.Cell cell |];
                  patch jump;
        | _ -> ()

  and drop_slots_at_curr_stmt _ : unit =
    let stmt = Stack.top curr_stmt in
      match htab_search cx.ctxt_post_stmt_slot_drops stmt with
          None -> ()
        | Some slots ->
            List.iter
              begin
                fun slot_id ->
                  let slot = get_slot cx slot_id in
                  let k = Hashtbl.find cx.ctxt_slot_keys slot_id in
                    iflog (fun _ ->
                             annotate
                               (Printf.sprintf
                                  "post-stmt, drop_slot %d = %s "
                                  (int_of_node slot_id)
                                  (Fmt.fmt_to_str Ast.fmt_slot_key k)));
                    drop_slot_in_current_frame
                      (cell_of_block_slot slot_id) slot
              end
              slots

  and trans_stmt (stmt:Ast.stmt) : unit =
    (* Helper to localize errors by stmt, at minimum. *)
    try
      iflog
        begin
          fun _ ->
            let s = Fmt.fmt_to_str Ast.fmt_stmt_body stmt in
              log cx "translating stmt: %s" s;
              annotate s;
        end;
      Stack.push stmt.id curr_stmt;
      (in_quad_category "stmt"
         (fun _ -> trans_stmt_full stmt));
      begin
        match stmt.node with
            Ast.STMT_be _
          | Ast.STMT_ret _ -> ()
          | _ -> drop_slots_at_curr_stmt();
      end;
      ignore (Stack.pop curr_stmt);
    with
        Semant_err (None, msg) -> raise (Semant_err ((Some stmt.id), msg))


  and maybe_init (id:node_id) (action:string) (dst:Ast.lval) : bool =
    let b = Hashtbl.mem cx.ctxt_stmt_is_init id in
    let act = if b then ("initializing-" ^ action) else action in
      iflog
        (fun _ ->
           annotate (Printf.sprintf "%s on dst lval %a"
                       act Ast.sprintf_lval dst));
      b


  and get_current_output_cell_and_slot _ : (Il.cell * Ast.slot) =
    let curr_fty =
      need_ty_fn (Hashtbl.find cx.ctxt_all_item_types (current_fn()))
    in
    let curr_args = get_args_for_current_frame () in
    let curr_outptr =
      get_element_ptr curr_args Abi.calltup_elt_out_ptr
    in
    let dst_cell = deref curr_outptr in
    let dst_slot = (fst curr_fty).Ast.sig_output_slot in
      (dst_cell, dst_slot)

  and trans_set_outptr (at:Ast.atom) : unit =
    let (dst_cell, dst_slot) = get_current_output_cell_and_slot () in
      trans_init_slot_from_atom
        CLONE_none dst_cell dst_slot at


  and trans_for_loop (fo:Ast.stmt_for) : unit =
    let ty_params = get_ty_params_of_current_frame () in
    let dst_slot_id = (fst (fo.Ast.for_slot)).id in
    let dst_slot = get_slot cx dst_slot_id in
    let dst_cell = cell_of_block_slot dst_slot_id in
    let seq = fo.Ast.for_seq in
    let (seq_cell, seq_ty) = trans_lval seq in
    let unit_ty = seq_unit_ty seq_ty in
      iter_seq_parts ty_params seq_cell seq_cell unit_ty
        (simplified_ty seq_ty = Ast.TY_str)
        begin
          fun _ src_cell unit_ty ->
            trans_init_slot_from_cell
              ty_params CLONE_none
              dst_cell dst_slot
              src_cell unit_ty;
            trans_block fo.Ast.for_body;
        end

  and trans_for_each_loop (stmt_id:node_id) (fe:Ast.stmt_for_each) : unit =
    let id = fe.Ast.for_each_body.id in
    let g = GLUE_loop_body id in
    let name = glue_str cx g in
    let fix = new_fixup name in
    let framesz = get_framesz cx id in
    let callsz = get_callsz cx id in
    let spill = Hashtbl.find cx.ctxt_spill_fixups id in
      push_new_emitter_with_vregs (Some id);
      iflog (fun _ -> annotate "prologue");
      abi.Abi.abi_emit_fn_prologue (emitter())
        framesz callsz nabi_rust (upcall_fixup "upcall_grow_task")
        false cx.ctxt_sess.Session.sess_minimal;
      write_frame_info_ptrs None;
      iflog (fun _ -> annotate "finished prologue");
      trans_block fe.Ast.for_each_body;
      trans_glue_frame_exit fix spill g;

      (* 
       * We've now emitted the body helper-fn. Next, set up a loop that
       * calls the iter and passes the helper-fn in.
       *)
      emit (Il.Enter
              (Hashtbl.find
                 cx.ctxt_block_fixups
                 fe.Ast.for_each_head.id));
      let (dst_slot, _) = fe.Ast.for_each_slot in
      let dst_cell = cell_of_block_slot dst_slot.id in
      let (flv, args) = fe.Ast.for_each_call in
      let ty_params =
        match htab_search cx.ctxt_call_lval_params (lval_base_id flv) with
            Some params -> params
          | None -> [| |]
      in
      let depth = get_stmt_depth cx stmt_id in
      let fc = { for_each_fixup = fix; for_each_depth = depth } in
        iflog (fun _ ->
                 log cx "for-each at depth %d\n" depth);
        let fn_ptr =
          trans_prepare_fn_call true cx dst_cell flv ty_params (Some fc) args
        in
          call_code (code_of_operand fn_ptr);
          emit Il.Leave;

  and trans_put (atom_opt:Ast.atom option) : unit =
    begin
      match atom_opt with
          None -> ()
        | Some at -> trans_set_outptr at
    end;
    let block_fptr = Il.Cell (get_iter_block_fn_for_current_frame ()) in
    let fp = get_iter_outer_frame_ptr_for_current_frame () in
    let vr = next_vreg_cell Il.voidptr_t in
      mov vr zero;
      trans_call_glue
        (code_of_operand block_fptr)
        None
        [| vr; fp |]
        None

  and trans_vec_append
      (dst_cell:Il.cell)
      (dst_ty:Ast.ty)
      (src_oper:Il.operand)
      (src_ty:Ast.ty)
      : unit =
    let elt_ty = seq_unit_ty dst_ty in
    let trailing_null = simplified_ty dst_ty = Ast.TY_str in
      match (simplified_ty dst_ty, simplified_ty src_ty) with
          (Ast.TY_str, Ast.TY_str)
        | (Ast.TY_vec _, Ast.TY_vec _)
            when (simplified_ty dst_ty) = (simplified_ty src_ty) ->

            let src_cell = need_cell src_oper in
            let src_vec = deref src_cell in
            let src_fill = get_element_ptr src_vec Abi.vec_elt_fill in

              aliasing true dst_cell
                begin
                  fun dst_vec_alias ->
                    trans_call_simple_static_glue
                      (get_vec_grow_glue ())
                      (get_ty_params_of_current_frame ())
                      [| get_tydesc None dst_ty;
                         get_tydesc None elt_ty;
                         dst_vec_alias;
                         src_fill; |]
                      None
                end;

              (*
               * By now, dst_cell points to a vec/str with room for us
               * to add to.
               *)

              let dst_vec = deref dst_cell in
              let dst_fill = get_element_ptr dst_vec Abi.vec_elt_fill in

                if trailing_null
                then sub_from dst_fill (imm 1L);

                (* Copy loop: *)
                let eltp_rty = Il.AddrTy (referent_type cx elt_ty) in
                let dptr = next_vreg_cell eltp_rty in
                let sptr = next_vreg_cell eltp_rty in
                let dlim = next_vreg_cell eltp_rty in
                let elt_sz = ty_sz_in_current_frame elt_ty in
                let dst_data =
                  get_element_ptr_dyn_in_current_frame
                    dst_vec Abi.vec_elt_data
                in
                let src_data =
                  get_element_ptr_dyn_in_current_frame
                    src_vec Abi.vec_elt_data
                in
                  lea dptr (fst (need_mem_cell dst_data));
                  lea sptr (fst (need_mem_cell src_data));
                  add_to dptr (Il.Cell dst_fill);
                  mov dlim (Il.Cell dptr);
                  add_to dlim (Il.Cell src_fill);
                  let fwd_jmp = mark () in
                    emit (Il.jmp Il.JMP Il.CodeNone);
                    let back_jmp_targ = mark () in
                      (* copy slot *)
                      trans_copy_ty
                        (get_ty_params_of_current_frame()) true
                        (deref dptr) elt_ty
                        (deref sptr) elt_ty;
                      add_to dptr elt_sz;
                      add_to sptr elt_sz;
                      patch fwd_jmp;
                      check_interrupt_flag ();
                      let back_jmp =
                        trans_compare_simple
                          Il.JB (Il.Cell dptr) (Il.Cell dlim)
                      in
                        List.iter
                          (fun j -> patch_existing j back_jmp_targ) back_jmp;
                        let v = next_vreg_cell word_sty in
                          mov v (Il.Cell src_fill);
                          add_to dst_fill (Il.Cell v);

        | (Ast.TY_str, e)
        | (Ast.TY_vec _, e)
            when e = simplified_ty elt_ty ->

            let elt_sz = ty_sz_in_current_frame elt_ty in
            let elt_sz_cell = Il.Reg (force_to_reg elt_sz) in
            let elt_sz = Il.Cell elt_sz_cell in

              aliasing true dst_cell
                begin
                  fun dst_vec_alias ->
                    trans_call_simple_static_glue
                      (get_vec_grow_glue ())
                      (get_ty_params_of_current_frame ())
                      [| get_tydesc None dst_ty;
                         get_tydesc None elt_ty;
                         dst_vec_alias;
                         elt_sz_cell; |]
                      None
                end;

              (* 
               * By now, dst_cell points to a vec/str with room for us
               * to add to.
               *)

              (* Reload dst vec, fill; might have changed. *)
              let dst_vec = deref dst_cell in
              let dst_fill = get_element_ptr dst_vec Abi.vec_elt_fill in

              let eltp_rty = Il.AddrTy (referent_type cx elt_ty) in
              let dptr = next_vreg_cell eltp_rty in
              let dst_data =
                get_element_ptr_dyn_in_current_frame
                  dst_vec Abi.vec_elt_data
              in
                lea dptr (fst (need_mem_cell dst_data));
                add_to dptr (Il.Cell dst_fill);
                if trailing_null
                then sub_from dptr elt_sz;
                trans_copy_ty
                  (get_ty_params_of_current_frame()) true
                  (deref dptr) elt_ty
                  (Il.Mem (force_to_mem src_oper)) elt_ty;
                add_to dptr elt_sz;
                if trailing_null
                then mov (deref dptr) zero_byte;
                add_to dst_fill elt_sz;

        | _ ->
            begin
              bug () "unsupported vector-append types %a += %a"
                Ast.sprintf_ty dst_ty
                Ast.sprintf_ty src_ty
            end


  and trans_copy_binop dst binop a_src =
    let (dst_cell, dst_ty) = trans_lval dst in
    let src_oper = trans_atom a_src in
      match dst_ty with
          Ast.TY_str
        | Ast.TY_vec _ when binop = Ast.BINOP_add ->
            trans_vec_append dst_cell dst_ty src_oper (atom_type cx a_src)
        | _ ->
            let (dst_cell, _) = deref_ty DEREF_none false dst_cell dst_ty in
            let bits = Il.operand_bits word_bits src_oper in
              (* 
               * FIXME: X86-ism going via a vreg; mem op= mem doesn't work and
               * IL lacks sufficient brains to cope just now.
               *)
            let src = Il.Reg (Il.next_vreg (emitter()), Il.ValTy bits) in
            let op = trans_binop binop in
              mov src src_oper;
              emit (Il.binary op dst_cell (Il.Cell dst_cell) (Il.Cell src));


  and trans_call id dst flv args =
    let init = maybe_init id "call" dst in
    let ty = lval_ty cx flv in
    let ty_params =
      match
        htab_search
          cx.ctxt_call_lval_params (lval_base_id flv)
      with
          Some params -> params
        | None -> [| |]
    in
      match simplified_ty ty with
          Ast.TY_fn _ ->
            let (dst_cell, _) = trans_lval_maybe_init init dst in
            let fn_ptr =
              trans_prepare_fn_call init cx dst_cell flv
                ty_params None args
            in
              call_code (code_of_operand fn_ptr)
        | _ -> bug () "Calling unexpected lval."


  and trans_log lev id a =
    match simplified_ty (atom_type cx a) with
        (* NB: If you extend this, be sure to update the
         * typechecking code in type.ml as well. *)
        Ast.TY_str -> trans_log_str lev a
      | Ast.TY_int | Ast.TY_uint | Ast.TY_bool
      | Ast.TY_char | Ast.TY_mach (TY_u8)
      | Ast.TY_mach (TY_u16) | Ast.TY_mach (TY_u32)
      | Ast.TY_mach (TY_i8) | Ast.TY_mach (TY_i16)
      | Ast.TY_mach (TY_i32) ->
          trans_log_int lev a
      | _ -> unimpl (Some id) "logging type"

  and trans_while (id:node_id) (sw:Ast.stmt_while) : unit =
    let (head_stmts, head_expr) = sw.Ast.while_lval in
    let fwd_jmp = mark () in
      emit (Il.jmp Il.JMP Il.CodeNone);
      let block_begin = mark () in
        Stack.push (Stack.create()) simple_break_jumps;
        trans_block sw.Ast.while_body;
        patch fwd_jmp;
        Array.iter trans_stmt head_stmts;
        check_interrupt_flag ();
        let flag = next_vreg_cell (Il.ValTy Il.Bits8) in
          mov flag imm_true;
          let true_jmps = trans_cond false head_expr in
            mov flag imm_false;
            List.iter patch true_jmps;
            begin
              begin
                match htab_search cx.ctxt_while_header_slots id with
                    None -> ()
                  | Some slots ->
                      let depth = get_stmt_depth cx id in
                        List.iter (drop_slot_by_id depth) slots
              end;
              let back_jmps =
                trans_compare_simple Il.JE (Il.Cell flag) imm_true
              in
                List.iter (fun j -> patch_existing j block_begin) back_jmps;
        end;
        Stack.iter patch (Stack.pop simple_break_jumps);


  and trans_stmt_full (stmt:Ast.stmt) : unit =
    match stmt.node with

        Ast.STMT_log a ->
          trans_log 1 stmt.id a

      | Ast.STMT_log_err a ->
          trans_log 0 stmt.id a

      | Ast.STMT_check_expr e ->
          trans_check_expr stmt.id e

      | Ast.STMT_yield ->
          trans_yield ()

      | Ast.STMT_fail ->
          trans_fail ()

      | Ast.STMT_join task ->
          trans_join task

      | Ast.STMT_send (chan,src) ->
          trans_send chan src

      | Ast.STMT_spawn (dst, domain, name, plv, args) ->
          trans_spawn (maybe_init stmt.id "spawn" dst) dst
            domain name plv args

      | Ast.STMT_recv (dst, chan) ->
          trans_recv (maybe_init stmt.id "recv" dst) dst chan

      | Ast.STMT_copy (dst, e_src) ->
          trans_copy (maybe_init stmt.id "copy" dst) dst e_src

      | Ast.STMT_copy_binop (dst, binop, a_src) ->
          trans_copy_binop dst binop a_src

      | Ast.STMT_call (dst, flv, args) ->
          trans_call stmt.id dst flv args

      | Ast.STMT_bind (dst, flv, args) ->
          begin
            let init = maybe_init stmt.id "bind" dst in
              match lval_ty cx flv with
                  Ast.TY_fn (tsig, _) ->
                    trans_bind_fn
                      init (call_ctrl flv) stmt.id dst flv tsig args
                | _ -> bug () "Binding unexpected lval."
          end

      | Ast.STMT_new_rec (dst, atab, base) ->
          let init = maybe_init stmt.id "new rec" dst in
          let (dst_cell, dst_ty) = trans_lval_maybe_init init dst in
          let (trec, dst_tys) =
            match dst_ty with
                Ast.TY_rec trec -> (trec, Array.map snd trec)
              | _ ->
                  bugi cx stmt.id
                    "non-rec destination type in stmt_new_rec"
          in
            begin
              drop_existing_if_not_init init dst_cell dst_ty;
              match base with
                  None ->
                    let atoms = Array.map (fun (_, _, atom) -> atom) atab in
                      trans_init_structural_from_atoms
                        dst_cell dst_tys atoms
                | Some base_lval ->
                    trans_new_rec_update
                      dst_cell dst_tys trec atab base_lval
            end

      | Ast.STMT_new_tup (dst, elems) ->
          let init = maybe_init stmt.id "new tup" dst in
          let (dst_cell, dst_ty) = trans_lval_maybe_init init dst in
          let dst_tys =
            match dst_ty with
                Ast.TY_tup ttup -> ttup
              | _ ->
                  bugi cx stmt.id
                    "non-tup destination type in stmt_new_tup"
          in
          let atoms = Array.map snd elems in
            drop_existing_if_not_init init dst_cell dst_ty;
            trans_init_structural_from_atoms dst_cell dst_tys atoms


      | Ast.STMT_new_str (dst, s) ->
          let init = maybe_init stmt.id "new str" dst in
            trans_new_str init dst s stmt.id

      | Ast.STMT_new_vec (dst, _, atoms) ->
          let init = maybe_init stmt.id "new vec" dst in
            trans_new_vec init dst atoms

      | Ast.STMT_new_port dst ->
          let init = maybe_init stmt.id "new port" dst in
            trans_new_port init dst

      | Ast.STMT_new_chan (dst, port) ->
          let init = maybe_init stmt.id "new chan" dst in
          begin
            match port with
                None ->
                  let (dst_cell, _) =
                    trans_lval_maybe_init init dst
                  in
                    mov dst_cell imm_false
              | Some p ->
                  trans_new_chan init dst p
          end

      | Ast.STMT_new_box (dst, _, src) ->
          let init = maybe_init stmt.id "new box" dst in
            trans_new_box init dst src

      | Ast.STMT_block block ->
          trans_block block

      | Ast.STMT_while sw ->
          trans_while stmt.id sw

      | Ast.STMT_if si ->
          let skip_thn_jmps = trans_cond true si.Ast.if_test in
            trans_block si.Ast.if_then;
            begin
              match si.Ast.if_else with
                  None -> List.iter patch skip_thn_jmps
                | Some els ->
                    let skip_els_jmp = mark () in
                      begin
                        emit (Il.jmp Il.JMP Il.CodeNone);
                        List.iter patch skip_thn_jmps;
                        trans_block els;
                        patch skip_els_jmp
                      end
            end

      | Ast.STMT_check (preds, calls) ->
          Array.iteri
            (fun i (fn, args) -> trans_call_pred_and_check preds.(i) fn args)
            calls

      | Ast.STMT_ret atom_opt ->
          if get_stmt_depth cx stmt.id > 0
          then unimpl (Some stmt.id) "ret within iterator-block";
          begin
            match atom_opt with
                None -> ()
              | Some at -> trans_set_outptr at
          end;
          drop_slots_at_curr_stmt();
          Stack.push (mark()) (Stack.top epilogue_jumps);
          emit (Il.jmp Il.JMP Il.CodeNone)

      | Ast.STMT_be (flv, args) ->
          let ty_params =
            match htab_search cx.ctxt_call_lval_params (lval_base_id flv) with
                Some params -> params
              | None -> [| |]
            in
          let (dst_cell, _) = get_current_output_cell_and_slot () in
            trans_be_fn cx dst_cell flv ty_params args

      | Ast.STMT_break ->
          if get_stmt_depth cx stmt.id > 0
          then unimpl (Some stmt.id) "break within iterator-block";
          drop_slots_at_curr_stmt();
          Stack.push (mark()) (Stack.top simple_break_jumps);
          emit (Il.jmp Il.JMP Il.CodeNone);

      | Ast.STMT_put atom_opt ->
          trans_put atom_opt

      | Ast.STMT_alt_tag stmt_alt_tag -> trans_alt_tag stmt_alt_tag

      | Ast.STMT_decl _ -> ()

      | Ast.STMT_for fo ->
          trans_for_loop fo

      | Ast.STMT_for_each fe ->
          trans_for_each_loop stmt.id fe

      | _ -> bugi cx stmt.id "unhandled form of statement in trans_stmt %a"
          Ast.sprintf_stmt stmt

  and capture_emitted_quads (fix:fixup) (node:node_id) : unit =
    let e = emitter() in
    let n_vregs = Il.num_vregs e in
    let quads = emitted_quads e in
    let name = path_name () in
    let f =
      if Stack.is_empty curr_file
      then bugi cx node "missing file scope when capturing quads."
      else Stack.top curr_file
    in
    let item_code = Hashtbl.find cx.ctxt_file_code f in
      begin
        iflog (fun _ ->
                 log cx "capturing quads for item #%d" (int_of_node node);
                 annotate_quads name);
        let vr_s =
          match htab_search cx.ctxt_spill_fixups node with
              None -> (assert (n_vregs = 0); None)
            | Some spill -> Some (n_vregs, spill)
        in
        let code = { code_fixup = fix;
                     code_quads = quads;
                     code_vregs_and_spill = vr_s; }
        in
          htab_put item_code node code;
          htab_put cx.ctxt_all_item_code node code
      end

  and get_frame_glue_fns (fnid:node_id) : Il.operand =
    let n_ty_params = n_item_ty_params cx fnid in
    let get_frame_glue glue inner =
      get_mem_glue glue
        begin
          (* `mem` here is a pointer to the frame we are marking, dropping,
             or relocing, etc. *)
          fun mem ->
            iter_frame_and_arg_slots cx fnid
              begin
                fun key slot_id slot ->
                  match htab_search cx.ctxt_slot_offsets slot_id with
                      Some off when not (slot_is_obj_state cx slot_id) ->
                        let referent_type = slot_id_referent_type slot_id in
                          (*
                           * This might look as though we're always taking the
                           * pointer-to-frame and giving it the type of the
                           * frame/arg of interest, but this is because our
                           * deref_off a few lines later takes the referent
                           * type of the given poiinter (`st`) as the referent
                           * type of the mem-offset-from-the-given-pointer
                           * that it returns.
                           *)
                        let fp_cell = rty_ptr_at mem referent_type in
                        let (fp, st) = force_to_reg (Il.Cell fp_cell) in
                        let ty_params =
                          get_ty_params_of_frame fnid fp n_ty_params
                        in
                        let slot_cell =
                          deref_off_sz ty_params (Il.Reg (fp,st)) off
                        in
                          inner key slot_id ty_params slot slot_cell
                    | _ -> ()
              end
        end
    in
    let frame_has_aliases =
      let r = ref false in
        iter_frame_and_arg_slots cx fnid
          begin
            fun _ _ slot ->
              match slot.Ast.slot_mode with
                  Ast.MODE_alias -> r := true
                | _ -> ()
          end;
        !r
    in
    let frame_points_to_heap =
      let r = ref false in
        iter_frame_and_arg_slots cx fnid
          begin
            fun _ slot_id _ ->
              if type_points_to_heap cx (slot_ty (get_slot cx slot_id))
              then r := true
          end;
        !r
    in
    let null_word = Asm.WORD (word_ty_mach, Asm.IMM 0L) in

    trans_crate_rel_data_operand
      (DATA_frame_glue_fns fnid)
      begin
        fun _ ->
          let mark_frame_word =
            if frame_points_to_heap
            then
              crate_rel_word
                begin
                  get_frame_glue (GLUE_mark_frame fnid)
                    begin
                      fun _ _ ty_params slot slot_cell ->
                        mark_slot ty_params slot_cell slot
                    end
                end
            else
              null_word
          in

          let drop_frame_word =
            if frame_points_to_heap
            then
              crate_rel_word
                begin
                  get_frame_glue (GLUE_drop_frame fnid)
                    begin
                      fun _ _ ty_params slot slot_cell ->
                        drop_slot ty_params slot_cell slot
                    end
                end
            else
              null_word
          in

          let reloc_frame_word =
            if frame_has_aliases
            then
              crate_rel_word
                begin
                  get_frame_glue (GLUE_reloc_frame fnid)
                    begin
                      fun _ _ _ _ _ ->
                        ()
                    end
                end
            else
              null_word
          in
            Asm.SEQ [|
              (* 
               * NB: this must match the struct-offsets given in ABI
               * & rust runtime library.
               *)
              mark_frame_word;
              drop_frame_word;
              reloc_frame_word;
            |]
      end
  in

  let trans_frame_entry
      (fnid:node_id)
      (obj_fn:bool)
      (yield_check:bool)
      : unit =
    let framesz = get_framesz cx fnid in
    let callsz = get_callsz cx fnid in
      Stack.push (Stack.create()) epilogue_jumps;
      push_new_emitter_with_vregs (Some fnid);
      iflog (fun _ -> annotate "prologue");
      iflog (fun _ -> annotate (Printf.sprintf
                                  "framesz %s"
                                  (string_of_size framesz)));
      iflog (fun _ -> annotate (Printf.sprintf
                                  "callsz %s"
                                  (string_of_size callsz)));
      abi.Abi.abi_emit_fn_prologue
        (emitter()) framesz callsz nabi_rust
        (upcall_fixup "upcall_grow_task") obj_fn
        cx.ctxt_sess.Session.sess_minimal;

      write_frame_info_ptrs (Some fnid);
      if yield_check
      then check_interrupt_flag ();
      iflog (fun _ -> annotate "finished prologue");
  in

  let trans_frame_exit (fnid:node_id) (drop_args:bool) : unit =
    Stack.iter patch (Stack.pop epilogue_jumps);
    if drop_args
    then
      begin
        iflog (fun _ -> annotate "drop args");
        iter_arg_slots cx fnid callee_drop_slot;
      end;
    iflog (fun _ -> annotate "epilogue");
    abi.Abi.abi_emit_fn_epilogue (emitter());
    capture_emitted_quads (get_fn_fixup cx fnid) fnid;
    pop_emitter ()
  in

  let trans_fn
      (fnid:node_id)
      (body:Ast.block)
      (obj_fn:bool)
      : unit =
    trans_frame_entry fnid obj_fn true;
    trans_block body;
    trans_frame_exit fnid true;
  in

  let trans_obj_ctor
      (obj_id:node_id)
      (header:Ast.header_slots)
      : unit =
    trans_frame_entry obj_id false false;

    let all_args_rty = current_fn_args_rty None in
    let all_args_cell = caller_args_cell all_args_rty in
    let frame_args =
      get_element_ptr_dyn_in_current_frame
        all_args_cell Abi.calltup_elt_args
    in
    let frame_ty_params =
      get_element_ptr_dyn_in_current_frame
        all_args_cell Abi.calltup_elt_ty_params
    in

    let obj_fields_tup =
      Array.map (fun (sloti,_) -> (slot_ty sloti.node)) header
    in
    let obj_fields_ty = Ast.TY_tup obj_fields_tup in
    let obj_body_ty = Ast.TY_tup [| Ast.TY_type; obj_fields_ty |] in
    let box_ptr_ty = Ast.TY_box obj_body_ty in
    let box_ptr_rty = referent_type cx box_ptr_ty in
    let box_malloc_sz = box_allocation_size box_ptr_ty in

    let ctor_ty = Hashtbl.find cx.ctxt_all_item_types obj_id in
    let obj_ty =
      slot_ty (fst (need_ty_fn ctor_ty)).Ast.sig_output_slot
    in

    let vtbl_ptr = get_obj_vtbl obj_id in
    let _ =
      iflog (fun _ -> annotate "calculate vtbl-ptr from displacement")
    in
    let vtbl_cell = crate_rel_to_ptr vtbl_ptr Il.CodeTy in

    let _ = iflog (fun _ -> annotate "load destination obj pair ptr") in
    let dst_pair_cell = deref (ptr_at (fp_imm out_mem_disp) obj_ty) in
    let dst_pair_item_cell =
      get_element_ptr dst_pair_cell Abi.obj_field_vtbl
    in
    let dst_pair_box_cell =
      get_element_ptr dst_pair_cell Abi.obj_field_box
    in

      (* Load first cell of pair with vtbl ptr.*)
      iflog (fun _ -> annotate "mov vtbl-ptr to obj.item cell");
      mov dst_pair_item_cell (Il.Cell vtbl_cell);

      (* Load second cell of pair with pointer to fresh body tuple.*)
      iflog (fun _ -> annotate "malloc state-tuple to obj.box-ptr cell");
      trans_malloc dst_pair_box_cell box_malloc_sz zero;

      (* Copy rc, tydesc, args into the obj. *)
      let box_ptr = next_vreg_cell (need_scalar_ty box_ptr_rty) in
        iflog (fun _ -> annotate "load obj.box ptr to vreg");
        mov box_ptr (Il.Cell dst_pair_box_cell);
        let box = deref box_ptr in
        let refcnt =
          get_element_ptr_dyn_in_current_frame box
            Abi.box_rc_field_refcnt
        in
        let body =
          get_element_ptr_dyn_in_current_frame box
            Abi.box_rc_field_body
        in
        let obj_tydesc =
          get_element_ptr_dyn_in_current_frame body Abi.obj_body_elt_tydesc
        in
        let obj_fields =
          get_element_ptr_dyn_in_current_frame body Abi.obj_body_elt_fields
        in
          iflog (fun _ -> annotate "write refcnt=1 to obj box");
          mov refcnt one;
          iflog (fun _ -> annotate "write tydesc to obj body");
          mov obj_tydesc
            (Il.Cell (get_tydesc
                        (Some obj_id)
                        (Ast.TY_tup obj_fields_tup)));
          iflog (fun _ -> annotate "copy ctor args to obj body fields");
          trans_copy_tup
            frame_ty_params true
            obj_fields frame_args obj_fields_tup;
          (* We have to do something curious here: we can't drop the
           * arg slots directly as in the normal frame-exit sequence,
           * because the arg slot ids are actually given layout
           * positions inside the object state, and are at different
           * offsets within that state than within the current
           * frame. So we manually drop the argument slots here,
           * without mentioning the slot ids.
           *)
          Array.iteri
            (fun i (sloti, _) ->
               let cell =
                 get_element_ptr_dyn_in_current_frame
                   frame_args i
               in
                 drop_slot frame_ty_params cell sloti.node)
            header;
          trans_frame_exit obj_id false;
  in

  let string_of_name_component (nc:Ast.name_component) : string =
    match nc with
        Ast.COMP_ident i -> i
      | _ -> bug ()
          "Trans.string_of_name_component on non-COMP_ident"
  in


  let trans_static_name_components
      (ncs:Ast.name_component list)
      : Il.operand =
    let f nc =
      trans_crate_rel_static_string_frag (string_of_name_component nc)
    in
      trans_crate_rel_data_operand
        (DATA_name (name_of ncs))
        (fun _ -> Asm.SEQ (Array.append
                             (Array.map f (Array.of_list ncs))
                             [| Asm.WORD (word_ty_mach, Asm.IMM 0L) |]))
  in

  let trans_required_fn (fnid:node_id) (blockid:node_id option) : unit =
    trans_frame_entry fnid false false;
    begin
      match blockid with
          None -> ()
        | Some blockid ->
            emit (Il.Enter (Hashtbl.find cx.ctxt_block_fixups blockid));
    end;
    let (ilib, conv) = Hashtbl.find cx.ctxt_required_items fnid in
    let lib_num =
      htab_search_or_add cx.ctxt_required_lib_num ilib
        (fun _ -> Hashtbl.length cx.ctxt_required_lib_num)
    in
    let f = next_vreg_cell (Il.AddrTy (Il.CodeTy)) in
    let n_ty_params = n_item_ty_params cx fnid in
    let args_rty = direct_call_args_referent_type cx fnid in
    let caller_args_cell = caller_args_cell args_rty in
      begin
        match ilib with
            REQUIRED_LIB_rust ls ->
              begin
                let c_sym_num =
                  htab_search_or_add cx.ctxt_required_c_sym_num
                    (ilib, "rust_crate")
                    (fun _ -> Hashtbl.length cx.ctxt_required_c_sym_num)
                in
                let rust_sym_num =
                  htab_search_or_add cx.ctxt_required_rust_sym_num fnid
                    (fun _ -> Hashtbl.length cx.ctxt_required_rust_sym_num)
                in
                let path_elts = stk_elts_from_bot cx.ctxt_curr_path in
                let _ =
                  assert (ls.required_prefix < (List.length path_elts))
                in
                let relative_path_elts =
                  list_drop ls.required_prefix path_elts
                in
                let libstr = trans_static_string ls.required_libname in
                let relpath =
                  trans_static_name_components relative_path_elts
                in
                  trans_upcall "upcall_require_rust_sym" f
                    [| Il.Cell (curr_crate_ptr());
                       imm (Int64.of_int lib_num);
                       imm (Int64.of_int c_sym_num);
                       imm (Int64.of_int rust_sym_num);
                       libstr;
                       relpath |];

                  trans_copy_forward_args args_rty;

                  call_code (code_of_operand (Il.Cell f));
              end

          | REQUIRED_LIB_c ls ->
              begin
                let c_sym_str =
                  match htab_search cx.ctxt_required_syms fnid with
                      Some s -> s
                    | None ->
                        string_of_name_component
                          (Stack.top cx.ctxt_curr_path)
                in
                let c_sym_num =
                  (* FIXME: permit remapping symbol names to handle
                   * mangled variants.
                   *)
                  htab_search_or_add cx.ctxt_required_c_sym_num
                    (ilib, c_sym_str)
                    (fun _ -> Hashtbl.length cx.ctxt_required_c_sym_num)
                in
                let libstr = trans_static_string ls.required_libname in
                let symstr = trans_static_string c_sym_str in
                let check_rty_sz rty =
                  let sz = force_sz (Il.referent_ty_size word_bits rty) in
                    if sz = 0L || sz = word_sz
                    then ()
                    else bug () "bad arg or ret cell size for native require"
                in
                let out =
                  get_element_ptr caller_args_cell Abi.calltup_elt_out_ptr
                in
                let _ = check_rty_sz (pointee_type out) in
                let args =
                  let ty_params_cell =
                    get_element_ptr caller_args_cell Abi.calltup_elt_ty_params
                  in
                  let args_cell =
                    get_element_ptr caller_args_cell Abi.calltup_elt_args
                  in
                  let n_args =
                    match args_cell with
                        Il.Mem (_, Il.StructTy elts) -> Array.length elts
                      | _ -> bug () "non-StructTy in Trans.trans_required_fn"
                  in
                  let mk_ty_param i =
                    Il.Cell (get_element_ptr ty_params_cell i)
                  in
                  let mk_arg i =
                    let arg = get_element_ptr args_cell i in
                    let _ = check_rty_sz (Il.cell_referent_ty arg) in
                      Il.Cell arg
                  in
                    Array.append
                      (Array.init n_ty_params mk_ty_param)
                      (Array.init n_args mk_arg)
                in
                let nabi = { nabi_convention = conv;
                             nabi_indirect = true }
                in
                  if conv <> CONV_rust
                  then assert (n_ty_params = 0);
                  trans_upcall "upcall_require_c_sym" f
                    [| Il.Cell (curr_crate_ptr());
                       imm (Int64.of_int lib_num);
                       imm (Int64.of_int c_sym_num);
                       libstr;
                       symstr |];

                  abi.Abi.abi_emit_native_call_in_thunk
                    (emitter())
                    (if pointee_type out = Il.NilTy then None else Some out)
                    nabi
                    (Il.Cell f)
                    args;
              end

          | _ -> bug ()
              "Trans.required_rust_fn on unexpected form of require library"
      end;
      if blockid <> None
      then emit Il.Leave;
      match ilib with
          REQUIRED_LIB_rust _ ->
            trans_frame_exit fnid false;
        | REQUIRED_LIB_c _ ->
            trans_frame_exit fnid true;
        | _ -> bug ()
            "Trans.required_rust_fn on unexpected form of require library"
  in

  let trans_tag_fn
      (n:Ast.ident)
      (tagid:node_id)
      (tag:(Ast.header_slots * opaque_id * int))
      : unit =
    trans_frame_entry tagid false false;
    trace_str cx.ctxt_sess.Session.sess_trace_tag
      ("in tag constructor " ^ n);
    let (header_tup, oid, i) = tag in
    let tinfo = Hashtbl.find cx.ctxt_all_tag_info oid in
    let (n, _, _) = Hashtbl.find tinfo.tag_nums i in
    let _ = iflog (fun _ -> log cx "tag variant: %s -> tag value #%d" n i) in
    let (dst_cell, dst_slot) = get_current_output_cell_and_slot() in
    let dst_cell = deref_slot true dst_cell dst_slot in
    let tag_cell = get_element_ptr dst_cell Abi.tag_elt_discriminant in
    let union_cell =
      get_element_ptr_dyn_in_current_frame dst_cell Abi.tag_elt_variant
    in
    let tag_body_cell = get_variant_ptr union_cell i in
    let tag_body_rty = snd (need_mem_cell tag_body_cell) in
    let ty_params = get_ty_params_of_current_frame() in
      (* A clever compiler will inline this. We are not clever. *)
      iflog (fun _ -> annotate (Printf.sprintf "write tag #%d" i));
      mov tag_cell (imm (Int64.of_int i));
      iflog (fun _ -> annotate ("copy tag-content tuple: tag_body_rty=" ^
                                  (Il.string_of_referent_ty tag_body_rty)));
      Array.iteri
        begin
          fun i (sloti, _) ->
            let slot = get_slot cx sloti.id in
            let ty = slot_ty slot in
              trans_copy_ty
                ty_params
                true
                (get_element_ptr_dyn ty_params tag_body_cell i) ty
                (deref_slot false (cell_of_block_slot sloti.id) slot) ty;
        end
        header_tup;
      trace_str cx.ctxt_sess.Session.sess_trace_tag
        ("finished tag constructor " ^ n);
      trans_frame_exit tagid true;
  in

  let trans_tag
      (n:Ast.ident)
      (tagid:node_id)
      (tag:(Ast.header_slots * opaque_id * int))
      : unit =
    let (header_tup, _, _) = tag in
      if Array.length header_tup <> 0
      then trans_tag_fn n tagid tag
  in

  let enter_file_for id =
    if Hashtbl.mem cx.ctxt_item_files id
    then Stack.push id curr_file
  in

  let leave_file_for id =
    if Hashtbl.mem cx.ctxt_item_files id
    then
      if Stack.is_empty curr_file
      then bugi cx id "Missing source file on file-scope exit."
      else ignore (Stack.pop curr_file)
  in

  let visit_defined_mod_item_pre n _ i =
    iflog (fun _ -> log cx "translating defined item #%d = %s"
             (int_of_node i.id) (path_name()));
    match i.node.Ast.decl_item with
        Ast.MOD_ITEM_fn f -> trans_fn i.id f.Ast.fn_body false
      | Ast.MOD_ITEM_tag t -> trans_tag n i.id t
      | Ast.MOD_ITEM_obj ob ->
          trans_obj_ctor i.id
            (Array.map (fun (sloti,ident) ->
                          ({sloti with node = get_slot cx sloti.id},ident))
               ob.Ast.obj_state)
      | _ -> ()
  in

  let visit_required_mod_item_pre _ _ i =
    iflog (fun _ -> log cx "translating required item #%d = %s"
             (int_of_node i.id) (path_name()));
    match i.node.Ast.decl_item with
        Ast.MOD_ITEM_fn f ->
          trans_required_fn i.id (Some f.Ast.fn_body.id)
      | Ast.MOD_ITEM_tag (hslots, _, _) ->
          if Array.length hslots = 0
          then ()
          else trans_required_fn i.id None
      | Ast.MOD_ITEM_mod _ -> ()
      | Ast.MOD_ITEM_type _ -> ()
      | _ -> unimpl (Some i.id)
          "unsupported type of require: %s" (path_name())
  in

  let visit_obj_drop_pre obj b =
    let g = GLUE_obj_drop obj.id in
    let fix =
      match htab_search cx.ctxt_glue_code g with
          Some code -> code.code_fixup
        | None -> bug () "visit_obj_drop_pre without assigned fixup"
    in
    let framesz = get_framesz cx b.id in
    let callsz = get_callsz cx b.id in
    let spill = Hashtbl.find cx.ctxt_spill_fixups b.id in
      push_new_emitter_with_vregs (Some b.id);
      iflog (fun _ -> annotate "prologue");
      abi.Abi.abi_emit_fn_prologue (emitter())
        framesz callsz nabi_rust (upcall_fixup "upcall_grow_task")
        true cx.ctxt_sess.Session.sess_minimal;
      write_frame_info_ptrs None;
      iflog (fun _ -> annotate "finished prologue");
      trans_block b;
      Hashtbl.remove cx.ctxt_glue_code g;
      trans_glue_frame_exit fix spill g;
      inner.Walk.visit_obj_drop_pre obj b
  in

  let visit_defined_obj_fn_pre _ _ fn =
    trans_fn fn.id fn.node.Ast.fn_body true
  in

  let visit_required_obj_fn_pre _ _ _ =
    ()
  in

  let visit_obj_fn_pre obj ident fn =
    enter_file_for fn.id;
    begin
      if Hashtbl.mem cx.ctxt_required_items fn.id
      then
        visit_required_obj_fn_pre obj ident fn
      else
        visit_defined_obj_fn_pre obj ident fn;
    end;
    inner.Walk.visit_obj_fn_pre obj ident fn
  in

  let visit_mod_item_pre n p i =
    enter_file_for i.id;
    begin
      if Hashtbl.mem cx.ctxt_required_items i.id
      then
        visit_required_mod_item_pre n p i
      else
        visit_defined_mod_item_pre n p i
    end;
    inner.Walk.visit_mod_item_pre n p i
  in

  let visit_mod_item_post n p i =
    inner.Walk.visit_mod_item_post n p i;
    leave_file_for i.id
  in

  let visit_obj_fn_post obj ident fn =
    inner.Walk.visit_obj_fn_post obj ident fn;
    leave_file_for fn.id
  in

  let visit_crate_pre crate =
    enter_file_for crate.id;
    inner.Walk.visit_crate_pre crate
  in

  let report_quads _ =
    if cx.ctxt_sess.Session.sess_report_quads
    then
      begin
        let cumulative = ref 0 in
          Printf.fprintf stdout "quads:\n\n";
          Array.iter
            begin
              fun name ->
                let t = Hashtbl.find quad_categories name in
                  Printf.fprintf stdout "%20s: %d\n" name (!t);
                  cumulative := (!cumulative) + (!t)
            end
            (sorted_htab_keys quad_categories);
          Printf.fprintf stdout "\n%20s: %d\n" "cumulative" (!cumulative)
      end
  in

  let visit_crate_post crate =

    inner.Walk.visit_crate_post crate;

    let emit_aux_global_glue cx glue fix fn =
      let glue_name = glue_str cx glue in
        push_new_emitter_without_vregs None;
        let e = emitter() in
          fn e;
          iflog (fun _ -> annotate_quads glue_name);
          if (Il.num_vregs e) != 0
          then bug () "%s uses nonzero vregs" glue_name;
          pop_emitter();
          let code =
            { code_fixup = fix;
              code_quads = emitted_quads e;
              code_vregs_and_spill = None; }
          in
            htab_put cx.ctxt_glue_code glue code
    in

    let tab_sz htab =
      Asm.WORD (word_ty_mach, Asm.IMM (Int64.of_int (Hashtbl.length htab)))
    in

    let crate_data =
      (cx.ctxt_crate_fixup,
       Asm.DEF
         (cx.ctxt_crate_fixup,
          Asm.SEQ [|
            (* 
             * NB: this must match the rust_crate structure
             * in the rust runtime library.
             *)
            crate_rel_word cx.ctxt_image_base_fixup;
            Asm.WORD (word_ty_mach, Asm.M_POS cx.ctxt_crate_fixup);

            crate_rel_word cx.ctxt_debug_abbrev_fixup;
            Asm.WORD (word_ty_mach, Asm.M_SZ cx.ctxt_debug_abbrev_fixup);

            crate_rel_word cx.ctxt_debug_info_fixup;
            Asm.WORD (word_ty_mach, Asm.M_SZ cx.ctxt_debug_info_fixup);

            crate_rel_word cx.ctxt_activate_fixup;
            crate_rel_word cx.ctxt_yield_fixup;
            crate_rel_word cx.ctxt_unwind_fixup;
            crate_rel_word cx.ctxt_gc_fixup;
            crate_rel_word cx.ctxt_exit_task_fixup;

            tab_sz cx.ctxt_required_rust_sym_num;
            tab_sz cx.ctxt_required_c_sym_num;
            tab_sz cx.ctxt_required_lib_num;

            Asm.WORD (word_ty_mach, Asm.IMM (Int64.of_int abi.Abi.abi_tag));
          |]))
    in

      (* Emit additional glue we didn't do elsewhere. *)
      emit_aux_global_glue cx GLUE_activate
        cx.ctxt_activate_fixup
        abi.Abi.abi_activate;

      emit_aux_global_glue cx GLUE_yield
        cx.ctxt_yield_fixup
        abi.Abi.abi_yield;

      emit_aux_global_glue cx GLUE_unwind
        cx.ctxt_unwind_fixup
        (fun e -> abi.Abi.abi_unwind
           e nabi_rust (upcall_fixup "upcall_exit"));

      emit_aux_global_glue cx GLUE_gc
        cx.ctxt_gc_fixup
        abi.Abi.abi_gc;

      ignore (get_exit_task_glue ());

      begin
        match abi.Abi.abi_get_next_pc_thunk with
            None -> ()
          | Some (_, fix, fn) ->
              emit_aux_global_glue cx GLUE_get_next_pc fix fn
      end;

      htab_put cx.ctxt_data
        DATA_crate crate_data;

      provide_existing_native cx SEG_data "rust_crate" cx.ctxt_crate_fixup;

      leave_file_for crate.id;

      report_quads()
  in

    { inner with
        Walk.visit_crate_pre = visit_crate_pre;
        Walk.visit_crate_post = visit_crate_post;
        Walk.visit_mod_item_pre = visit_mod_item_pre;
        Walk.visit_mod_item_post = visit_mod_item_post;
        Walk.visit_obj_fn_pre = visit_obj_fn_pre;
        Walk.visit_obj_fn_post = visit_obj_fn_post;
        Walk.visit_obj_drop_pre = visit_obj_drop_pre;
    }
;;


let fixup_assigning_visitor
    (cx:ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =

  let path_name (_:unit) : string =
    Fmt.fmt_to_str Ast.fmt_name (path_to_name cx.ctxt_curr_path)
  in

  let enter_file_for id =
    if Hashtbl.mem cx.ctxt_item_files id
    then
      begin
        let name =
          if Stack.is_empty cx.ctxt_curr_path
          then "crate root"
          else path_name()
        in
        htab_put cx.ctxt_file_fixups id (new_fixup name);
        if not (Hashtbl.mem cx.ctxt_file_code id)
        then htab_put cx.ctxt_file_code id (Hashtbl.create 0);
      end
  in

  let visit_mod_item_pre n p i =
    enter_file_for i.id;
    begin
      match i.node.Ast.decl_item with

          Ast.MOD_ITEM_tag _ ->
            htab_put cx.ctxt_fn_fixups i.id
              (new_fixup (path_name()));

        | Ast.MOD_ITEM_fn _ ->
            begin
              let path = path_to_name cx.ctxt_curr_path in
              let fixup =
                if (not cx.ctxt_sess.Session.sess_library_mode)
                  && (Some path) = cx.ctxt_main_name
                then
                  match cx.ctxt_main_fn_fixup with
                      None -> bug () "missing main fixup in trans"
                    | Some fix -> fix
                else
                  new_fixup (path_name())
              in
                htab_put cx.ctxt_fn_fixups i.id fixup;
            end

        | Ast.MOD_ITEM_obj _ ->
            htab_put cx.ctxt_fn_fixups i.id
              (new_fixup (path_name()));

        | _ -> ()
    end;
    inner.Walk.visit_mod_item_pre n p i
  in

  let visit_obj_fn_pre obj ident fn =
    htab_put cx.ctxt_fn_fixups fn.id
      (new_fixup (path_name()));
    inner.Walk.visit_obj_fn_pre obj ident fn
  in

  let visit_obj_drop_pre obj b =
    let g = GLUE_obj_drop obj.id in
    let fix = new_fixup (path_name()) in
    let tmp_code = { code_fixup = fix;
                     code_quads = [| |];
                     code_vregs_and_spill = None; } in
      htab_put cx.ctxt_glue_code g tmp_code;
      inner.Walk.visit_obj_drop_pre obj b
  in

  let visit_block_pre b =
    htab_put cx.ctxt_block_fixups b.id
      (new_fixup ("lexical block in " ^ (path_name())));
    inner.Walk.visit_block_pre b
  in

  let visit_crate_pre c =
    enter_file_for c.id;
    inner.Walk.visit_crate_pre c;
  in

  { inner with
      Walk.visit_crate_pre = visit_crate_pre;
      Walk.visit_mod_item_pre = visit_mod_item_pre;
      Walk.visit_obj_fn_pre = visit_obj_fn_pre;
      Walk.visit_obj_drop_pre = visit_obj_drop_pre;
      Walk.visit_block_pre = visit_block_pre; }


let process_crate
    (cx:ctxt)
    (crate:Ast.crate)
    : unit =
  let passes =
    [|
      (unreferenced_required_item_ignoring_visitor cx
         (fixup_assigning_visitor cx Walk.empty_visitor));
      (unreferenced_required_item_ignoring_visitor cx
         (trans_visitor cx Walk.empty_visitor))
    |];
  in
    log cx "translating crate";
    begin
      match cx.ctxt_main_name with
          None -> ()
        | Some m ->
            log cx "with main fn %a"
              Ast.sprintf_name m
    end;
    run_passes cx "trans" passes
      cx.ctxt_sess.Session.sess_log_trans log crate;
;;

(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
