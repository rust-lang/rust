(*
 * x86/ia32 instructions have 6 parts:
 *
 *    [pre][op][modrm][sib][disp][imm]
 *
 * [pre] = 0..4 bytes of prefix
 * [op] = 1..3 byte opcode
 * [modrm] = 0 or 1 byte: [mod:2][reg/op:3][r/m:3]
 * [sib] = 0 or 1 byte: [scale:2][index:3][base:3]
 * [disp] = 1, 2 or 4 byte displacement
 * [imm] = 1, 2 or 4 byte immediate
 *
 * So between 1 and 17 bytes total.
 *
 * We're not going to use sib, but modrm is worth discussing.
 *
 * The high two bits of modrm denote an addressing mode. The modes are:
 *
 *   00 - "mostly" *(reg)
 *   01 - "mostly" *(reg) + disp8
 *   10 - "mostly" *(reg) + disp32
 *   11 - reg
 *
 * The next-lowest 3 bits denote a specific register, or a subopcode if
 * there is a fixed register or only one operand. The instruction format
 * reference will say "/<n>" for some number n, if a fixed subopcode is used.
 * It'll say "/r" if the instruction uses this field to specify a register.
 *
 * The registers specified in this field are:
 *
 *   000 - EAX or XMM0
 *   001 - ECX or XMM1
 *   010 - EDX or XMM2
 *   011 - EBX or XMM3
 *   100 - ESP or XMM4
 *   101 - EBP or XMM5
 *   110 - ESI or XMM6
 *   111 - EDI or XMM7
 *
 * The final low 3 bits denote sub-modes of the primary mode selected
 * with the top 2 bits. In particular, they "mostly" select the reg that is
 * to be used for effective address calculation.
 *
 * For the most part, these follow the same numbering order: EAX, ECX, EDX,
 * EBX, ESP, EBP, ESI, EDI. There are two unusual deviations from the rule
 * though:
 *
 *  - In primary modes 00, 01 and 10, r/m=100 means "use SIB byte".  You can
 *    use (unscaled) ESP as the base register in these modes by appending the
 *    SIB byte 0x24. We do that in our rm_r operand-encoder function.
 *
 *  - In primary mode 00, r/m=101 means "just disp32", no register is
 *    involved.  There is no way to use EBP in primary mode 00. If you try, we
 *    just decay into a mode 01 with an appended 8-bit immediate displacement.
 *
 * Some opcodes are written 0xNN +rd. This means "we decided to chew up a
 * whole pile of opcodes here, with each opcode including a hard-wired
 * reference to a register". For example, POP is "0x58 +rd", which means that
 * the 1-byte insns 0x58..0x5f are chewed up for "POP EAX" ... "POP EDI"
 * (again, the canonical order of register numberings)
 *)

(*
 * Notes on register availability of x86:
 *
 * There are 8 GPRs but we use 2 of them for specific purposes:
 *
 *   - ESP always points to the current stack frame.
 *   - EBP always points to the current frame base.
 *
 * We tell IL that we have 6 GPRs then, and permit most register-register ops
 * on any of these 6, mostly-unconstrained.
 *
 *)


let log (sess:Session.sess) =
  Session.log "insn"
    sess.Session.sess_log_insn
    sess.Session.sess_log_out
;;

let iflog (sess:Session.sess) (thunk:(unit -> unit)) : unit =
  if sess.Session.sess_log_insn
  then thunk ()
  else ()
;;

open Common;;

exception Unrecognized
;;

let modrm m rm reg_or_subopcode =
  if (((m land 0b11) != m) or
        ((rm land 0b111) != rm) or
        ((reg_or_subopcode land 0b111) != reg_or_subopcode))
  then raise (Invalid_argument "X86.modrm_deref")
  else
    ((((m land 0b11) lsl 6)
      lor
      (rm land 0b111))
     lor
      ((reg_or_subopcode land 0b111) lsl 3))
;;

let modrm_deref_reg = modrm 0b00 ;;
let modrm_deref_disp32 = modrm 0b00 0b101 ;;
let modrm_deref_reg_plus_disp8 = modrm 0b01 ;;
let modrm_deref_reg_plus_disp32 = modrm 0b10 ;;
let modrm_reg = modrm 0b11 ;;

let slash0 = 0;;
let slash1 = 1;;
let slash2 = 2;;
let slash3 = 3;;
let slash4 = 4;;
let slash5 = 5;;
let slash6 = 6;;
let slash7 = 7;;


(*
 * Translate an IL-level hwreg number from 0..nregs into the 3-bit code number
 * used through the mod r/m byte and /r sub-register specifiers of the x86
 * ISA.
 *
 * See "Table 2-2: 32-Bit Addressing Forms with the ModR/M Byte", in the IA32
 * Architecture Software Developer's Manual, volume 2a.
 *)

let eax = 0
let ecx = 1
let ebx = 2
let esi = 3
let edi = 4
let edx = 5
let ebp = 6
let esp = 7

let code_eax = 0b000;;
let code_ecx = 0b001;;
let code_edx = 0b010;;
let code_ebx = 0b011;;
let code_esp = 0b100;;
let code_ebp = 0b101;;
let code_esi = 0b110;;
let code_edi = 0b111;;

let reg r =
  match r with
      0 -> code_eax
    | 1 -> code_ecx
    | 2 -> code_ebx
    | 3 -> code_esi
    | 4 -> code_edi
    | 5 -> code_edx
        (* Never assigned by the register allocator, but synthetic code uses
           them *)
    | 6 -> code_ebp
    | 7 -> code_esp
    | _ -> raise (Invalid_argument "X86.reg")
;;


let dwarf_eax = 0;;
let dwarf_ecx = 1;;
let dwarf_edx = 2;;
let dwarf_ebx = 3;;
let dwarf_esp = 4;;
let dwarf_ebp = 5;;
let dwarf_esi = 6;;
let dwarf_edi = 7;;

let dwarf_reg r =
  match r with
      0 -> dwarf_eax
    | 1 -> dwarf_ecx
    | 2 -> dwarf_ebx
    | 3 -> dwarf_esi
    | 4 -> dwarf_edi
    | 5 -> dwarf_edx
    | 6 -> dwarf_ebp
    | 7 -> dwarf_esp
    | _ -> raise (Invalid_argument "X86.dwarf_reg")

let reg_str r =
  match r with
      0 -> "eax"
    | 1 -> "ecx"
    | 2 -> "ebx"
    | 3 -> "esi"
    | 4 -> "edi"
    | 5 -> "edx"
    | 6 -> "ebp"
    | 7 -> "esp"
    | _ -> raise (Invalid_argument "X86.reg_str")
;;

(* This is a basic ABI. You might need to customize it by platform. *)
let (n_hardregs:int) = 6;;

(* Includes ebx, esi, edi; does *not* include ebp, which has ABI-specified
 * rules concerning its location and save/restore sequence.
 * 
 * See http://refspecs.freestandards.org/elf/abi386-4.pdf
 * Page 36, Figure 3-15 and friends.
 *)
let (n_callee_saves:int) = 3;;

let is_ty32 (ty:Il.scalar_ty) : bool =
  match ty with
      Il.ValTy (Il.Bits32) -> true
    | Il.AddrTy _ -> true
    | _ -> false
;;

let is_r32 (c:Il.cell) : bool =
  match c with
      Il.Reg (_, st) -> is_ty32 st
    | _ -> false
;;

let is_rm32 (c:Il.cell) : bool =
  match c with
      Il.Mem (_, Il.ScalarTy st) -> is_ty32 st
    | Il.Reg (_, st) -> is_ty32 st
    | _ -> false
;;

let is_ty8 (ty:Il.scalar_ty) : bool =
  match ty with
      Il.ValTy (Il.Bits8) -> true
    | _ -> false
;;

let is_m32 (c:Il.cell) : bool =
  match c with
      Il.Mem (_, Il.ScalarTy st) -> is_ty32 st
    | _ -> false
;;

let is_m8 (c:Il.cell) : bool =
  match c with
      Il.Mem (_, Il.ScalarTy st) -> is_ty8 st
    | _ -> false
;;

let is_ok_r8 (r:Il.hreg) : bool =
  (r == eax || r == ebx || r == ecx || r == edx)
;;

let is_r8 (c:Il.cell) : bool =
  match c with
      Il.Reg (Il.Hreg r, st) when is_ok_r8 r -> is_ty8 st
    | _ -> false
;;

let is_rm8 (c:Il.cell) : bool =
  match c with
      Il.Mem (_, Il.ScalarTy st) -> is_ty8 st
    | _ -> is_r8 c
;;


let emit_target_specific
    (e:Il.emitter)
    (q:Il.quad)
    : unit =
  let fixup = ref q.Il.quad_fixup in
  let put q' =
    Il.append_quad e { Il.quad_body = q';
                       Il.quad_fixup = (!fixup) };
    fixup := None;
  in
  let op_vreg op =
    Il.next_vreg_cell e (Il.operand_scalar_ty op)
  in
  let cell_vreg cell = op_vreg (Il.Cell cell) in
  let mem_vreg mem = cell_vreg (Il.Mem mem) in
  let movop = Il.default_mov q.Il.quad_body in
  let mov dst src =
    (* Decay mem-mem moves to use a vreg. *)
    match dst, src with
        Il.Mem dm, Il.Cell (Il.Mem _) ->
          let v = mem_vreg dm in
            put (Il.unary movop v src);
            put (Il.unary movop dst (Il.Cell v))
      | _ -> put (Il.unary movop dst src)
  in

  let hr_like_op hr op =
    Il.Reg (Il.Hreg hr, Il.operand_scalar_ty op)
  in
  let hr_like_cell hr c = hr_like_op hr (Il.Cell c) in
  let q = q.Il.quad_body in

    match q with
        Il.Binary ({ Il.binary_op = op;
                     Il.binary_dst = dst;
                     Il.binary_lhs = lhs;
                     Il.binary_rhs = rhs; } as b) ->
          begin
            match op with

                Il.IMUL | Il.UMUL
              | Il.IDIV | Il.UDIV
              | Il.IMOD | Il.UMOD ->
                  let dst_eax = hr_like_cell eax dst in
                  let lhs_eax = hr_like_op eax lhs in
                  let rhs_ecx = hr_like_op ecx rhs in
                    (* Horrible: we bounce mul/div/mod inputs off spill slots
                     * to ensure non-interference between the temporaries used
                     * during mem-base-reg reloads and the registers we're
                     * preparing.  *)
                  let next_spill_like op =
                    Il.Mem (Il.next_spill_slot e
                              (Il.ScalarTy (Il.operand_scalar_ty op)))
                  in
                  let is_eax cell =
                    match cell with
                        Il.Cell (Il.Reg (Il.Hreg hr, _)) -> hr = eax
                      | _ -> false
                  in
                    if is_eax lhs
                    then
                      mov rhs_ecx rhs
                    else
                      begin
                        let lhs_spill = next_spill_like lhs in
                        let rhs_spill = next_spill_like rhs in

                          mov lhs_spill lhs;
                          mov rhs_spill rhs;

                          mov lhs_eax (Il.Cell lhs_spill);
                          mov rhs_ecx (Il.Cell rhs_spill);
                      end;

                    put (Il.Binary
                           { b with
                               Il.binary_lhs = (Il.Cell lhs_eax);
                               Il.binary_rhs = (Il.Cell rhs_ecx);
                               Il.binary_dst = dst_eax; });
                    if dst <> dst_eax
                    then mov dst (Il.Cell dst_eax);

              | _ when (Il.Cell dst) <> lhs ->
                  mov dst lhs;
                  put (Il.Binary
                         { b with Il.binary_lhs = Il.Cell dst })

              | _ -> put q
          end

      | Il.Unary ({ Il.unary_op = op;
                    Il.unary_dst = dst;
                    Il.unary_src = src; } as u) ->
          begin
            match op with

                Il.UMOV | Il.IMOV ->
                  mov dst src

              (* x86 can only NEG or NOT in-place. *)
              | Il.NEG | Il.NOT when (Il.Cell dst) <> src ->
                  mov dst src;
                  put (Il.Unary { u with Il.unary_src = Il.Cell dst })

              | _ -> put q
          end

      | Il.Call c ->
          let dst_eax = hr_like_cell eax c.Il.call_dst in
            put (Il.Call { c with Il.call_dst = dst_eax });
            if c.Il.call_dst <> dst_eax
            then mov c.Il.call_dst (Il.Cell dst_eax)

      (* 
       * For the get-next-pc thunk hack to work, we need to lea an immptr
       * to eax, always.
       *)
      | Il.Lea ({ Il.lea_dst = dst;
                  Il.lea_src = Il.ImmPtr _  } as lea) ->
          let eax_dst = hr_like_cell eax dst in
            put (Il.Lea { lea with Il.lea_dst = eax_dst });
            if dst <> eax_dst
            then mov dst (Il.Cell eax_dst);

      | q -> put q
;;


let constrain_vregs (q:Il.quad) (hregs:(Il.vreg,Bits.t) Hashtbl.t) : unit =

  let involves_8bit_cell =
    let b = ref false in
    let qp_cell _ c =
      match c with
          Il.Reg (_, Il.ValTy Il.Bits8)
        | Il.Mem (_, Il.ScalarTy (Il.ValTy Il.Bits8)) ->
            (b := true; c)
        | _ -> c
    in
      ignore (Il.process_quad { Il.identity_processor with
                                  Il.qp_cell_read = qp_cell;
                                  Il.qp_cell_write = qp_cell } q);
      !b
  in

  let get_hregs v =
    htab_search_or_add hregs v (fun _ -> Bits.create n_hardregs true)
  in

  let qp_mem _ m = m in
  let qp_cell _ c =
    begin
      match c with
          Il.Reg (Il.Vreg v, _) when involves_8bit_cell ->
            (* 8-bit register cells must only be al, cl, dl, bl.
             * Not esi/edi. *)
            let hv = get_hregs v in
              List.iter (fun bad -> Bits.set hv bad false) [esi; edi]
        | _ -> ()
    end;
    c
  in
    begin
      match q.Il.quad_body with
          Il.Binary b ->
            begin
              match b.Il.binary_op with
                  (* Shifts *)
                | Il.LSL | Il.LSR | Il.ASR ->
                    begin
                      match b.Il.binary_rhs with
                          Il.Cell (Il.Reg (Il.Vreg v, _)) ->
                            let hv = get_hregs v in
                              (* Shift src has to be ecx. *)
                              List.iter
                                (fun bad -> Bits.set hv bad false)
                                [eax; edx; ebx; esi; edi]
                        | _ -> ()
                    end
                | _ -> ()
            end
        | _ -> ()
    end;
    ignore
      (Il.process_quad { Il.identity_processor with
                           Il.qp_mem = qp_mem;
                           Il.qp_cell_read = qp_cell;
                           Il.qp_cell_write = qp_cell } q)
;;


let clobbers (quad:Il.quad) : Il.hreg list =
  match quad.Il.quad_body with
      Il.Binary bin ->
        begin
          match bin.Il.binary_op with
              Il.IMUL | Il.UMUL
            | Il.IDIV | Il.UDIV -> [ edx ]
            | Il.IMOD | Il.UMOD -> [ edx ]
            | _ -> []
        end
    | Il.Unary un ->
        begin
          match un.Il.unary_op with
              Il.ZERO -> [ eax; edi; ecx ]
            | _ -> [ ]
        end
    | Il.Call _ -> [ eax; ecx; edx; ]
    | Il.Regfence -> [ eax; ecx; ebx; edx; edi; esi; ]
    | _ -> []
;;


let word_sz = 4L
;;

let word_bits = Il.Bits32
;;

let word_ty = TY_u32
;;

let annotate (e:Il.emitter) (str:string) =
  Hashtbl.add e.Il.emit_annotations e.Il.emit_pc str
;;

let c (c:Il.cell) : Il.operand = Il.Cell c ;;
let r (r:Il.reg) : Il.cell = Il.Reg ( r, (Il.ValTy word_bits) ) ;;
let h (x:Il.hreg) : Il.reg = Il.Hreg x ;;
let rc (x:Il.hreg) : Il.cell = r (h x) ;;
let ro (x:Il.hreg) : Il.operand = c (rc x) ;;
let vreg (e:Il.emitter) : (Il.reg * Il.cell) =
  let vr = Il.next_vreg e in
    (vr, (Il.Reg (vr, (Il.ValTy word_bits))))
;;
let imm (x:Asm.expr64) : Il.operand =
  Il.Imm (x, word_ty)
;;
let immi (x:int64) : Il.operand =
  imm (Asm.IMM x)
;;

let imm_byte (x:Asm.expr64) : Il.operand =
  Il.Imm (x, TY_u8)
;;
let immi_byte (x:int64) : Il.operand =
  imm_byte (Asm.IMM x)
;;


let byte_off_n (i:int) : Asm.expr64 =
  Asm.IMM (Int64.of_int i)
;;

let byte_n (reg:Il.reg) (i:int) : Il.cell =
  let imm = byte_off_n i in
  let mem = Il.RegIn (reg, Some imm) in
    Il.Mem (mem, Il.ScalarTy (Il.ValTy Il.Bits8))
;;

let word_off_n (i:int) : Asm.expr64 =
  Asm.IMM (Int64.mul (Int64.of_int i) word_sz)
;;

let word_at (reg:Il.reg) : Il.cell =
  let mem = Il.RegIn (reg, None) in
    Il.Mem (mem, Il.ScalarTy (Il.ValTy word_bits))
;;

let word_at_off (reg:Il.reg) (off:Asm.expr64) : Il.cell =
  let mem = Il.RegIn (reg, Some off) in
    Il.Mem (mem, Il.ScalarTy (Il.ValTy word_bits))
;;

let word_n (reg:Il.reg) (i:int) : Il.cell =
  word_at_off reg (word_off_n i)
;;

let reg_codeptr (reg:Il.reg) : Il.code =
  Il.CodePtr (Il.Cell (Il.Reg (reg, Il.AddrTy Il.CodeTy)))
;;

let word_n_low_byte (reg:Il.reg) (i:int) : Il.cell =
  let imm = word_off_n i in
  let mem = Il.RegIn (reg, Some imm) in
    Il.Mem (mem, Il.ScalarTy (Il.ValTy Il.Bits8))
;;

let wordptr_n (reg:Il.reg) (i:int) : Il.cell =
  let imm = word_off_n i in
  let mem = Il.RegIn (reg, Some imm) in
    Il.Mem (mem, Il.ScalarTy (Il.AddrTy (Il.ScalarTy (Il.ValTy word_bits))))
;;

let get_element_ptr = Il.get_element_ptr word_bits reg_str ;;

let establish_frame_base (e:Il.emitter) : unit =
    (* Establish i386-ABI-compliant frame base. *)
    Il.emit e (Il.Push (ro ebp));
    Il.emit e (Il.umov (rc ebp) (ro esp));
;;

let save_callee_saves (e:Il.emitter) : unit =
    Il.emit e (Il.Push (ro edi));
    Il.emit e (Il.Push (ro esi));
    Il.emit e (Il.Push (ro ebx));
;;

let restore_callee_saves (e:Il.emitter) : unit =
    Il.emit e (Il.Pop (rc ebx));
    Il.emit e (Il.Pop (rc esi));
    Il.emit e (Il.Pop (rc edi));
;;

let leave_frame (e:Il.emitter) : unit =
    Il.emit e (Il.Pop (rc ebp));
;;


(* restores registers from the frame base without updating esp:
 *   - restores the callee-saves: edi, esi, ebx
 *   - restores ebp to stored values from frame base
 *   - sets `retpc' register to stored retpc from frame base
 *   - sets `base' register to current fp
 *)
let restore_frame_regs (e:Il.emitter) (base:Il.reg) (retpc:Il.reg)
    : unit =
  let emit = Il.emit e in
  let mov dst src = emit (Il.umov dst src) in
    mov (r base) (ro ebp);
    mov (rc ebx) (c (word_n base (-3)));
    mov (rc esi) (c (word_n base (-2)));
    mov (rc edi) (c (word_n base (-1)));
    mov (rc ebp) (c (word_at base));
    mov (r retpc) (c (word_n base 1));
;;


(*
 * Our arrangement on x86 is this:
 *
 *   *ebp+8+(4*N)  = [argN   ]
 *   ...
 *   *ebp+16       = [arg2   ] = obj/closure ptr
 *   *ebp+12       = [arg1   ] = task ptr
 *   *ebp+8        = [arg0   ] = out ptr
 *   *ebp+4        = [retpc  ]
 *   *ebp          = [old_ebp]
 *   *ebp-4        = [old_edi]
 *   *ebp-8        = [old_esi]
 *   *ebp-12       = [old_ebx]
 *
 * For x86-cdecl:
 *
 *  %eax, %ecx, %edx are "caller save" registers
 *  %ebp, %ebx, %esi, %edi are "callee save" registers
 *
 *)

let frame_base_words = 2 (* eip,ebp *) ;;
let frame_base_sz = Int64.mul (Int64.of_int frame_base_words) word_sz;;

let frame_info_words = 2 (* crate ptr, crate-rel frame info disp *) ;;
let frame_info_sz = Int64.mul (Int64.of_int frame_info_words) word_sz;;

let implicit_arg_words = 3 (* task ptr, out ptr, closure ptr *);;
let implicit_args_sz = Int64.mul (Int64.of_int implicit_arg_words) word_sz;;

let callee_saves_sz = Int64.mul (Int64.of_int n_callee_saves) word_sz;;

let out_ptr = wordptr_n (Il.Hreg ebp) (frame_base_words);;
let task_ptr = wordptr_n (Il.Hreg ebp) (frame_base_words+1);;
let closure_ptr = wordptr_n (Il.Hreg ebp) (frame_base_words+2);;
let ty_param_n i =
  wordptr_n (Il.Hreg ebp) (frame_base_words + implicit_arg_words + i);;

let spill_slot (i:Il.spill) : Il.mem =
  let imm = (Asm.IMM
               (Int64.neg
                  (Int64.add
                     (Int64.add frame_info_sz callee_saves_sz)
                     (Int64.mul word_sz
                        (Int64.of_int (i+1))))))
  in
    Il.RegIn ((Il.Hreg ebp), Some imm)
;;


let get_next_pc_thunk_fixup = new_fixup "glue$get_next_pc"
;;

let emit_get_next_pc_thunk (e:Il.emitter) : unit =
  let sty = Il.AddrTy Il.CodeTy in
  let rty = Il.ScalarTy sty in
  let deref_esp = Il.Mem (Il.RegIn (Il.Hreg esp, None), rty) in
  let eax = (Il.Reg (Il.Hreg eax, sty)) in
    Il.emit_full e (Some get_next_pc_thunk_fixup)
      (Il.umov eax (Il.Cell deref_esp));
    Il.emit e Il.Ret;
;;

let get_next_pc_thunk : (Il.reg * fixup * (Il.emitter -> unit)) =
    (Il.Hreg eax, get_next_pc_thunk_fixup, emit_get_next_pc_thunk)
;;

let emit_c_call
    (e:Il.emitter)
    (ret:Il.cell)
    (tmp1:Il.reg)
    (tmp2:Il.reg)
    (nabi:nabi)
    (in_prologue:bool)
    (fptr:Il.code)
    (args:Il.operand array)
    : unit =

  let emit = Il.emit e in
  let mov dst src = emit (Il.umov dst src) in
  let imov dst src = emit (Il.imov dst src) in
  let add dst src = emit (Il.binary Il.ADD dst (Il.Cell dst) src) in
  let binary op dst imm = emit (Il.binary op dst (c dst) (immi imm)) in

  (* rust calls get task as arg0  *)
  let args =
    if nabi.nabi_convention = CONV_rust
    then Array.append [| c task_ptr |] args
    else args
  in
  let nargs = Array.length args in
  let arg_sz = Int64.mul (Int64.of_int nargs) word_sz
  in

    mov (r tmp1) (c task_ptr);               (* tmp1 = task from argv[-1] *)
    mov (r tmp2) (ro esp);                   (* tmp2 = esp                *)
    mov                                      (* task->rust_sp = tmp2      *)
      (word_n tmp1 Abi.task_field_rust_sp)
      (c (r tmp2));
    mov                                      (* esp = task->runtime_sp    *)
      (rc esp)
      (c (word_n tmp1 Abi.task_field_runtime_sp));

    binary Il.SUB (rc esp) arg_sz;           (* make room on the stack    *)
    binary Il.AND (rc esp)                   (* and 16-byte align sp      *)
      0xfffffffffffffff0L;

    Array.iteri
      begin
        fun i (arg:Il.operand) ->   (* write args to C stack     *)
          match arg with
              Il.Cell (Il.Mem (a, ty)) ->
                begin
                  match a with
                      Il.RegIn (Il.Hreg base, off) when base == esp ->
                        mov (r tmp1) (c (Il.Mem (Il.RegIn (tmp2, off), ty)));
                        mov (word_n (h esp) i) (c (r tmp1));
                    | _ ->
                        mov (r tmp1) arg;
                        mov (word_n (h esp) i) (c (r tmp1));
                end
            | Il.Imm (_, tm) when mach_is_signed tm ->
                imov (word_n (h esp) i) arg
            | _ ->
                mov (word_n (h esp) i) arg
      end
      args;

    match ret with
        Il.Mem (Il.RegIn (Il.Hreg base, _), _) when base == esp ->
          assert (not in_prologue);

          (* If ret is esp-relative, use a temporary register until we
             switched stacks. *)

          emit (Il.call (r tmp1) fptr);
          mov (r tmp2) (c task_ptr);
          mov (rc esp) (c (word_n tmp2 Abi.task_field_rust_sp));
          mov ret (c (r tmp1));

      | _ when in_prologue ->
          (*
           * We have to do something a little surprising here:
           * we're doing a 'grow' call so ebp is going to point
           * into a dead stack frame on call-return. So we
           * temporarily store task-ptr into ebp and then reload
           * esp *and* ebp via ebp->rust_sp on the other side of
           * the call.
           *)
          mov (rc ebp) (c task_ptr);
          emit (Il.call ret fptr);
          mov (rc esp) (c (word_n (h ebp) Abi.task_field_rust_sp));
          mov (rc ebp) (ro esp);
          add (rc ebp) (immi callee_saves_sz);

      | _ ->
          emit (Il.call ret fptr);
          mov (r tmp2) (c task_ptr);
          mov (rc esp) (c (word_n tmp2 Abi.task_field_rust_sp));
;;

let emit_void_prologue_call
    (e:Il.emitter)
    (nabi:nabi)
    (fn:fixup)
    (args:Il.operand array)
    : unit =
  let callee = Abi.load_fixup_codeptr e (h eax) fn true nabi.nabi_indirect in
    emit_c_call e (rc eax) (h edx) (h ecx) nabi true callee args
;;

let emit_native_call
    (e:Il.emitter)
    (ret:Il.cell)
    (nabi:nabi)
    (fn:fixup)
    (args:Il.operand array)
    : unit =

  let (tmp1, _) = vreg e in
  let (tmp2, _) = vreg e in
  let (freg, _) = vreg e in
  let callee = Abi.load_fixup_codeptr e freg fn true nabi.nabi_indirect in
    emit_c_call e ret tmp1 tmp2 nabi false callee args
;;

let emit_native_void_call
    (e:Il.emitter)
    (nabi:nabi)
    (fn:fixup)
    (args:Il.operand array)
    : unit =

  let (ret, _) = vreg e in
    emit_native_call e (r ret) nabi fn args
;;

let emit_native_call_in_thunk
    (e:Il.emitter)
    (ret:Il.cell option)
    (nabi:nabi)
    (fn:Il.operand)
    (args:Il.operand array)
    : unit =

  let emit = Il.emit e in
  let mov dst src = emit (Il.umov dst src) in

    begin
      match fn with
          (*
           * NB: old path, remove when/if you're sure you don't
           * want native-linker-symbol-driven requirements.
           *)
          Il.ImmPtr (fix, _) ->
            let code =
              Abi.load_fixup_codeptr e (h eax) fix true nabi.nabi_indirect
            in
              emit_c_call e (rc eax) (h edx) (h ecx) nabi false code args;

        | _ ->
            (*
             * NB: new path, ignores nabi_indirect, assumes
             * indirect via pointer from upcall_require_c_sym
             * or crate cache.
             *)
            mov (rc eax) fn;
            let cell = Il.Reg (h eax, Il.AddrTy Il.CodeTy) in
            let fptr = Il.CodePtr (Il.Cell cell) in
              emit_c_call e (rc eax) (h edx) (h ecx) nabi false fptr args;
    end;

    match ret with
        Some (Il.Reg (r, _)) ->
          mov (word_at r) (ro eax)
      | Some ret ->
          mov (rc edx) (c ret);
          mov (word_at (h edx)) (ro eax)
      | _ -> ()
;;


let crawl_stack_calling_glue
    (e:Il.emitter)
    (glue_field:int)
    : unit =

  let fp_n = word_n (Il.Hreg ebp) in
  let edi_n = word_n (Il.Hreg edi) in
  let emit = Il.emit e in
  let mov dst src = emit (Il.umov dst src) in
  let push x = emit (Il.Push x) in
  let pop x = emit (Il.Pop x) in
  let add x y = emit (Il.binary Il.ADD (rc x) (ro x) (ro y)) in
  let codefix fix = Il.CodePtr (Il.ImmPtr (fix, Il.CodeTy)) in
  let mark fix = Il.emit_full e (Some fix) Il.Dead in

  let repeat_jmp_fix = new_fixup "repeat jump" in
  let skip_jmp_fix = new_fixup "skip jump" in
  let exit_jmp_fix = new_fixup "exit jump" in

    push (ro ebp);                      (* save ebp at entry            *)

    mark repeat_jmp_fix;

    mov (rc esi)                        (* esi <- crate ptr             *)
      (c (fp_n ((-1) - n_callee_saves)));
    mov (rc edi)                        (* edi <- frame glue functions. *)
      (c (fp_n ((-2) - n_callee_saves)));
    emit (Il.cmp (ro edi) (immi 0L));

    emit
      (Il.jmp Il.JE
         (codefix skip_jmp_fix));       (* if struct* is nonzero        *)
    add edi esi;                        (* add crate ptr to disp.       *)
    mov
      (rc ecx)
      (c (edi_n glue_field));           (* ecx <-  glue                 *)
    emit (Il.cmp (ro ecx) (immi 0L));

    emit
      (Il.jmp Il.JE
         (codefix skip_jmp_fix));       (* if glue-fn is nonzero        *)
    add ecx esi;                        (* add crate ptr to disp.       *)
    push (ro ebp);                      (* frame-arg                    *)
    push (immi 0L);                     (* null closure-ptr             *)
    push (c task_ptr);                  (* self-task ptr                *)
    push (immi 0L);                     (* outptr                       *)
    emit (Il.call (rc eax)
            (reg_codeptr (h ecx)));     (* call glue_fn, trashing eax.  *)
    pop (rc eax);
    pop (rc eax);
    pop (rc eax);

    mark skip_jmp_fix;
    mov (rc edi) (c (fp_n 0));          (* load next fp (fp[0])           *)
    emit (Il.cmp (ro edi) (immi 0L));
    emit (Il.jmp Il.JE
            (codefix exit_jmp_fix));    (* if nonzero                     *)
    mov (rc ebp) (ro edi);              (* move to next frame             *)
    emit (Il.jmp Il.JMP
            (codefix repeat_jmp_fix));  (* loop                           *)

    (* exit path. *)
    mark exit_jmp_fix;
    pop (rc ebp);                       (* restore ebp                    *)
;;

let sweep_gc_chain
    (e:Il.emitter)
    (glue_field:int)
    (clear_mark:bool)
    : unit =

  let emit = Il.emit e in
  let mov dst src = emit (Il.umov dst src) in
  let push x = emit (Il.Push x) in
  let pop x = emit (Il.Pop x) in
  let band x y = emit (Il.binary Il.AND x (c x) y) in
  let add x y = emit (Il.binary Il.ADD (rc x) (ro x) (ro y)) in
  let edi_n = word_n (Il.Hreg edi) in
  let ecx_n = word_n (Il.Hreg ecx) in
  let codefix fix = Il.CodePtr (Il.ImmPtr (fix, Il.CodeTy)) in
  let mark fix = Il.emit_full e (Some fix) Il.Dead in
  let repeat_jmp_fix = new_fixup "repeat jump" in
  let skip_jmp_fix = new_fixup "skip jump" in
  let exit_jmp_fix = new_fixup "exit jump" in

    mov (rc edi) (c task_ptr);
    mov (rc edi) (c (edi_n Abi.task_field_gc_alloc_chain));
    mark repeat_jmp_fix;
    emit (Il.cmp (ro edi) (immi 0L));
    emit (Il.jmp Il.JE
            (codefix exit_jmp_fix));            (* if nonzero             *)
    mov (rc ecx)                                (* Load GC ctrl word      *)
      (c (edi_n Abi.box_gc_field_ctrl));
    mov (rc eax) (ro ecx);
    band (rc eax) (immi 1L);                    (* Extract mark to eax.   *)
    band                                        (* Clear mark in ecx.     *)
      (rc ecx)
      (immi 0xfffffffffffffffeL);

    if clear_mark
    then
      mov                                       (* Write-back cleared.    *)
        ((edi_n Abi.box_gc_field_ctrl))
        (ro ecx);

    emit (Il.cmp (ro eax) (immi 0L));
    emit
      (Il.jmp Il.JNE
         (codefix skip_jmp_fix));               (* if unmarked (garbage)  *)

    push (ro edi);                              (* Push gc_val.           *)

    (* NB: ecx is a type descriptor now. *)

    mov (rc eax)                                (* Load typarams ptr.     *)
      (c (ecx_n Abi.tydesc_field_first_param));
    push (ro eax);                              (* Push typarams ptr.     *)

    push (immi 0L);                             (* Push null closure-ptr  *)
    push (c task_ptr);                          (* Push task ptr.         *)
    push (immi 0L);                             (* Push null outptr.      *)

    mov (rc eax)                                (* Load glue tydesc-off.  *)
      (c (ecx_n glue_field));
    add eax ecx;                                (* Add to tydesc*         *)
    emit (Il.call (rc eax)
            (reg_codeptr (h eax)));             (* Call glue.             *)
    pop (rc eax);
    pop (rc eax);
    pop (rc eax);
    pop (rc eax);

    mark skip_jmp_fix;
    mov (rc edi)                                (* Advance down chain     *)
      (c (edi_n Abi.box_gc_field_next));
    emit (Il.jmp Il.JMP
            (codefix repeat_jmp_fix));          (* loop                   *)
    mark exit_jmp_fix;
;;



let gc_glue
    (e:Il.emitter)
    : unit =

  let emit = Il.emit e in
  let mov dst src = emit (Il.umov dst src) in
  let push x = emit (Il.Push x) in
  let pop x = emit (Il.Pop x) in
  let edi_n = word_n (Il.Hreg edi) in

    mov (rc edi) (c task_ptr);            (* switch back to rust stack    *)
    mov
      (rc esp)
      (c (edi_n Abi.task_field_rust_sp));

    (* Mark pass. *)

    push (ro ebp);
    save_callee_saves e;
    push (ro eax);
    crawl_stack_calling_glue e Abi.frame_glue_fns_field_mark;

    (* The sweep pass has two sub-passes over the GC chain:
     *
     *    - In pass #1, 'severing', we goes through and disposes of all
     *      mutable box slots in each record. That is, rc-- the referent,
     *      and then null-out.  If the rc-- gets to zero, that just means the
     *      mutable is part of the garbage set currently being collected. But
     *      a mutable may be live-and-outside; this detaches the garbage set
     *      from the non-garbage set within the mutable heap.
     *
     *    - In pass #2, 'freeing', we run the normal free-glue. This winds up
     *      running drop-glue on the zero-reference-reaching immutables only,
     *      since all the mutables were nulled out in pass #1. This is where
     *      you do the unlinking from the double-linked chain and call free(),
     *      also.
     *
     *)
    sweep_gc_chain e Abi.tydesc_field_sever_glue false;
    sweep_gc_chain e Abi.tydesc_field_free_glue true;

    pop (rc eax);
    restore_callee_saves e;
    pop (rc ebp);
    Il.emit e Il.Ret;
;;


let unwind_glue
    (e:Il.emitter)
    (nabi:nabi)
    (exit_task_fixup:fixup)
    : unit =

  let emit = Il.emit e in
  let mov dst src = emit (Il.umov dst src) in
  let edi_n = word_n (Il.Hreg edi) in

    mov (rc edi) (c task_ptr);          (* switch back to rust stack    *)
    mov
      (rc esp)
      (c (edi_n Abi.task_field_rust_sp));

    crawl_stack_calling_glue e Abi.frame_glue_fns_field_drop;
    let callee =
      Abi.load_fixup_codeptr
        e (h eax) exit_task_fixup false nabi.nabi_indirect
    in
      emit_c_call
        e (rc eax) (h edi) (h ecx) nabi false callee [| (c task_ptr) |];
;;


(* Puts result in eax; clobbers ecx, edx in the process. *)
let rec calculate_sz
    (e:Il.emitter)
    (size:size)
    (in_obj:bool)
    : unit =
  let emit = Il.emit e in
  let mov dst src = emit (Il.umov dst src) in
  let push x = emit (Il.Push x) in
  let pop x = emit (Il.Pop x) in
  let neg x = emit (Il.unary Il.NEG (rc x) (ro x)) in
  let bnot x = emit (Il.unary Il.NOT (rc x) (ro x)) in
  let band x y = emit (Il.binary Il.AND (rc x) (ro x) (ro y)) in
  let add x y = emit (Il.binary Il.ADD (rc x) (ro x) (ro y)) in
  let mul x y = emit (Il.binary Il.UMUL (rc x) (ro x) (ro y)) in
  let subi x y = emit (Il.binary Il.SUB (rc x) (ro x) (immi y)) in
  let eax_gets_a_and_ecx_gets_b a b =
    calculate_sz e b in_obj;
    push (ro eax);
    calculate_sz e a in_obj;
    pop (rc ecx);
  in

  let ty_param_n_in_obj_fn i =
    (*
     * Here we are trying to immitate the obj-fn branch of
     * Trans.get_ty_params_of_current_frame while using
     * eax as our only register.
     *)

    (* Bind all the referent types we'll need... *)

    let obj_box_rty = Semant.obj_box_rty word_bits in
    let tydesc_rty = Semant.tydesc_rty word_bits in
    (* Note that we cheat here and pretend only to have i+1 tydescs (because
       we GEP to the i'th while still in this function, so no one outside
       finds out about the lie. *)
    let tydesc_rtys =
      Array.init (i + 1)
        (fun _ ->  (Il.ScalarTy (Il.AddrTy tydesc_rty)))
    in
    let ty_params_rty = Il.StructTy tydesc_rtys in

      (* ... and fetch! *)

      mov (rc eax) (Il.Cell closure_ptr);
      let obj_body = word_n (h eax) Abi.box_rc_field_body in
      let obj_body = Il.cell_cast obj_body obj_box_rty in
      let tydesc_ptr = get_element_ptr obj_body Abi.obj_body_elt_tydesc in

        mov (rc eax) (Il.Cell tydesc_ptr);
        let tydesc = Il.cell_cast (word_at (h eax)) tydesc_rty in
        let ty_params_ptr =
          get_element_ptr tydesc Abi.tydesc_field_first_param
        in

          mov (rc eax) (Il.Cell ty_params_ptr);
          let ty_params = Il.cell_cast (word_at (h eax)) ty_params_rty in
            get_element_ptr ty_params i
  in

    match size with
        SIZE_fixed i ->
          mov (rc eax) (immi i)

      | SIZE_fixup_mem_sz f ->
          mov (rc eax) (imm (Asm.M_SZ f))

      | SIZE_fixup_mem_pos f ->
          mov (rc eax) (imm (Asm.M_POS f))

      | SIZE_param_size i ->
          if in_obj
          then
            mov (rc eax) (Il.Cell (ty_param_n_in_obj_fn i))
          else
            mov (rc eax) (Il.Cell (ty_param_n i));
          mov (rc eax) (Il.Cell (word_n (h eax) Abi.tydesc_field_size))

      | SIZE_param_align i ->
          if in_obj
          then
            mov (rc eax) (Il.Cell (ty_param_n_in_obj_fn i))
          else
            mov (rc eax) (Il.Cell (ty_param_n i));
          mov (rc eax) (Il.Cell (word_n (h eax) Abi.tydesc_field_align))

      | SIZE_rt_neg a ->
          calculate_sz e a in_obj;
          neg eax

      | SIZE_rt_add (a, b) ->
          eax_gets_a_and_ecx_gets_b a b;
          add eax ecx

      | SIZE_rt_mul (a, b) ->
          eax_gets_a_and_ecx_gets_b a b;
          mul eax ecx

      | SIZE_rt_max (a, b) ->
          eax_gets_a_and_ecx_gets_b a b;
          emit (Il.cmp (ro eax) (ro ecx));
          let jmp_pc = e.Il.emit_pc in
            emit (Il.jmp Il.JAE Il.CodeNone);
            mov (rc eax) (ro ecx);
            Il.patch_jump e jmp_pc e.Il.emit_pc;

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
          eax_gets_a_and_ecx_gets_b off align;
          subi ecx 1L;
          add eax ecx;
          bnot ecx;
          band eax ecx;
;;

let rec size_calculation_stack_highwater (size:size) : int =
  match size with
      SIZE_fixed _
    | SIZE_fixup_mem_sz _
    | SIZE_fixup_mem_pos _
    | SIZE_param_size _
    | SIZE_param_align _ -> 0
    | SIZE_rt_neg a  ->
        (size_calculation_stack_highwater a)
    | SIZE_rt_max (a, b) ->
        (size_calculation_stack_highwater a)
        + (size_calculation_stack_highwater b)
    | SIZE_rt_add (a, b)
    | SIZE_rt_mul (a, b)
    | SIZE_rt_align (a, b) ->
        (size_calculation_stack_highwater a)
        + (size_calculation_stack_highwater b)
        + 1
;;

let minimal_call_sz = Int64.add frame_base_sz callee_saves_sz;;
let boundary_sz =
  (Asm.IMM
     (Int64.add                   (* Extra non-frame room:           *)
        minimal_call_sz           (* to safely enter the next frame, *)
        minimal_call_sz))         (* and make a 'grow' upcall there. *)
;;

let stack_growth_check
    (e:Il.emitter)
    (nabi:nabi)
    (grow_task_fixup:fixup)
    (growsz:Il.operand)
    (grow_jmp:Il.label option)
    (restart_pc:Il.label)
    (end_reg:Il.reg)              (* 
                                   * stack limit on entry,
                                   * new stack pointer on exit 
                                   *)
    (tmp_reg:Il.reg)              (* temporary (trashed) *)
    : unit =
  let emit = Il.emit e in
  let mov dst src = emit (Il.umov dst src) in
  let add dst src = emit (Il.binary Il.ADD dst (Il.Cell dst) src) in
  let sub dst src = emit (Il.binary Il.SUB dst (Il.Cell dst) src) in
    mov (r tmp_reg) (ro esp);         (* tmp = esp                 *)
    sub (r tmp_reg) growsz;           (* tmp -= size-request       *)
    emit (Il.cmp (c (r end_reg)) (c (r tmp_reg)));
    (* 
     * Jump *over* 'grow' upcall on non-underflow:
     * if end_reg <= tmp_reg
     *)

    let bypass_grow_upcall_jmp_pc = e.Il.emit_pc in
      emit (Il.jmp Il.JBE Il.CodeNone);

      begin
        match grow_jmp with
            None -> ()
          | Some j -> Il.patch_jump e j e.Il.emit_pc
      end;
      (* Extract growth-amount from tmp_reg. *)
      mov (r end_reg) (ro esp);
      sub (r end_reg) (c (r tmp_reg));
      add (r end_reg) (Il.Imm (boundary_sz, word_ty));
      (* Perform 'grow' upcall, then restart frame-entry. *)
      emit_void_prologue_call e nabi grow_task_fixup [| c (r end_reg) |];
      emit (Il.jmp Il.JMP (Il.CodeLabel restart_pc));
      Il.patch_jump e bypass_grow_upcall_jmp_pc e.Il.emit_pc
;;

let n_glue_args = Int64.of_int Abi.worst_case_glue_call_args;;
let n_glue_words = Int64.mul word_sz n_glue_args;;

let combined_frame_size
    (framesz:size)
    (callsz:size)
    : size =
  (*
   * We double the reserved callsz because we need a 'temporary tail-call
   * region' above the actual call region, in case there's a drop call at the
   * end of assembling the tail-call args and before copying them to callee
   * position.
   *)

  let callsz = add_sz callsz callsz in

  (*
   * Add in *another* word to handle an extra-awkward spill of the
   * callee address that might occur during an indirect tail call.
   *)
  let callsz = add_sz (SIZE_fixed word_sz) callsz in

  (*
   * Add in enough words for a glue-call (these occur underneath esp)
   *)
  let callsz = add_sz (SIZE_fixed n_glue_words) callsz in

    add_sz callsz framesz
;;

let minimal_fn_prologue
    (e:Il.emitter)
    (call_and_frame_sz:Asm.expr64)
    : unit =

  let emit = Il.emit e in
  let mov dst src = emit (Il.umov dst src) in
  let add dst src = emit (Il.binary Il.ADD dst (Il.Cell dst) src) in
  let sub dst src = emit (Il.binary Il.SUB dst (Il.Cell dst) src) in

    (* See diagram and explanation in full_fn_prologue, below.    *)
    establish_frame_base e;
    save_callee_saves e;
    sub (rc esp) (imm call_and_frame_sz);  (* Establish a frame.     *)
    mov (rc edi) (ro esp);                 (* Zero the frame. *)
    mov (rc ecx) (imm call_and_frame_sz);
    emit (Il.unary Il.ZERO (word_at (h edi)) (ro ecx));
    (* Move esp back up over the glue region. *)
    add (rc esp) (immi n_glue_words);
;;

let full_fn_prologue
    (e:Il.emitter)
    (call_and_frame_sz:size)
    (nabi:nabi)
    (grow_task_fixup:fixup)
    (is_obj_fn:bool)
    : unit =

  let esi_n = word_n (h esi) in
  let emit = Il.emit e in
  let mov dst src = emit (Il.umov dst src) in
  let add dst src = emit (Il.binary Il.ADD dst (Il.Cell dst) src) in
  let sub dst src = emit (Il.binary Il.SUB dst (Il.Cell dst) src) in

  (* We may be in a dynamic-sized frame. This makes matters complex,
   * as we can't just perform a simple growth check in terms of a
   * static size. The check is against a dynamic size, and we need to
   * calculate that size.
   *
   * Unlike size-calculations in 'trans', we do not use vregs to
   * calculate the frame size; instead we use a PUSH/POP stack-machine
   * translation that doesn't disturb the registers we're
   * somewhat-carefully *using* during frame setup.
   *
   * This only pushes the problem back a little ways though: we still
   * need to be sure we have enough room to do the PUSH/POP
   * calculation.  We refer to this amount of space as the 'primordial'
   * frame size, which can *thankfully* be calculated exactly from the
   * arithmetic expression we're aiming to calculate. So we make room
   * for the primordial frame, run the calculation of the full dynamic
   * frame size, then make room *again* for this dynamic size.
   *
   * Our caller reserved enough room for us to push our own frame-base,
   * as well as the frame-base that it will cost to do an upcall.
   *)

  (*
   *  After we save callee-saves, We have a stack like this:
   *
   *  | ...           |
   *  | caller frame  |
   *  | + spill       |
   *  | caller arg K  |
   *  | ...           |
   *  | caller arg 0  |
   *  | retpc         | <-- sp we received, top of callee frame
   *  | callee save 1 | <-- ebp after frame-base setup
   *  | ...           |
   *  | callee save N | <-- esp after saving callee-saves
   *  | ...           |
   *  | callee frame  |
   *  | + spill       |
   *  | callee arg J  |
   *  | ...           |
   *  | callee arg 0  | <-- bottom of callee frame
   *  | next retpc    |
   *  | next save 1   |
   *  | ...           |
   *  | next save N   | <-- bottom of region we must reserve
   *  | ...           |
   *
   * A "frame base" is the retpc + ebp.
   *
   * We need to reserve room for our frame *and* the next frame-base and
   * callee-saves, because we're going to be blindly entering the next
   * frame-base (pushing eip and callee-saves) before we perform the next
   * check.
   *)

    (* Already have room to save regs on entry. *)
    establish_frame_base e;
    save_callee_saves e;

    let restart_pc = e.Il.emit_pc in

      mov (rc esi) (c task_ptr);         (* esi = task                *)
      mov
        (rc esi)
        (c (esi_n Abi.task_field_stk));  (* esi = task->stk           *)
      add (rc esi) (imm
                      (Asm.ADD
                         ((word_off_n Abi.stk_field_data),
                          boundary_sz)));

      let (dynamic_frame_sz, dynamic_grow_jmp) =
        match Il.size_to_expr64 call_and_frame_sz with
            None ->
              begin
                let primordial_frame_sz =
                  Asm.IMM
                    (Int64.mul word_sz
                       (Int64.of_int
                          (size_calculation_stack_highwater
                             call_and_frame_sz)))
                in
                  (* Primordial size-check. *)
                  mov (rc edi) (ro esp);  (* edi = esp            *)
                  sub                     (* edi -= size-request  *)
                    (rc edi)
                    (imm primordial_frame_sz);
                  emit (Il.cmp (ro esi) (ro edi));

                  (* Jump to 'grow' upcall on underflow: if esi (bottom) is >
                     edi (proposed-esp) *)

                  let primordial_underflow_jmp_pc = e.Il.emit_pc in
                    emit (Il.jmp Il.JA Il.CodeNone);

                    (* Calculate dynamic frame size. *)
                    calculate_sz e call_and_frame_sz is_obj_fn;
                    ((ro eax), Some primordial_underflow_jmp_pc)
              end
          | Some e -> ((imm e), None)
      in

        (* "Full" frame size-check. *)
        stack_growth_check e nabi grow_task_fixup
          dynamic_frame_sz dynamic_grow_jmp restart_pc (h esi) (h edi);

        (* Establish a frame, wherever we landed. *)
        sub (rc esp) dynamic_frame_sz;

        (* Zero the frame.
         * 
         * FIXME (ssue 27): this is awful, will go away when we have proper
         * CFI.
         *)

        mov (rc edi) (ro esp);
        mov (rc ecx) dynamic_frame_sz;
        emit (Il.unary Il.ZERO (word_at (h edi)) (ro ecx));

        (* Move esp back up over the glue region. *)
        add (rc esp) (immi n_glue_words);
;;

let fn_prologue
    (e:Il.emitter)
    (framesz:size)
    (callsz:size)
    (nabi:nabi)
    (grow_task_fixup:fixup)
    (is_obj_fn:bool)
    (minimal:bool)
    : unit =

  let call_and_frame_sz = combined_frame_size framesz callsz in

  let full _ =
    full_fn_prologue e call_and_frame_sz nabi grow_task_fixup is_obj_fn
  in

    if minimal
    then
      match Il.size_to_expr64 call_and_frame_sz with
          None -> full()
        | Some sz -> minimal_fn_prologue e sz
    else
      full()
;;

let fn_epilogue (e:Il.emitter) : unit =
  (* Tear down existing frame. *)
  let emit = Il.emit e in
  let mov dst src = emit (Il.umov dst src) in
  let sub dst src = emit (Il.binary Il.SUB dst (Il.Cell dst) src) in
    sub (rc ebp) (immi callee_saves_sz);
    mov (rc esp) (ro ebp);
    restore_callee_saves e;
    leave_frame e;
    emit Il.Ret;
;;

let inline_memcpy
    (e:Il.emitter)
    (n_bytes:int64)
    (dst_ptr:Il.reg)
    (src_ptr:Il.reg)
    (tmp_reg:Il.reg)
    (ascending:bool)
    : unit =
  let emit = Il.emit e in
  let mov dst src = emit (Il.umov dst src) in
  let bpw = Int64.to_int word_sz in
  let w = Int64.to_int (Int64.div n_bytes word_sz) in
  let b = Int64.to_int (Int64.rem n_bytes word_sz) in
    if ascending
    then
      begin
        for i = 0 to (w-1) do
          mov (r tmp_reg) (c (word_n src_ptr i));
          mov (word_n dst_ptr i) (c (r tmp_reg));
        done;
        for i = 0 to (b-1) do
          let off = (w*bpw) + i in
            mov (r tmp_reg) (c (byte_n src_ptr off));
            mov (byte_n dst_ptr off) (c (r tmp_reg));
        done;
      end
    else
      begin
        for i = (b-1) downto 0 do
          let off = (w*bpw) + i in
            mov (r tmp_reg) (c (byte_n src_ptr off));
            mov (byte_n dst_ptr off) (c (r tmp_reg));
        done;
        for i = (w-1) downto 0 do
          mov (r tmp_reg) (c (word_n src_ptr i));
          mov (word_n dst_ptr i) (c (r tmp_reg));
        done;
      end
;;



let fn_tail_call
    (e:Il.emitter)
    (caller_callsz:int64)
    (caller_argsz:int64)
    (callee_code:Il.code)
    (callee_argsz:int64)
    : unit =
  let emit = Il.emit e in
  let binary op dst imm = emit (Il.binary op dst (c dst) (immi imm)) in
  let mov dst src = emit (Il.umov dst src) in
  let argsz_diff = Int64.sub caller_argsz callee_argsz in
  let callee_spill_cell = word_at_off (h esp) (Asm.IMM caller_callsz) in

    (*
     * Our outgoing arguments were prepared in a region above the call region;
     * this is reserved for the purpose of making tail-calls *only*, so we do
     * not collide with glue calls we had to make while dropping the frame,
     * after assembling our arg region.
     *
     * Thus, esp points to the "normal" arg region, and we need to move it
     * to point to the tail-call arg region. To make matters simple, both
     * regions are the same size, one atop the other.
     *)

    annotate e "tail call: move esp to temporary tail call arg-prep area";
    binary Il.ADD (rc esp) caller_callsz;

    (*
     * If we're given a non-ImmPtr callee, we may need to move it to a known
     * cell to avoid clobbering its register while we do the argument shuffle
     * below.
     *
     * Sadly, we are too register-starved to just flush our callee to a reg;
     * so we carve out an extra word of the temporary call-region and use
     * it.
     *
     * This is ridiculous, but works.
     *)
    begin
      match callee_code with
          Il.CodePtr (Il.Cell c) ->
              annotate e "tail call: spill callee-ptr to temporary memory";
              mov callee_spill_cell (Il.Cell c);

        | _ -> ()
    end;

    (* edx <- ebp; restore ebp, edi, esi, ebx; ecx <- retpc *)
    annotate e "tail call: restore registers from frame base";
    restore_frame_regs e (h edx) (h ecx);
    (* move edx past frame base and adjust for difference in call sizes *)
    annotate e "tail call: adjust temporary fp";
    binary Il.ADD (rc edx) (Int64.add frame_base_sz argsz_diff);

    (*
     * stack grows downwards; copy from high to low
     *
     *   bpw = word_sz
     *   w = floor(callee_argsz / word_sz)
     *   b = callee_argsz % word_sz
     *
     * byte copies:
     *   +------------------------+
     *   |                        |
     *   +------------------------+ <-- base + (w * word_sz) + (b - 1)
     *   .                        .
     *   +------------------------+
     *   |                        |
     *   +------------------------+ <-- base + (w * word_sz) + (b - b)
     * word copies:                     =
     *   +------------------------+ <-- base + ((w-0) * word_sz)
     *   | bytes                  |
     *   | (w-1)*bpw..w*bpw-1     |
     *   +------------------------+ <-- base + ((w-1) * word_sz)
     *   | bytes                  |
     *   | (w-2)*bpw..(w-1)*bpw-1 |
     *   +------------------------+ <-- base + ((w-2) * word_sz)
     *   .                        .
     *   .                        .
     *   .                        .
     *   +------------------------+
     *   | bytes                  |
     *   | 0..bpw - 1             |
     *   +------------------------+ <-- base + ((w-w) * word_sz)
     *)

    annotate e "tail call: move arg-tuple up to top of frame";
    (* NOTE: must copy top-to-bottom in case the regions overlap *)
    inline_memcpy e callee_argsz (h edx) (h esp) (h eax) false;

    (*
     * We're done with eax now; so in the case where we had to spill
     * our callee codeptr, we can reload it into eax here and rewrite
     * our callee into *eax.
     *)
    let callee_code =
      match callee_code with
          Il.CodePtr (Il.Cell _) ->
              annotate e "tail call: reload callee-ptr from temporary memory";
              mov (rc eax) (Il.Cell callee_spill_cell);
              reg_codeptr (h eax)

        | _ -> callee_code
    in


    (* esp <- edx *)
    annotate e "tail call: adjust stack pointer";
    mov (rc esp) (ro edx);
    (* PUSH ecx (retpc) *)
    annotate e "tail call: push retpc";
    emit (Il.Push (ro ecx));
    (* JMP callee_code *)
    emit (Il.jmp Il.JMP callee_code);
;;


let activate_glue (e:Il.emitter) : unit =
  (*
   * This is a bit of glue-code. It should be emitted once per
   * compilation unit.
   *
   *   - save regs on C stack
   *   - align sp on a 16-byte boundary
   *   - save sp to task.runtime_sp (runtime_sp is thus always aligned)
   *   - load saved task sp (switch stack)
   *   - restore saved task regs
   *   - return to saved task pc
   *
   * Our incoming stack looks like this:
   *
   *   *esp+4        = [arg1   ] = task ptr
   *   *esp          = [retpc  ]
   *)

  let sp_n = word_n (Il.Hreg esp) in
  let edx_n = word_n (Il.Hreg edx) in
  let emit = Il.emit e in
  let mov dst src = emit (Il.umov dst src) in
  let binary op dst imm = emit (Il.binary op dst (c dst) (immi imm)) in

    mov (rc edx) (c (sp_n 1));            (* edx <- task             *)
    establish_frame_base e;
    save_callee_saves e;
    mov
      (edx_n Abi.task_field_runtime_sp)
      (ro esp);                           (* task->runtime_sp <- esp *)
    mov
      (rc esp)
      (c (edx_n Abi.task_field_rust_sp)); (* esp <- task->rust_sp    *)

    (*
     * There are two paths we can arrive at this code from:
     *
     *
     *   1. We are activating a task for the first time. When we switch into
     *      the task stack and 'ret' to its first instruction, we'll start
     *      doing whatever the first instruction says. Probably saving
     *      registers and starting to establish a frame. Harmless stuff,
     *      doesn't look at task->rust_sp again except when it clobbers it
     *      during a later upcall.
     *
     *
     *   2. We are resuming a task that was descheduled by the yield glue
     *      below.  When we switch into the task stack and 'ret', we'll be
     *      ret'ing to a very particular instruction:
     *
     *              "esp <- task->rust_sp"
     *
     *      this is the first instruction we 'ret' to after this glue, because
     *      it is the first instruction following *any* upcall, and the task
     *      we are activating was descheduled mid-upcall.
     *
     *      Unfortunately for us, we have already restored esp from
     *      task->rust_sp and are about to eat the 5 words off the top of it.
     *
     *
     *      | ...    | <-- where esp will be once we restore + ret, below,
     *      | retpc  |     and where we'd *like* task->rust_sp to wind up.
     *      | ebp    |
     *      | edi    |
     *      | esi    |
     *      | ebx    | <-- current task->rust_sp == current esp
     *
     * 
     *      This is a problem. If we return to "esp <- task->rust_sp" it will
     *      push esp back down by 5 words. This manifests as a rust stack that
     *      grows by 5 words on each yield/reactivate. Not good.
     * 
     *      So what we do here is just adjust task->rust_sp up 5 words as
     *      well, to mirror the movement in esp we're about to perform. That
     *      way the "esp <- task->rust_sp" we 'ret' to below will be a
     *      no-op. Esp won't move, and the task's stack won't grow.
     *)

    binary Il.ADD (edx_n Abi.task_field_rust_sp)
      (Int64.mul (Int64.of_int (n_callee_saves + 2)) word_sz);

    (**** IN TASK STACK ****)
    restore_callee_saves e;
    leave_frame e;
    emit Il.Ret;
    (***********************)
  ()
;;

let yield_glue (e:Il.emitter) : unit =

  (* More glue code, this time the 'bottom half' of yielding.
   *
   * We arrived here because an upcall decided to deschedule the
   * running task. So the upcall's return address got patched to the
   * first instruction of this glue code.
   *
   * When the upcall does 'ret' it will come here, and its esp will be
   * pointing to the last argument pushed on the C stack before making
   * the upcall: the 0th argument to the upcall, which is always the
   * task ptr performing the upcall. That's where we take over.
   *
   * Our goal is to complete the descheduling
   *
   *   - Switch over to the task stack temporarily.
   *
   *   - Save the task's callee-saves onto the task stack.
   *     (the task is now 'descheduled', safe to set aside)
   *
   *   - Switch *back* to the C stack.
   *
   *   - Restore the C-stack callee-saves.
   *
   *   - Return to the caller on the C stack that activated the task.
   *
   *)
  let esp_n = word_n (Il.Hreg esp) in
  let edx_n = word_n (Il.Hreg edx) in
  let emit = Il.emit e in
  let mov dst src = emit (Il.umov dst src) in

    mov
      (rc edx) (c (esp_n 0));                (* edx <- arg0 (task)      *)
    mov
      (rc esp)
      (c (edx_n Abi.task_field_rust_sp));    (* esp <- task->rust_sp    *)
    establish_frame_base e;
    save_callee_saves e;
    mov                                      (* task->rust_sp <- esp    *)
      (edx_n Abi.task_field_rust_sp)
      (ro esp);
    mov
      (rc esp)
      (c (edx_n Abi.task_field_runtime_sp)); (* esp <- task->runtime_sp *)

    (**** IN C STACK ****)
    restore_callee_saves e;
    leave_frame e;
    emit Il.Ret;
    (***********************)
  ()
;;


let push_pos32 (e:Il.emitter) (fix:fixup) : unit =
  let (reg, _, _) = get_next_pc_thunk in
    Abi.load_fixup_addr e reg fix Il.OpaqueTy;
    Il.emit e (Il.Push (Il.Cell (Il.Reg (reg, Il.AddrTy Il.OpaqueTy))))
;;

let objfile_start
    (e:Il.emitter)
    ~(start_fixup:fixup)
    ~(rust_start_fixup:fixup)
    ~(main_fn_fixup:fixup)
    ~(crate_fixup:fixup)
    ~(indirect_start:bool)
    : unit =
  let ebp_n = word_n (Il.Hreg ebp) in
  let emit = Il.emit e in
  let mov dst src = emit (Il.umov dst src) in
  let push_pos32 = push_pos32 e in
    Il.emit_full e (Some start_fixup) Il.Dead;
    establish_frame_base e;
    save_callee_saves e;

    (* If we're very lucky, the platform will have left us with
     * something sensible in the startup stack like so:
     * 
     *   *ebp+12       = [arg1   ] = argv
     *   *ebp+8        = [arg0   ] = argc
     *   *ebp+4        = [retpc  ]
     *   *ebp          = [old_ebp]
     *   *ebp-4        = [old_edi]
     *   *ebp-8        = [old_esi]
     *   *ebp-12       = [old_ebx]
     * 
     * This is not the case everywhere, but we start with this
     * assumption and correct it in the runtime library.
     *)

    (* Copy argv. *)
    mov (rc eax) (c (ebp_n 3));
    Il.emit e (Il.Push (ro eax));

    (* Copy argc. *)
    mov (rc eax) (c (ebp_n 2));
    Il.emit e (Il.Push (ro eax));

    push_pos32 crate_fixup;
    push_pos32 main_fn_fixup;
    let fptr =
      Abi.load_fixup_codeptr e (h eax) rust_start_fixup true indirect_start
    in
      Il.emit e (Il.call (rc eax) fptr);
      Il.emit e (Il.Pop (rc ecx));
      Il.emit e (Il.Pop (rc ecx));
      Il.emit e (Il.Pop (rc ecx));
      Il.emit e (Il.Pop (rc ecx));
      restore_callee_saves e;
      leave_frame e;
      Il.emit e Il.Ret;
;;

let (abi:Abi.abi) =
  {
    Abi.abi_word_sz = word_sz;
    Abi.abi_word_bits = word_bits;
    Abi.abi_word_ty = word_ty;

    Abi.abi_tag = Abi.abi_x86_rustboot_cdecl;

    Abi.abi_has_pcrel_data = false;
    Abi.abi_has_pcrel_code = true;

    Abi.abi_n_hardregs = n_hardregs;
    Abi.abi_str_of_hardreg = reg_str;
    Abi.abi_emit_target_specific = emit_target_specific;
    Abi.abi_constrain_vregs = constrain_vregs;

    Abi.abi_emit_fn_prologue = fn_prologue;
    Abi.abi_emit_fn_epilogue = fn_epilogue;
    Abi.abi_emit_fn_tail_call = fn_tail_call;
    Abi.abi_clobbers = clobbers;

    Abi.abi_emit_native_call = emit_native_call;
    Abi.abi_emit_native_void_call = emit_native_void_call;
    Abi.abi_emit_native_call_in_thunk = emit_native_call_in_thunk;
    Abi.abi_emit_inline_memcpy = inline_memcpy;

    Abi.abi_activate = activate_glue;
    Abi.abi_yield = yield_glue;
    Abi.abi_unwind = unwind_glue;
    Abi.abi_gc = gc_glue;
    Abi.abi_get_next_pc_thunk = Some get_next_pc_thunk;

    Abi.abi_sp_reg = (Il.Hreg esp);
    Abi.abi_fp_reg = (Il.Hreg ebp);
    Abi.abi_dwarf_fp_reg = dwarf_ebp;
    Abi.abi_tp_cell = task_ptr;
    Abi.abi_frame_base_sz = frame_base_sz;
    Abi.abi_callee_saves_sz = callee_saves_sz;
    Abi.abi_frame_info_sz = frame_info_sz;
    Abi.abi_implicit_args_sz = implicit_args_sz;
    Abi.abi_spill_slot = spill_slot;
  }


(*
 * NB: factor the instruction selector often. There's lots of
 * semi-redundancy in the ISA.
 *)


let imm_is_signed_byte (n:int64) : bool =
  (i64_le (-128L) n) && (i64_le n 127L)
;;

let imm_is_unsigned_byte (n:int64) : bool =
  (i64_le (0L) n) && (i64_le n 255L)
;;


let rm_r (c:Il.cell) (r:int) : Asm.frag =
  let reg_ebp = 6 in
  let reg_esp = 7 in

  (*
   * We do a little contortion here to accommodate the special case of
   * being asked to form esp-relative addresses; these require SIB
   * bytes on x86. Of course!
   *)
  let sib_esp_base = Asm.BYTE 0x24 in
  let seq1 rm modrm =
    if rm = reg_esp
    then Asm.SEQ [| modrm; sib_esp_base |]
    else modrm
  in
  let seq2 rm modrm disp =
    if rm = reg_esp
    then Asm.SEQ [| modrm; sib_esp_base; disp |]
    else Asm.SEQ [| modrm; disp |]
  in

    match c with
        Il.Reg ((Il.Hreg rm), _) ->
          Asm.BYTE (modrm_reg (reg rm) r)
      | Il.Mem (a, _) ->
          begin
            match a with
                Il.Abs disp ->
                  Asm.SEQ [| Asm.BYTE (modrm_deref_disp32 r);
                             Asm.WORD (TY_i32, disp) |]

              | Il.RegIn ((Il.Hreg rm), None) when rm != reg_ebp ->
                  seq1 rm (Asm.BYTE (modrm_deref_reg (reg rm) r))

              | Il.RegIn ((Il.Hreg rm), Some (Asm.IMM 0L))
                  when rm != reg_ebp ->
                  seq1 rm (Asm.BYTE (modrm_deref_reg (reg rm) r))

              (* The next two are just to save the relaxation system some
               * churn.
               *)

              | Il.RegIn ((Il.Hreg rm), Some (Asm.IMM n))
                  when imm_is_signed_byte n ->
                  seq2 rm
                    (Asm.BYTE (modrm_deref_reg_plus_disp8 (reg rm) r))
                    (Asm.WORD (TY_i8, Asm.IMM n))

              | Il.RegIn ((Il.Hreg rm), Some (Asm.IMM n)) ->
                  seq2 rm
                    (Asm.BYTE (modrm_deref_reg_plus_disp32 (reg rm) r))
                    (Asm.WORD (TY_i32, Asm.IMM n))

              | Il.RegIn ((Il.Hreg rm), Some disp) ->
                  Asm.new_relaxation
                    [|
                      seq2 rm
                        (Asm.BYTE (modrm_deref_reg_plus_disp32 (reg rm) r))
                        (Asm.WORD (TY_i32, disp));
                      seq2 rm
                        (Asm.BYTE (modrm_deref_reg_plus_disp8 (reg rm) r))
                        (Asm.WORD (TY_i8, disp))
                    |]
              | _ -> raise Unrecognized
          end
      | _ -> raise Unrecognized
;;


let insn_rm_r (op:int) (c:Il.cell) (r:int) : Asm.frag =
  Asm.SEQ [| Asm.BYTE op; rm_r c r |]
;;


let insn_rm_r_imm
    (op:int)
    (c:Il.cell)
    (r:int)
    (ty:ty_mach)
    (i:Asm.expr64)
    : Asm.frag =
  Asm.SEQ [| Asm.BYTE op; rm_r c r; Asm.WORD (ty, i) |]
;;

let insn_rm_r_imm_s8_s32
    (op8:int)
    (op32:int)
    (c:Il.cell)
    (r:int)
    (i:Asm.expr64)
    : Asm.frag =
  match i with
      Asm.IMM n when imm_is_signed_byte n ->
        insn_rm_r_imm op8 c r TY_i8 i
    | _ ->
        Asm.new_relaxation
          [|
            insn_rm_r_imm op32 c r TY_i32 i;
            insn_rm_r_imm op8 c r TY_i8 i
          |]
;;

let insn_rm_r_imm_u8_u32
    (op8:int)
    (op32:int)
    (c:Il.cell)
    (r:int)
    (i:Asm.expr64)
    : Asm.frag =
  match i with
      Asm.IMM n when imm_is_unsigned_byte n ->
        insn_rm_r_imm op8 c r TY_u8 i
    | _ ->
        Asm.new_relaxation
          [|
            insn_rm_r_imm op32 c r TY_u32 i;
            insn_rm_r_imm op8 c r TY_u8 i
          |]
;;


let insn_pcrel_relax
    (op8_frag:Asm.frag)
    (op32_frag:Asm.frag)
    (fix:fixup)
    : Asm.frag =
  let pcrel_mark_fixup = new_fixup "pcrel mark fixup" in
  let def = Asm.DEF (pcrel_mark_fixup, Asm.MARK) in
  let pcrel_expr = (Asm.SUB (Asm.M_POS fix,
                             Asm.M_POS pcrel_mark_fixup))
  in
    Asm.new_relaxation
      [|
        Asm.SEQ [| op32_frag; Asm.WORD (TY_i32, pcrel_expr); def |];
        Asm.SEQ [| op8_frag; Asm.WORD (TY_i8, pcrel_expr); def |];
      |]
;;

let insn_pcrel_simple (op32:int) (fix:fixup) : Asm.frag =
  let pcrel_mark_fixup = new_fixup "pcrel mark fixup" in
  let def = Asm.DEF (pcrel_mark_fixup, Asm.MARK) in
  let pcrel_expr = (Asm.SUB (Asm.M_POS fix,
                             Asm.M_POS pcrel_mark_fixup))
  in
    Asm.SEQ [| Asm.BYTE op32; Asm.WORD (TY_i32, pcrel_expr); def |]
;;

let insn_pcrel (op8:int) (op32:int) (fix:fixup) : Asm.frag =
  insn_pcrel_relax (Asm.BYTE op8) (Asm.BYTE op32) fix
;;

let insn_pcrel_prefix32
    (op8:int)
    (prefix32:int)
    (op32:int)
    (fix:fixup)
    : Asm.frag =
  insn_pcrel_relax (Asm.BYTE op8) (Asm.BYTES [| prefix32; op32 |]) fix
;;

(* FIXME: tighten imm-based dispatch by imm type. *)
let cmp (a:Il.operand) (b:Il.operand) : Asm.frag =
  match (a,b) with
      (Il.Cell c, Il.Imm (i, TY_i8)) when is_rm8 c ->
        insn_rm_r_imm 0x80 c slash7 TY_i8 i
    | (Il.Cell c, Il.Imm (i, TY_u8)) when is_rm8 c ->
        insn_rm_r_imm 0x80 c slash7 TY_u8 i
    | (Il.Cell c, Il.Imm (i, _)) when is_rm32 c ->
        (*
         * NB: We can't switch on signed-ness here, as 'cmp' is
         * defined to sign-extend its operand; i.e. we have to treat
         * it as though you're emitting a signed byte (in the sense of
         * immediate-size selection) even if the incoming value is
         * unsigned.
         *)
        insn_rm_r_imm_s8_s32 0x83 0x81 c slash7 i
    | (Il.Cell c, Il.Cell (Il.Reg (Il.Hreg r, _))) ->
        insn_rm_r 0x39 c (reg r)
    | (Il.Cell (Il.Reg (Il.Hreg r, _)), Il.Cell c) ->
        insn_rm_r 0x3b c (reg r)
    | _ -> raise Unrecognized
;;

let zero (dst:Il.cell) (count:Il.operand) : Asm.frag =
  match (dst, count) with

      ((Il.Mem (Il.RegIn ((Il.Hreg dst_ptr), None), _)),
       Il.Cell (Il.Reg ((Il.Hreg count), _)))
        when dst_ptr = edi && count = ecx ->
          Asm.BYTES [|
            0xb0; 0x0;  (* mov %eax, 0 : move a zero into al. *)
            0xf3; 0xaa; (* rep stos m8 : fill ecx bytes at [edi] with al *)
          |]

    | _ -> raise Unrecognized
;;

let mov (signed:bool) (dst:Il.cell) (src:Il.operand) : Asm.frag =
  if is_ty8 (Il.cell_scalar_ty dst)
  then
    begin
      match dst with
          Il.Reg (Il.Hreg r, _) -> assert (is_ok_r8 r)
        | _ -> ()
    end;

  if is_ty8 (Il.operand_scalar_ty src)
  then
    begin
      match src with
          Il.Cell (Il.Reg (Il.Hreg r, _)) -> assert (is_ok_r8 r)
        | _ -> ()
    end;

  match (signed, dst, src) with

      (* m8 <- r??, r8 or truncate(r32). *)
      (_,  _, Il.Cell (Il.Reg ((Il.Hreg r), _)))
        when is_m8 dst ->
          insn_rm_r 0x88 dst (reg r)

    (* r8 <- r8: treat as r32 <- r32. *)
    | (_,  Il.Reg ((Il.Hreg r), _), Il.Cell src_cell)
        when is_r8 dst && is_r8 src_cell ->
        insn_rm_r 0x8b src_cell (reg r)

    (* rm32 <- r32 *)
    | (_,  _, Il.Cell (Il.Reg ((Il.Hreg r), src_ty)))
        when (is_r8 dst || is_rm32 dst) && is_ty32 src_ty ->
        insn_rm_r 0x89 dst (reg r)

    (* r32 <- rm32 *)
    | (_,  (Il.Reg ((Il.Hreg r), dst_ty)), Il.Cell src_cell)
        when is_ty32 dst_ty && is_rm32 src_cell ->
          insn_rm_r 0x8b src_cell (reg r)

    (* MOVZX: r8/r32 <- zx(rm8) *)
    | (false, Il.Reg ((Il.Hreg r, _)), Il.Cell src_cell)
        when (is_r8 dst || is_r32 dst) && is_rm8 src_cell ->
        Asm.SEQ [| Asm.BYTE 0x0f;
                   insn_rm_r 0xb6 src_cell (reg r) |]

    (* MOVZX: m32 <- zx(r8) *)
    | (false, _, (Il.Cell (Il.Reg ((Il.Hreg r), _) as src_cell)))
        when (is_m32 dst) && is_r8 src_cell ->
        (* Fake with 2 insns:
         *
         * movzx r32 <- r8;   (in-place zero-extension)
         * mov m32 <- r32;    (NB: must happen in AL/CL/DL/BL)
         *)
        Asm.SEQ [| Asm.BYTE 0x0f;
                   insn_rm_r 0xb6 src_cell (reg r);
                   insn_rm_r 0x89 dst (reg r);
                |]

    (* MOVSX: r8/r32 <- sx(rm8) *)
    | (true, Il.Reg ((Il.Hreg r), _), Il.Cell src_cell)
        when (is_r8 dst || is_r32 dst) && is_rm8 src_cell ->
        Asm.SEQ [| Asm.BYTE 0x0f;
                   insn_rm_r 0xbe src_cell (reg r) |]

    (* MOVSX: m32 <- sx(r8) *)
    | (true, _, (Il.Cell (Il.Reg ((Il.Hreg r), _) as src_cell)))
        when (is_m32 dst) && is_r8 src_cell ->
        (* Fake with 2 insns:
         *
         * movsx r32 <- r8;   (in-place sign-extension)
         * mov m32 <- r32;    (NB: must happen in AL/CL/DL/BL)
         *)
        Asm.SEQ [| Asm.BYTE 0x0f;
                   insn_rm_r 0xbe src_cell (reg r);
                   insn_rm_r 0x89 dst (reg r);
                |]

    (* m8 <- imm8 (signed) *)
    | (_, _, Il.Imm ((Asm.IMM n), _))
        when is_m8 dst && imm_is_signed_byte n && signed ->
          insn_rm_r_imm 0xc6 dst slash0 TY_i8 (Asm.IMM n)

    (* m8 <- imm8 (unsigned) *)
    | (_, _, Il.Imm ((Asm.IMM n), _))
        when is_m8 dst && imm_is_unsigned_byte n && (not signed) ->
          insn_rm_r_imm 0xc6 dst slash0 TY_u8 (Asm.IMM n)

    (* rm32 <- imm32 *)
    | (_, _, Il.Imm (i, _)) when is_rm32 dst || is_r8 dst ->
        let t = if signed then TY_i32 else TY_u32 in
          insn_rm_r_imm 0xc7 dst slash0 t i

    | _ -> raise Unrecognized
;;


let lea (dst:Il.cell) (src:Il.operand) : Asm.frag =
  match (dst, src) with
      (Il.Reg ((Il.Hreg r), dst_ty),
       Il.Cell (Il.Mem (mem, _)))
        when is_ty32 dst_ty ->
          insn_rm_r 0x8d (Il.Mem (mem, Il.OpaqueTy)) (reg r)

    | (Il.Reg ((Il.Hreg r), dst_ty),
       Il.ImmPtr (fix, _))
        when is_ty32 dst_ty && r = eax ->
        let anchor = new_fixup "anchor" in
        let fix_off = Asm.SUB ((Asm.M_POS fix),
                               (Asm.M_POS anchor))
        in
          (* NB: These instructions must come as a
           * cluster, w/o any separation.
           *)
          Asm.SEQ [|
            insn_pcrel_simple 0xe8 get_next_pc_thunk_fixup;
            Asm.DEF (anchor, insn_rm_r_imm 0x81 dst slash0 TY_i32 fix_off);
          |]

    | _ -> raise Unrecognized
;;


let select_insn_misc (q:Il.quad') : Asm.frag =

  match q with
      Il.Call c ->
        begin
          match c.Il.call_dst with
              Il.Reg ((Il.Hreg dst), _) when dst = eax ->
                begin
                  match c.Il.call_targ with

                      Il.CodePtr (Il.Cell c)
                        when Il.cell_referent_ty c
                          = Il.ScalarTy (Il.AddrTy Il.CodeTy) ->
                        insn_rm_r 0xff c slash2

                    | Il.CodePtr (Il.ImmPtr (f, Il.CodeTy)) ->
                        insn_pcrel_simple 0xe8 f

                    | _ -> raise Unrecognized
                end
            | _ -> raise Unrecognized
        end

    | Il.Push (Il.Cell (Il.Reg ((Il.Hreg r), t))) when is_ty32 t ->
        Asm.BYTE (0x50 + (reg r))

    | Il.Push (Il.Cell c) when is_rm32 c ->
        insn_rm_r 0xff c slash6

    | Il.Push (Il.Imm (Asm.IMM i, _)) when imm_is_unsigned_byte i ->
        Asm.SEQ [| Asm.BYTE 0x6a; Asm.WORD (TY_u8, (Asm.IMM i)) |]

    | Il.Push (Il.Imm (i, _)) ->
        Asm.SEQ [| Asm.BYTE 0x68; Asm.WORD (TY_u32, i) |]

    | Il.Pop (Il.Reg ((Il.Hreg r), t)) when is_ty32 t ->
        Asm.BYTE (0x58 + (reg r))

    | Il.Pop c when is_rm32 c ->
        insn_rm_r 0x8f c slash0

    | Il.Ret -> Asm.BYTE 0xc3

    | Il.Jmp j ->
        begin
          match (j.Il.jmp_op, j.Il.jmp_targ) with

              (Il.JMP, Il.CodePtr (Il.ImmPtr (f, Il.CodeTy))) ->
                insn_pcrel 0xeb 0xe9 f

            | (Il.JMP, Il.CodePtr (Il.Cell c))
                when Il.cell_referent_ty c
                  = Il.ScalarTy (Il.AddrTy Il.CodeTy) ->
                insn_rm_r 0xff c slash4

            (* FIXME: refactor this to handle cell-based jumps
             * if we ever need them. So far not. *)
            | (_, Il.CodePtr (Il.ImmPtr (f, Il.CodeTy))) ->
                let (op8, op32) =
                  match j.Il.jmp_op with
                    | Il.JC  -> (0x72, 0x82)
                    | Il.JNC -> (0x73, 0x83)
                    | Il.JZ  -> (0x74, 0x84)
                    | Il.JNZ -> (0x75, 0x85)
                    | Il.JO  -> (0x70, 0x80)
                    | Il.JNO -> (0x71, 0x81)
                    | Il.JE  -> (0x74, 0x84)
                    | Il.JNE -> (0x75, 0x85)

                    | Il.JL  -> (0x7c, 0x8c)
                    | Il.JLE -> (0x7e, 0x8e)
                    | Il.JG  -> (0x7f, 0x8f)
                    | Il.JGE -> (0x7d, 0x8d)

                    | Il.JB  -> (0x72, 0x82)
                    | Il.JBE -> (0x76, 0x86)
                    | Il.JA  -> (0x77, 0x87)
                    | Il.JAE -> (0x73, 0x83)
                    | _ -> raise Unrecognized
                in
                let prefix32 = 0x0f in
                  insn_pcrel_prefix32 op8 prefix32 op32 f

            | _ -> raise Unrecognized
        end

    | Il.Dead -> Asm.MARK
    | Il.Debug -> Asm.BYTES [| 0xcc |] (* int 3 *)
    | Il.Regfence -> Asm.MARK
    | Il.End -> Asm.BYTES [| 0x90 |]
    | Il.Nop -> Asm.BYTES [| 0x90 |]
    | _ -> raise Unrecognized
;;


type alu_binop_codes =
     {
       insn: string;
       immslash: int;    (* mod/rm "slash" code for imm-src variant *)
       rm_dst_op8: int;  (* opcode for 8-bit r/m dst variant *)
       rm_dst_op32: int; (* opcode for 32-bit r/m dst variant *)
       rm_src_op8: int;  (* opcode for 8-bit r/m src variant *)
       rm_src_op32: int; (* opcode for 32-bit r/m src variant *)
     }
;;

let alu_binop
    (dst:Il.cell) (src:Il.operand) (codes:alu_binop_codes)
    : Asm.frag =
  match (dst, src) with
      (Il.Reg ((Il.Hreg r), dst_ty), Il.Cell c)
        when (is_ty32 dst_ty && is_rm32 c) || (is_ty8 dst_ty && is_rm8 c)
          -> insn_rm_r codes.rm_src_op32 c (reg r)

    | (_, Il.Cell (Il.Reg ((Il.Hreg r), src_ty)))
        when (is_rm32 dst && is_ty32 src_ty) || (is_rm8 dst && is_ty8 src_ty)
          -> insn_rm_r codes.rm_dst_op32 dst (reg r)

    | (_, Il.Imm (i, _)) when is_rm32 dst || is_rm8 dst
        -> insn_rm_r_imm_s8_s32 0x83 0x81 dst codes.immslash i

    | _ -> raise Unrecognized
;;


let mul_like (src:Il.operand) (signed:bool) (slash:int)
    : Asm.frag =
  match src with
      Il.Cell src when is_rm32 src ->
        insn_rm_r 0xf7 src slash

    | Il.Cell src when is_rm8 src ->
        insn_rm_r 0xf6 src slash

    | Il.Imm (_, TY_u32)
    | Il.Imm (_, TY_i32) ->
        let tmp = Il.Reg ((Il.Hreg edx), Il.ValTy Il.Bits32) in
        Asm.SEQ [| mov signed tmp src;
                   insn_rm_r 0xf7 tmp slash |]

    | Il.Imm (_, TY_u8)
    | Il.Imm (_, TY_i8) ->
        let tmp = Il.Reg ((Il.Hreg edx), Il.ValTy Il.Bits8) in
        Asm.SEQ [| mov signed tmp src;
                   insn_rm_r 0xf6 tmp slash |]

    | _ -> raise Unrecognized
;;


let select_insn (q:Il.quad) : Asm.frag =
  match q.Il.quad_body with
      Il.Unary u ->
        let unop s =
          if u.Il.unary_src = Il.Cell u.Il.unary_dst
          then insn_rm_r 0xf7 u.Il.unary_dst s
          else raise Unrecognized
        in
          begin
            match u.Il.unary_op with
                Il.UMOV -> mov false u.Il.unary_dst u.Il.unary_src
              | Il.IMOV -> mov true u.Il.unary_dst u.Il.unary_src
              | Il.NEG -> unop slash3
              | Il.NOT -> unop slash2
              | Il.ZERO -> zero u.Il.unary_dst u.Il.unary_src
          end

    | Il.Lea le -> lea le.Il.lea_dst le.Il.lea_src

    | Il.Cmp c -> cmp c.Il.cmp_lhs c.Il.cmp_rhs

    | Il.Binary b ->
        begin
          if Il.Cell b.Il.binary_dst = b.Il.binary_lhs
          then
            let binop = alu_binop b.Il.binary_dst b.Il.binary_rhs in
            let mulop = mul_like b.Il.binary_rhs in

            let divop signed slash =
              Asm.SEQ [|
                (* xor edx edx, then mul_like. *)
                insn_rm_r 0x33 (rc edx) (reg edx);
                mul_like b.Il.binary_rhs signed slash
              |]
            in

            let modop signed slash =
              Asm.SEQ [|
                (* divop, then mov remainder to eax instead. *)
                divop signed slash;
                mov false (rc eax) (ro edx)
              |]
            in

            let shiftop slash =
              let src = b.Il.binary_rhs in
              let dst = b.Il.binary_dst in
              let mask i = Asm.AND (i, Asm.IMM 0xffL) in
              if is_rm8 dst
              then
                match src with
                    Il.Imm (i, _) ->
                      insn_rm_r_imm 0xC0 dst slash TY_u8 (mask i)
                  | Il.Cell (Il.Reg ((Il.Hreg r), _))
                      when r = ecx ->
                      Asm.SEQ [| Asm.BYTE 0xD2; rm_r dst slash |]
                  | _ -> raise Unrecognized
              else
                match src with
                    Il.Imm (i, _) ->
                        insn_rm_r_imm 0xC1 dst slash TY_u8 (mask i)
                  | Il.Cell (Il.Reg ((Il.Hreg r), _))
                      when r = ecx ->
                      Asm.SEQ [| Asm.BYTE 0xD3; rm_r dst slash |]
                  | _ -> raise Unrecognized
            in

              match (b.Il.binary_dst, b.Il.binary_op) with
                  (_, Il.ADD) -> binop { insn="ADD";
                                         immslash=slash0;
                                         rm_dst_op8=0x0;
                                         rm_dst_op32=0x1;
                                         rm_src_op8=0x2;
                                         rm_src_op32=0x3; }
                | (_, Il.SUB) -> binop { insn="SUB";
                                         immslash=slash5;
                                         rm_dst_op8=0x28;
                                         rm_dst_op32=0x29;
                                         rm_src_op8=0x2a;
                                         rm_src_op32=0x2b; }
                | (_, Il.AND) -> binop { insn="AND";
                                         immslash=slash4;
                                         rm_dst_op8=0x20;
                                         rm_dst_op32=0x21;
                                         rm_src_op8=0x22;
                                         rm_src_op32=0x23; }
                | (_, Il.OR) -> binop { insn="OR";
                                        immslash=slash1;
                                        rm_dst_op8=0x08;
                                        rm_dst_op32=0x09;
                                        rm_src_op8=0x0a;
                                        rm_src_op32=0x0b; }
                | (_, Il.XOR) -> binop { insn="XOR";
                                         immslash=slash6;
                                         rm_dst_op8=0x30;
                                         rm_dst_op32=0x31;
                                         rm_src_op8=0x32;
                                         rm_src_op32=0x33; }

                | (_, Il.LSL) -> shiftop slash4
                | (_, Il.LSR) -> shiftop slash5
                | (_, Il.ASR) -> shiftop slash7

                | (Il.Reg (Il.Hreg r, t), Il.UMUL)
                    when (is_ty32 t || is_ty8 t) && r = eax ->
                    mulop false slash4

                | (Il.Reg (Il.Hreg r, t), Il.IMUL)
                    when (is_ty32 t || is_ty8 t) && r = eax ->
                    mulop true slash5

                | (Il.Reg (Il.Hreg r, t), Il.UDIV)
                    when (is_ty32 t || is_ty8 t) && r = eax ->
                    divop false slash6

                | (Il.Reg (Il.Hreg r, t), Il.IDIV)
                    when (is_ty32 t || is_ty8 t) && r = eax ->
                    divop true slash7

                | (Il.Reg (Il.Hreg r, t), Il.UMOD)
                    when (is_ty32 t || is_ty8 t) && r = eax ->
                    modop false slash6

                | (Il.Reg (Il.Hreg r, t), Il.IMOD)
                    when (is_ty32 t || is_ty8 t) && r = eax ->
                    modop true slash7

                | _ -> raise Unrecognized
          else raise Unrecognized
        end
    | _ -> select_insn_misc q.Il.quad_body
;;


let new_emitter_without_vregs _ : Il.emitter =
  Il.new_emitter
    abi.Abi.abi_emit_target_specific
    false None
;;

let select_insns (sess:Session.sess) (qs:Il.quads) : Asm.frag =
  let scopes = Stack.create () in
  let fixups = Stack.create () in
  let append frag =
    Queue.add frag (Stack.top scopes)
  in
  let pop_frags _ =
    Asm.SEQ (queue_to_arr (Stack.pop scopes))
  in
    ignore (Stack.push (Queue.create()) scopes);
    Array.iteri
      begin
        fun i q ->
          begin
            match q.Il.quad_fixup with
                None -> ()
              | Some f -> append (Asm.DEF (f, Asm.MARK))
          end;
          begin
            let qstr _ = Il.string_of_quad reg_str q in
              match q.Il.quad_body with
                  Il.Enter f ->
                    Stack.push f fixups;
                    Stack.push (Queue.create()) scopes;
                | Il.Leave ->
                    append (Asm.DEF (Stack.pop fixups, pop_frags ()))
                | _ ->
                    try
                      let _ =
                        iflog sess (fun _ ->
                                      log sess "quad %d: %s" i (qstr()))
                      in
                      let frag = select_insn q in
                      let _ =
                        iflog sess (fun _ ->
                                      log sess "frag %d: %a" i
                                        Asm.sprintf_frag frag)
                      in
                        append frag
                    with
                    Unrecognized ->
                      Session.fail sess
                        "E:Assembly error: unrecognized quad %d: %s\n%!"
                        i (qstr());
                      ()
          end
      end
      qs;
      pop_frags()
;;

let frags_of_emitted_quads (sess:Session.sess) (e:Il.emitter) : Asm.frag =
  let frag = select_insns sess e.Il.emit_quads in
    if sess.Session.sess_failed
    then raise Unrecognized
    else frag
;;


(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
