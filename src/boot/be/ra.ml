open Il;;
open Common;;

type ctxt =
    {
      ctxt_sess: Session.sess;
      ctxt_n_vregs: int;
      ctxt_abi: Abi.abi;
      mutable ctxt_quads: Il.quads;
      mutable ctxt_next_spill: int;
      mutable ctxt_next_label: int;
      (* More state as necessary. *)
    }
;;

let new_ctxt
    (sess:Session.sess)
    (quads:Il.quads)
    (vregs:int)
    (abi:Abi.abi)
    : ctxt =
  {
    ctxt_sess = sess;
    ctxt_quads = quads;
    ctxt_n_vregs = vregs;
    ctxt_abi = abi;
    ctxt_next_spill = 0;
    ctxt_next_label = 0;
  }
;;

let log (cx:ctxt) =
  Session.log "ra"
    cx.ctxt_sess.Session.sess_log_ra
    cx.ctxt_sess.Session.sess_log_out
;;

let iflog (cx:ctxt) (thunk:(unit -> unit)) : unit =
  if cx.ctxt_sess.Session.sess_log_ra
  then thunk ()
  else ()
;;

let list_to_str list eltstr =
  (String.concat "," (List.map eltstr (List.sort compare list)))
;;

let next_spill (cx:ctxt) : int =
  let i = cx.ctxt_next_spill in
    cx.ctxt_next_spill <- i + 1;
    i
;;

let next_label (cx:ctxt) : string =
  let i = cx.ctxt_next_label in
    cx.ctxt_next_label <- i + 1;
    (".L" ^ (string_of_int i))
;;

exception Ra_error of string ;;

let convert_labels (cx:ctxt) : unit =
  let quad_fixups = Array.map (fun q -> q.quad_fixup) cx.ctxt_quads in
  let qp_code (_:Il.quad_processor) (c:Il.code) : Il.code =
    match c with
        Il.CodeLabel lab ->
          let fix =
            match quad_fixups.(lab) with
                None ->
                  let fix = new_fixup (next_label cx) in
                    begin
                      quad_fixups.(lab) <- Some fix;
                      fix
                    end
              | Some f -> f
          in
            Il.CodePtr (Il.ImmPtr (fix, Il.CodeTy))
      | _ -> c
  in
  let qp = { Il.identity_processor
             with Il.qp_code = qp_code }
  in
    Il.rewrite_quads qp cx.ctxt_quads;
    Array.iteri (fun i fix ->
                   cx.ctxt_quads.(i) <- { cx.ctxt_quads.(i) with
                                            Il.quad_fixup = fix })
      quad_fixups;
;;

let convert_pre_spills
    (cx:ctxt)
    (mkspill:(Il.spill -> Il.mem))
    : int =
  let n = ref 0 in
  let qp_mem (_:Il.quad_processor) (a:Il.mem) : Il.mem =
    match a with
        Il.Spill i ->
          begin
            if i+1 > (!n)
            then n := i+1;
            mkspill i
          end
      | _ -> a
  in
  let qp = Il.identity_processor in
  let qp = { qp with
               Il.qp_mem = qp_mem  }
  in
    begin
      Il.rewrite_quads qp cx.ctxt_quads;
      !n
    end
;;

let kill_quad (i:int) (cx:ctxt) : unit =
  cx.ctxt_quads.(i) <-
    { Il.deadq with
        Il.quad_fixup = cx.ctxt_quads.(i).Il.quad_fixup }
;;

let kill_redundant_moves (cx:ctxt) : unit =
  let process_quad i q =
    match q.Il.quad_body with
        Il.Unary u when
          ((Il.is_mov u.Il.unary_op) &&
             (Il.Cell u.Il.unary_dst) = u.Il.unary_src) ->
            kill_quad i cx
      | _ -> ()
  in
    Array.iteri process_quad cx.ctxt_quads
;;

let quad_jump_target_labels (q:quad) : Il.label list =
  match q.Il.quad_body with
      Il.Jmp jmp ->
        begin
          match jmp.Il.jmp_targ with
              Il.CodeLabel lab  -> [ lab ]
            | _ -> []
        end
    | _ -> []
;;

let quad_used_vregs (q:quad) : Il.vreg list =
  let vregs = ref [] in
  let qp_reg _ r =
    match r with
        Il.Vreg v -> (vregs := (v :: (!vregs)); r)
      | _ -> r
  in
  let qp_cell_write qp c =
    match c with
        Il.Reg _ -> c
      | Il.Mem (a, b) -> Il.Mem (qp.qp_mem qp a, b)
  in
  let qp = { Il.identity_processor with
               Il.qp_reg = qp_reg;
               Il.qp_cell_write = qp_cell_write }
  in
    ignore (Il.process_quad qp q);
    !vregs
;;

let quad_defined_vregs (q:quad) : Il.vreg list =
  let vregs = ref [] in
  let qp_cell_write _ c =
    match c with
        Il.Reg (Il.Vreg v, _) -> (vregs := (v :: (!vregs)); c)
      | _ -> c
  in
  let qp = { Il.identity_processor with
               Il.qp_cell_write = qp_cell_write }
  in
    ignore (Il.process_quad qp q);
    !vregs
;;

let quad_is_unconditional_jump (q:quad) : bool =
  match q.Il.quad_body with
      Il.Jmp { jmp_op = Il.JMP; jmp_targ = _ } -> true
    | Il.Ret -> true
    | _ -> false
;;

let calculate_live_bitvectors
    (cx:ctxt)
    : ((Bits.t array) * (Bits.t array)) =

  iflog cx (fun _ -> log cx "calculating live bitvectors");

  let quads = cx.ctxt_quads in
  let n_quads = Array.length quads in
  let n_vregs = cx.ctxt_n_vregs in
  let new_bitv _ = Bits.create n_vregs false in
  let new_true_bitv _ = Bits.create n_vregs true in
  let (live_in_vregs:Bits.t array) = Array.init n_quads new_bitv in
  let (live_out_vregs:Bits.t array) = Array.init n_quads new_bitv in

  let (quad_used_vrs:Bits.t array) = Array.init n_quads new_bitv in
  let (quad_not_defined_vrs:Bits.t array) =
    Array.init n_quads new_true_bitv
  in
  let (quad_uncond_jmp:bool array) = Array.make n_quads false in
  let (quad_jmp_targs:(Il.label list) array) = Array.make n_quads [] in

  (* Working bit-vector. *)
  let scratch = new_bitv() in
  let changed = ref true in

  (* bit-vector helpers. *)
    (* Setup pass. *)
    for i = 0 to n_quads - 1 do
      let q = quads.(i) in
        quad_uncond_jmp.(i) <- quad_is_unconditional_jump q;
        quad_jmp_targs.(i) <- quad_jump_target_labels q;
        List.iter
          (fun v -> Bits.set quad_used_vrs.(i) v true)
          (quad_used_vregs q);
        List.iter
          (fun v -> Bits.set quad_not_defined_vrs.(i) v false)
          (quad_defined_vregs q);
    done;

    while !changed do
      changed := false;
      iflog cx
        (fun _ ->
           log cx "iterating inner bitvector calculation over %d quads"
             n_quads);
      for i = n_quads - 1 downto 0 do

        let note_change b = if b then changed := true in
        let live_in = live_in_vregs.(i) in
        let live_out = live_out_vregs.(i) in
        let used = quad_used_vrs.(i) in
        let not_defined = quad_not_defined_vrs.(i) in

          (* Union in the vregs we use. *)
          note_change (Bits.union live_in used);

          (* Union in all our jump targets. *)
          List.iter
            (fun i -> note_change (Bits.union live_out live_in_vregs.(i)))
            (quad_jmp_targs.(i));

          (* Union in our block successor if we have one *)
          if i < (n_quads - 1) && (not (quad_uncond_jmp.(i)))
          then note_change (Bits.union live_out live_in_vregs.(i+1));

          (* Propagate live-out to live-in on anything we don't define. *)
          ignore (Bits.copy scratch not_defined);
          ignore (Bits.intersect scratch live_out);
          note_change (Bits.union live_in scratch);

      done;
    done;
    iflog cx
      begin
        fun _ ->
          log cx "finished calculating live bitvectors";
          log cx "=========================";
          for q = 0 to n_quads - 1 do
            let buf = Buffer.create 128 in
              for v = 0 to (n_vregs - 1)
              do
                if ((Bits.get live_in_vregs.(q) v)
                    && (Bits.get live_out_vregs.(q) v))
                then Printf.bprintf buf " %-2d" v
                else Buffer.add_string buf "   "
              done;
              log cx "[%6d] live vregs: %s" q (Buffer.contents buf)
          done;
          log cx "========================="
      end;
    (live_in_vregs, live_out_vregs)
;;


let is_end_of_basic_block (q:quad) : bool =
  match q.Il.quad_body with
      Il.Jmp _ -> true
    | Il.Ret -> true
    | _ -> false
;;

let is_beginning_of_basic_block (q:quad) : bool =
  match q.Il.quad_fixup with
      None -> false
    | Some _ -> true
;;

let dump_quads cx =
  let f = cx.ctxt_abi.Abi.abi_str_of_hardreg in
  let len = (Array.length cx.ctxt_quads) - 1 in
  let ndigits_of n = (int_of_float (log10 (float_of_int n))) in
  let padded_num n maxnum =
    let ndigits = ndigits_of n in
    let maxdigits = ndigits_of maxnum in
    let pad = String.make (maxdigits - ndigits) ' ' in
      Printf.sprintf "%s%d" pad n
  in
  let padded_str str maxlen =
    let pad = String.make (maxlen - (String.length str)) ' ' in
      Printf.sprintf "%s%s" pad str
  in
  let maxlablen = ref 0 in
  for i = 0 to len
  do
    let q = cx.ctxt_quads.(i) in
    match q.quad_fixup with
        None -> ()
      | Some f ->
          maxlablen := max (!maxlablen) ((String.length f.fixup_name) + 1)
  done;
  for i = 0 to len
  do
    let q = cx.ctxt_quads.(i) in
    let qs = (string_of_quad f q) in
    let lab = match q.quad_fixup with
        None -> ""
      | Some f -> f.fixup_name ^ ":"
    in
      iflog cx
        (fun _ ->
           log cx "[%s] %s %s"
             (padded_num i len) (padded_str lab (!maxlablen)) qs)
  done
;;

let calculate_vreg_constraints
    (cx:ctxt)
    (constraints:(Il.vreg,Bits.t) Hashtbl.t)
    (q:quad)
    : unit =
  let abi = cx.ctxt_abi in
    Hashtbl.clear constraints;
    abi.Abi.abi_constrain_vregs q constraints;
    iflog cx
      begin
        fun _ ->
          let hr_str = cx.ctxt_abi.Abi.abi_str_of_hardreg in
            log cx "constraints for quad %s"
              (string_of_quad hr_str q);
            let qp_reg _ r =
              begin
                match r with
                    Il.Hreg _ -> ()
                  | Il.Vreg v ->
                      match htab_search constraints v with
                          None -> log cx "<v%d> unconstrained" v
                        | Some c ->
                            let hregs = Bits.to_list c in
                              log cx "<v%d> constrained to hregs: [%s]"
                                v (list_to_str hregs hr_str)
              end;
              r
            in
              ignore (Il.process_quad { Il.identity_processor with
                                          Il.qp_reg = qp_reg } q)
      end
;;

(* Simple local register allocator. Nothing fancy. *)
let reg_alloc
    (sess:Session.sess)
    (quads:Il.quads)
    (vregs:int)
    (abi:Abi.abi) =
 try
    let cx = new_ctxt sess quads vregs abi in
    let _ =
      iflog cx
        begin
          fun _ ->
            log cx "un-allocated quads:";
            dump_quads cx
        end
    in

    (* Work out pre-spilled slots and allocate 'em. *)
    let spill_slot (s:Il.spill) = abi.Abi.abi_spill_slot s in
    let n_pre_spills = convert_pre_spills cx spill_slot in

    let (live_in_vregs, live_out_vregs) =
      calculate_live_bitvectors cx
    in
      (* vreg idx -> hreg bits.t *)
    let (vreg_constraints:(Il.vreg,Bits.t) Hashtbl.t) =
      Hashtbl.create 0
    in
    let inactive_hregs = ref [] in (* [hreg] *)
    let active_hregs = ref [] in (* [hreg] *)
    let dirty_vregs = Hashtbl.create 0 in (* vreg -> () *)
    let hreg_to_vreg = Hashtbl.create 0 in  (* hreg -> vreg *)
    let vreg_to_hreg = Hashtbl.create 0 in (* vreg -> hreg *)
    let vreg_to_spill = Hashtbl.create 0 in (* vreg -> spill *)
    let (word_ty:Il.scalar_ty) = Il.ValTy abi.Abi.abi_word_bits in
    let vreg_spill_cell v =
      Il.Mem ((spill_slot (Hashtbl.find vreg_to_spill v)),
              Il.ScalarTy word_ty)
    in
    let newq = ref [] in
    let fixup = ref None in
    let prepend q =
      newq := {q with quad_fixup = !fixup} :: (!newq);
      fixup := None
    in
    let hr h = Il.Reg (Il.Hreg h, Il.voidptr_t) in
    let hr_str = cx.ctxt_abi.Abi.abi_str_of_hardreg in
    let clean_hreg i hreg =
      if (Hashtbl.mem hreg_to_vreg hreg) &&
        (hreg < cx.ctxt_abi.Abi.abi_n_hardregs)
      then
        let vreg = Hashtbl.find hreg_to_vreg hreg in
          if Hashtbl.mem dirty_vregs vreg
          then
            begin
              Hashtbl.remove dirty_vregs vreg;
              if (Bits.get (live_out_vregs.(i)) vreg) ||
                (Bits.get (live_in_vregs.(i)) vreg)
              then
                let spill_idx =
                  if Hashtbl.mem vreg_to_spill vreg
                  then Hashtbl.find vreg_to_spill vreg
                  else
                    begin
                      let s = next_spill cx in
                        Hashtbl.replace vreg_to_spill vreg s;
                        s
                    end
                in
                let spill_mem = spill_slot spill_idx in
                let spill_cell = Il.Mem (spill_mem, Il.ScalarTy word_ty) in
                  iflog cx
                    (fun _ ->
                       log cx "spilling <%d> from %s to %s"
                         vreg (hr_str hreg) (string_of_mem
                                               hr_str spill_mem));
                  prepend (Il.mk_quad
                             (Il.umov spill_cell (Il.Cell (hr hreg))));
              else ()
            end
          else ()
      else ()
    in

    let inactivate_hreg hreg =
      if (Hashtbl.mem hreg_to_vreg hreg) &&
        (hreg < cx.ctxt_abi.Abi.abi_n_hardregs)
      then
        let vreg = Hashtbl.find hreg_to_vreg hreg in
          Hashtbl.remove vreg_to_hreg vreg;
          Hashtbl.remove hreg_to_vreg hreg;
          active_hregs := List.filter (fun x -> x != hreg) (!active_hregs);
          inactive_hregs := hreg :: (!inactive_hregs);
      else ()
    in

    let spill_specific_hreg i hreg =
      clean_hreg i hreg;
      inactivate_hreg hreg
    in

    let rec select_constrained
        (constraints:Bits.t)
        (hregs:Il.hreg list)
        : Il.hreg option =
      match hregs with
          [] -> None
        | h::hs ->
            if Bits.get constraints h
            then Some h
            else select_constrained constraints hs
    in

    let spill_constrained constrs i =
      match select_constrained constrs (!active_hregs) with
          None ->
            raise (Ra_error ("unable to spill according to constraint"));
        | Some h ->
            begin
              spill_specific_hreg i h;
              h
            end
    in

    let all_hregs = Bits.create abi.Abi.abi_n_hardregs true in

    let spill_all_regs i =
      while (!active_hregs) != []
      do
        let _ = spill_constrained all_hregs i in
          ()
      done
    in

    let reload vreg hreg =
      if Hashtbl.mem vreg_to_spill vreg
      then
        prepend (Il.mk_quad
                   (Il.umov
                      (hr hreg)
                      (Il.Cell (vreg_spill_cell vreg))))
      else ()
    in

    let get_vreg_constraints v =
      match htab_search vreg_constraints v with
          None -> all_hregs
        | Some c -> c
    in


    let use_vreg def i vreg =
      if Hashtbl.mem vreg_to_hreg vreg
      then
        begin
          let h = Hashtbl.find vreg_to_hreg vreg in
          iflog cx (fun _ -> log cx "found cached assignment %s for <v%d>"
                      (hr_str h) vreg);
            h
        end
      else
        let hreg =
          let constrs = get_vreg_constraints vreg in
            match select_constrained constrs (!inactive_hregs) with
                None ->
                  let h = spill_constrained constrs i in
                    iflog cx
                      (fun _ ->
                         log cx "selected %s to spill and use for <v%d>"
                         (hr_str h) vreg);
                    h
              | Some h ->
                  iflog cx (fun _ -> log cx "selected inactive %s for <v%d>"
                              (hr_str h) vreg);
                  h
        in
          inactive_hregs :=
            List.filter (fun x -> x != hreg) (!inactive_hregs);
          active_hregs := (!active_hregs) @ [hreg];
          Hashtbl.replace hreg_to_vreg hreg vreg;
          Hashtbl.replace vreg_to_hreg vreg hreg;
          if def
          then ()
          else
            reload vreg hreg;
          hreg
    in
    let qp_reg def i _ r =
      match r with
          Il.Hreg h -> (spill_specific_hreg i h; r)
        | Il.Vreg v -> (Il.Hreg (use_vreg def i v))
    in
    let qp_cell def i qp c =
      match c with
          Il.Reg (r, b) -> Il.Reg (qp_reg def i qp r, b)
        | Il.Mem  (a, b) ->
            let qp = { qp with Il.qp_reg = qp_reg false i } in
              Il.Mem (qp.qp_mem qp a, b)
    in
    let qp i = { Il.identity_processor with
                   Il.qp_cell_read = qp_cell false i;
                   Il.qp_cell_write = qp_cell true i;
                   Il.qp_reg = qp_reg false i }
    in
      cx.ctxt_next_spill <- n_pre_spills;
      convert_labels cx;
      for i = 0 to cx.ctxt_abi.Abi.abi_n_hardregs - 1
      do
        inactive_hregs := i :: (!inactive_hregs)
      done;
      for i = 0 to (Array.length cx.ctxt_quads) - 1
      do
        let quad = cx.ctxt_quads.(i) in
        let _ = calculate_vreg_constraints cx vreg_constraints quad in
        let clobbers = cx.ctxt_abi.Abi.abi_clobbers quad in
        let used = quad_used_vregs quad in
        let defined = quad_defined_vregs quad in

          begin

            (* If the quad has any nontrivial vreg constraints, regfence.
             * This is awful but it saves us from cached/constrained
             * interference as was found in issue #152. *)
            if List.exists
              (fun v -> not (Bits.equal (get_vreg_constraints v) all_hregs))
              used
            then
              begin
                (* Regfence. *)
                spill_all_regs i;
                (* Check for over-constrained-ness after any such regfence. *)
                let vreg_constrs v =
                  (v, Bits.to_list (get_vreg_constraints v))
                in
                let constrs = List.map vreg_constrs (used @ defined) in
                let constrs_collide (v1,c1) =
                  if List.length c1 <> 1
                  then false
                  else
                    List.exists
                      (fun (v2,c2) -> if v1 = v2 then false else c1 = c2)
                      constrs
                in
                  if List.exists constrs_collide constrs
                  then raise (Ra_error ("over-constrained vregs"));
              end;

            if List.exists (fun def -> List.mem def clobbers) defined
            then raise (Ra_error ("clobber and defined sets overlap"));
            iflog cx
              begin
                fun _ ->
                  let hr (v:int) : string =
                    if Hashtbl.mem vreg_to_hreg v
                    then hr_str (Hashtbl.find vreg_to_hreg v)
                    else "??"
                  in
                  let vr_str (v:int) : string =
                    Printf.sprintf "v%d=%s" v (hr v)
                  in
                  let lstr lab ls fn =
                    if List.length ls = 0
                    then ()
                    else log cx "\t%s: [%s]" lab (list_to_str ls fn)
                  in
                    log cx "processing quad %d = %s"
                      i (string_of_quad hr_str quad);
                    (lstr "dirt" (htab_keys dirty_vregs) vr_str);
                    (lstr "clob" clobbers hr_str);
                    (lstr "in" (Bits.to_list live_in_vregs.(i)) vr_str);
                    (lstr "out" (Bits.to_list live_out_vregs.(i)) vr_str);
                    (lstr "use" used vr_str);
                    (lstr "def" defined vr_str);
              end;
            List.iter (clean_hreg i) clobbers;
            if is_beginning_of_basic_block quad
            then
              begin
                spill_all_regs i;
                fixup := quad.quad_fixup;
                prepend (Il.process_quad (qp i) quad)
              end
            else
              begin
                fixup := quad.quad_fixup;
                let newq = (Il.process_quad (qp i) quad) in
                  begin
                    if is_end_of_basic_block quad
                    then spill_all_regs i
                    else ()
                  end;
                  prepend newq
              end
          end;
          List.iter inactivate_hreg clobbers;
          List.iter (fun i -> Hashtbl.replace dirty_vregs i ()) defined;
      done;
      cx.ctxt_quads <- Array.of_list (List.rev (!newq));
      kill_redundant_moves cx;

      iflog cx
        begin
          fun _ ->
            log cx "spills: %d pre-spilled, %d total"
              n_pre_spills cx.ctxt_next_spill;
            log cx "register-allocated quads:";
            dump_quads cx;
        end;
      (cx.ctxt_quads, cx.ctxt_next_spill)

 with
     Ra_error s ->
       Session.fail sess "RA error: %s\n" s;
       (quads, 0)

;;


(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
