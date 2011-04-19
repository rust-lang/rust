open Semant;;
open Common;;


let log cx = Session.log "typestate"
  (should_log cx cx.ctxt_sess.Session.sess_log_typestate)
  cx.ctxt_sess.Session.sess_log_out
;;

let iflog cx thunk =
  if (should_log cx cx.ctxt_sess.Session.sess_log_typestate)
  then thunk ()
  else ()
;;

type node_graph = (node_id, (node_id list)) Hashtbl.t;;
type sibling_map = (node_id, node_id) Hashtbl.t;;

type typestate_tables =
    { ts_constrs: (constr_id,constr_key) Hashtbl.t;
      ts_constr_ids: (constr_key,constr_id) Hashtbl.t;
      ts_preconditions: (node_id,Bits.t) Hashtbl.t;
      ts_postconditions: (node_id,Bits.t) Hashtbl.t;
      ts_prestates: (node_id,Bits.t) Hashtbl.t;
      ts_poststates: (node_id,Bits.t) Hashtbl.t;
      ts_graph: node_graph;
      ts_stmts: Ast.stmt Stack.t;
      ts_maxid: int ref;
    }
;;

let new_tables _ =
  { ts_constrs = Hashtbl.create 0;
    ts_constr_ids = Hashtbl.create 0;
    ts_preconditions = Hashtbl.create 0;
    ts_postconditions = Hashtbl.create 0;
    ts_poststates = Hashtbl.create 0;
    ts_prestates = Hashtbl.create 0;
    ts_graph = Hashtbl.create 0;
    ts_stmts = Stack.create ();
    ts_maxid = ref 0 }
;;

type item_tables = (node_id, typestate_tables) Hashtbl.t
;;

let get_tables (all_tables:item_tables) (n:node_id) : typestate_tables =
  htab_search_or_add all_tables n new_tables
;;

let tables_managing_visitor
    (all_tables:item_tables)
    (tables_stack:typestate_tables Stack.t)
    (inner:Walk.visitor)
    : Walk.visitor =

  let enter id =
    Stack.push (get_tables all_tables id) tables_stack
  in

  let leave _ =
    ignore (Stack.pop tables_stack)
  in

  let visit_mod_item_pre n p i =
    enter i.id;
    inner.Walk.visit_mod_item_pre n p i
  in

  let visit_mod_item_post n p i =
    inner.Walk.visit_mod_item_post n p i;
    leave()
  in

  let visit_obj_fn_pre obj ident fn =
    enter fn.id;
    inner.Walk.visit_obj_fn_pre obj ident fn
  in

  let visit_obj_fn_post obj ident fn =
    inner.Walk.visit_obj_fn_post obj ident fn;
    leave()
  in

  let visit_obj_drop_pre obj b =
    enter b.id;
    inner.Walk.visit_obj_drop_pre obj b
  in

  let visit_obj_drop_post obj b =
    inner.Walk.visit_obj_drop_post obj b;
    leave()
  in
    { inner with
        Walk.visit_mod_item_pre = visit_mod_item_pre;
        Walk.visit_mod_item_post = visit_mod_item_post;
        Walk.visit_obj_fn_pre = visit_obj_fn_pre;
        Walk.visit_obj_fn_post = visit_obj_fn_post;
        Walk.visit_obj_drop_pre = visit_obj_drop_pre;
        Walk.visit_obj_drop_post = visit_obj_drop_post; }
;;


let name_base_to_slot_key (nb:Ast.name_base) : Ast.slot_key =
  match nb with
      Ast.BASE_ident ident -> Ast.KEY_ident ident
    | Ast.BASE_temp tmp -> Ast.KEY_temp tmp
    | Ast.BASE_app _ -> bug () "name_base_to_slot_key on parametric name"
;;

let determine_constr_key
    (cx:ctxt)
    (scopes:(scope list))
    (formal_base:node_id option)
    (c:Ast.constr)
    : constr_key =

  let cid =
    match lookup_by_name cx [] scopes c.Ast.constr_name with
        RES_ok (_, cid) ->
          if defn_id_is_item cx cid
          then
            begin
              match Hashtbl.find cx.ctxt_all_item_types cid with
                  Ast.TY_fn _ -> cid
                | _ -> bug () "bad type of predicate"
            end
          else
            bug () "slot used as predicate"
      | RES_failed _ -> bug () "predicate not found"
  in

  let constr_arg_of_carg carg =
    match carg with
        Ast.CARG_path pth ->
          let rec node_base_of pth =
            match pth with
                Ast.CARG_base Ast.BASE_formal ->
                  begin
                    match formal_base with
                        Some id -> id
                      | None ->
                          bug () "formal symbol * used in free constraint"
                  end
              | Ast.CARG_ext (pth, _) -> node_base_of pth
              | Ast.CARG_base (Ast.BASE_named nb) ->
                  begin
                    match lookup_by_name cx [] scopes (Ast.NAME_base nb) with
                        RES_failed _ -> bug () "constraint-arg not found"
                      | RES_ok (_, aid) ->
                          if defn_id_is_slot cx aid
                          then
                            if type_has_state cx
                              (strip_mutable_or_constrained_ty
                                 (slot_ty (get_slot cx aid)))
                            then err (Some aid)
                              "predicate applied to slot of state type"
                            else aid
                          else
                            (* Items are always constant, they're ok. 
                             * Weird to be using them in a constr, but ok. *)
                            aid
                  end
          in
            Constr_arg_node (node_base_of pth, pth)

      | Ast.CARG_lit lit -> Constr_arg_lit lit
  in
    Constr_pred (cid, Array.map constr_arg_of_carg c.Ast.constr_args)
;;

let fmt_constr_key cx ckey =
  match ckey with
      Constr_pred (cid, args) ->
        let fmt_constr_arg carg =
          match carg with
              Constr_arg_lit lit ->
                Fmt.fmt_to_str Ast.fmt_lit lit
            | Constr_arg_node (id, pth) ->
                let rec fmt_pth pth =
                  match pth with
                      Ast.CARG_base _ ->
                        if defn_id_is_slot cx id
                        then
                          let key = Hashtbl.find cx.ctxt_slot_keys id in
                            Fmt.fmt_to_str Ast.fmt_slot_key key
                        else
                          let n = Hashtbl.find cx.ctxt_all_item_names id in
                            Fmt.fmt_to_str Ast.fmt_name n
                    | Ast.CARG_ext (pth, nc) ->
                        let b = fmt_pth pth in
                          b ^ (Fmt.fmt_to_str Ast.fmt_name_component nc)
                in
                  fmt_pth pth
        in
        let pred_name = Hashtbl.find cx.ctxt_all_item_names cid in
          Printf.sprintf "%s(%s)"
            (Fmt.fmt_to_str Ast.fmt_name pred_name)
            (String.concat ", "
               (List.map
                  fmt_constr_arg
                  (Array.to_list args)))

    | Constr_init n when Hashtbl.mem cx.ctxt_slot_keys n ->
        Printf.sprintf "<init #%d = %s>"
          (int_of_node n)
          (Fmt.fmt_to_str Ast.fmt_slot_key (Hashtbl.find cx.ctxt_slot_keys n))
    | Constr_init n ->
        Printf.sprintf "<init #%d>" (int_of_node n)
;;

let entry_keys header constrs resolver =
  let init_keys =
    Array.map
      (fun (sloti, _) -> (Constr_init sloti.id))
      header
  in
  let names =
    Array.map
      (fun (_, ident) -> (Some (Ast.BASE_ident ident)))
      header
  in
  let input_constrs =
    Array.map (apply_names_to_constr names) constrs in
  let input_keys = Array.map resolver input_constrs in
    (input_keys, init_keys)
;;

let obj_keys ob resolver =
    entry_keys ob.Ast.obj_state ob.Ast.obj_constrs resolver
;;

let fn_keys fn resolver =
    entry_keys fn.Ast.fn_input_slots fn.Ast.fn_input_constrs resolver
;;


let rec lval_slots (cx:ctxt) (lv:Ast.lval) : node_id array =
  match lv with
      Ast.LVAL_base nbi ->
        let defn_id = lval_base_id_to_defn_base_id cx nbi.id in
          if defn_id_is_slot cx defn_id
          then [| defn_id |]
          else [| |]
    | Ast.LVAL_ext (lv, Ast.COMP_named _)
    | Ast.LVAL_ext (lv, Ast.COMP_deref) -> lval_slots cx lv
    | Ast.LVAL_ext (lv, Ast.COMP_atom a) ->
        Array.append (lval_slots cx lv) (atom_slots cx a)

and atom_slots (cx:ctxt) (a:Ast.atom) : node_id array =
  match a with
      Ast.ATOM_literal _ -> [| |]
    | Ast.ATOM_lval lv -> lval_slots cx lv
    | Ast.ATOM_pexp _ -> bug () "Typestate.atom_slots on ATOM_pexp"
;;

let lval_option_slots (cx:ctxt) (lv:Ast.lval option) : node_id array =
  match lv with
      None -> [| |]
    | Some lv -> lval_slots cx lv
;;

let atoms_slots (cx:ctxt) (az:Ast.atom array) : node_id array =
  Array.concat (List.map (atom_slots cx) (Array.to_list az))
;;

let tup_inputs_slots (cx:ctxt) (az:Ast.tup_input array) : node_id array =
  Array.concat (List.map (atom_slots cx) (Array.to_list (Array.map snd az)))
;;

let rec_inputs_slots (cx:ctxt)
    (inputs:Ast.rec_input array) : node_id array =
  Array.concat (List.map
                  (fun (_, _, atom) -> atom_slots cx atom)
                  (Array.to_list inputs))
;;

let expr_slots (cx:ctxt) (e:Ast.expr) : node_id array =
    match e with
        Ast.EXPR_binary (_, a, b) ->
          Array.append (atom_slots cx a) (atom_slots cx b)
      | Ast.EXPR_unary (_, u) -> atom_slots cx u
      | Ast.EXPR_atom a -> atom_slots cx a
;;

let constr_id_assigning_visitor
    (cx:ctxt)
    (tables_stack:typestate_tables Stack.t)
    (scopes:(scope list) ref)
    (inner:Walk.visitor)
    : Walk.visitor =

  let tables _ = Stack.top tables_stack in

  let resolve_constr_to_key
      (formal_base:node_id)
      (constr:Ast.constr)
      : constr_key =
    determine_constr_key cx (!scopes) (Some formal_base) constr
  in

  let note_constr_key key =
    let ts = tables () in
    let idref = ts.ts_maxid in
      if not (Hashtbl.mem ts.ts_constr_ids key)
      then
        begin
          let cid = Constr (!idref) in
            iflog cx
              (fun _ -> log cx "assigning constr id #%d to constr %s"
                 (!idref) (fmt_constr_key cx key));
            incr idref;
            htab_put ts.ts_constrs cid key;
            htab_put ts.ts_constr_ids key cid;
        end
  in

  let note_keys = Array.iter note_constr_key in

  let visit_mod_item_pre n p i =
    let resolver = resolve_constr_to_key i.id in
    begin
    match i.node.Ast.decl_item with
        Ast.MOD_ITEM_fn f ->
          let (input_keys, init_keys) = fn_keys f resolver in
            note_keys input_keys;
            note_keys init_keys
      | Ast.MOD_ITEM_obj ob ->
          let (input_keys, init_keys) = obj_keys ob resolver in
            note_keys input_keys;
            note_keys init_keys
      | _ -> ()
    end;
    inner.Walk.visit_mod_item_pre n p i
  in

  let visit_obj_fn_pre obj ident fn =
    let (obj_input_keys, obj_init_keys) =
      obj_keys obj.node (resolve_constr_to_key obj.id)
    in
      note_keys obj_input_keys;
      note_keys obj_init_keys;
      inner.Walk.visit_obj_fn_pre obj ident fn
  in

  let visit_obj_drop_pre obj b =
    let (obj_input_keys, obj_init_keys) =
      obj_keys obj.node (resolve_constr_to_key obj.id)
    in
      note_keys obj_input_keys;
      note_keys obj_init_keys;
      inner.Walk.visit_obj_drop_pre obj b
  in

  let visit_constr_pre formal_base c =
    let key = determine_constr_key cx (!scopes) formal_base c in
      note_constr_key key;
      inner.Walk.visit_constr_pre formal_base c
  in
    (* 
     * We want to generate, for any call site, a variant of 
     * the callee's entry typestate specialized to the arguments
     * that the caller passes.
     * 
     * Also, for any slot-decl node, we have to generate a 
     * variant of Constr_init for the slot (because the slot is
     * the sort of thing that can vary in init-ness over time).
     *)
  let visit_stmt_pre s =
    begin
      match s.node with
          Ast.STMT_call (_, lv, args) ->
            let defn_id = lval_base_defn_id cx lv in
            let defn_ty = lval_ty cx lv in
              begin
                match defn_ty with
                    Ast.TY_fn (tsig,_) ->
                      let constrs = tsig.Ast.sig_input_constrs in
                      let names = atoms_to_names args in
                      let constrs' =
                        Array.map (apply_names_to_constr names) constrs
                      in
                        Array.iter (visit_constr_pre (Some defn_id)) constrs'

                  | _ -> ()
              end

        | _ -> ()
    end;
    inner.Walk.visit_stmt_pre s
  in

  let visit_slot_identified_pre s =
    note_constr_key (Constr_init s.id);
    inner.Walk.visit_slot_identified_pre s
  in
    { inner with
        Walk.visit_mod_item_pre = visit_mod_item_pre;
        Walk.visit_obj_fn_pre = visit_obj_fn_pre;
        Walk.visit_obj_drop_pre = visit_obj_drop_pre;
        Walk.visit_slot_identified_pre = visit_slot_identified_pre;
        Walk.visit_stmt_pre = visit_stmt_pre;
        Walk.visit_constr_pre = visit_constr_pre }
;;

let bitmap_assigning_visitor
    (cx:ctxt)
    (tables_stack:typestate_tables Stack.t)
    (inner:Walk.visitor)
    : Walk.visitor =

  let tables _ = Stack.top tables_stack in

  let visit_stmt_pre s =
    let ts = tables () in
    let idref = ts.ts_maxid in
      iflog cx (fun _ -> log cx "building %d-entry bitmap for node %d"
                  (!idref) (int_of_node s.id));
      htab_put ts.ts_preconditions s.id (Bits.create (!idref) false);
      htab_put ts.ts_postconditions s.id (Bits.create (!idref) false);
      htab_put ts.ts_prestates s.id (Bits.create (!idref) false);
      htab_put ts.ts_poststates s.id (Bits.create (!idref) false);
      inner.Walk.visit_stmt_pre s
  in
  let visit_block_pre b =
    let ts = tables () in
    let idref = ts.ts_maxid in
      iflog cx (fun _ -> log cx "building %d-entry bitmap for node %d"
                  (!idref) (int_of_node b.id));
      htab_put ts.ts_preconditions b.id (Bits.create (!idref) false);
      htab_put ts.ts_postconditions b.id (Bits.create (!idref) false);
      htab_put ts.ts_prestates b.id (Bits.create (!idref) false);
      htab_put ts.ts_poststates b.id (Bits.create (!idref) false);
      inner.Walk.visit_block_pre b
  in
    { inner with
        Walk.visit_stmt_pre = visit_stmt_pre;
        Walk.visit_block_pre = visit_block_pre }
;;

type slots_stack = node_id Stack.t;;
type block_slots_stack = slots_stack Stack.t;;
type frame_block_slots_stack = block_slots_stack Stack.t;;
type loop_block_slots_stack = block_slots_stack option Stack.t;;

(* Like ret drops slots from all blocks in the frame
 * break from a simple loop drops slots from all block in a loop
 *)

let (loop_blocks:loop_block_slots_stack) =
  let s = Stack.create() in Stack.push None s; s

let condition_assigning_visitor
    (cx:ctxt)
    (tables_stack:typestate_tables Stack.t)
    (scopes:(scope list) ref)
    (inner:Walk.visitor)
    : Walk.visitor =

  let tables _ = Stack.top tables_stack in

  let raise_bits (bitv:Bits.t) (keys:constr_key array) : unit =
    let ts = tables () in
      Array.iter
        (fun key ->
           let cid = Hashtbl.find ts.ts_constr_ids key in
           let i = int_of_constr cid in
             iflog cx (fun _ -> log cx "setting bit %d, constraint %s"
                         i (fmt_constr_key cx key));
             Bits.set bitv (int_of_constr cid) true)
        keys
  in

  let slot_inits ss = Array.map (fun s -> Constr_init s) ss in

  let raise_postcondition (id:node_id) (keys:constr_key array) : unit =
    let ts = tables () in
    let bitv = Hashtbl.find ts.ts_postconditions id in
      raise_bits bitv keys
  in

  let raise_precondition (id:node_id) (keys:constr_key array) : unit =
    let ts = tables () in
    let bitv = Hashtbl.find ts.ts_preconditions id in
      raise_bits bitv keys
  in

  let raise_pre_post_cond (id:node_id) (keys:constr_key array) : unit =
    raise_precondition id keys;
    raise_postcondition id keys;
  in

  let resolve_constr_to_key
      (formal_base:node_id option)
      (constr:Ast.constr)
      : constr_key =
    determine_constr_key cx (!scopes) formal_base constr
  in

  let raise_entry_state input_keys init_keys block =
    iflog cx
      (fun _ -> log cx
         "setting entry state as block %d postcondition (\"entry\" prestate)"
         (int_of_node block.id));
    raise_postcondition block.id input_keys;
    raise_postcondition block.id init_keys;
    iflog cx (fun _ -> log cx "done setting block postcondition")
  in

  let visit_mod_item_pre n p i =
    begin
      match i.node.Ast.decl_item with
          Ast.MOD_ITEM_fn f ->
            let (input_keys, init_keys) =
              fn_keys f (resolve_constr_to_key (Some i.id))
            in
              raise_entry_state input_keys init_keys f.Ast.fn_body

        | _ -> ()
    end;
    inner.Walk.visit_mod_item_pre n p i
  in

  let visit_obj_fn_pre obj ident fn =
    let (obj_input_keys, obj_init_keys) =
      obj_keys obj.node (resolve_constr_to_key (Some obj.id))
    in
    let (fn_input_keys, fn_init_keys) =
      fn_keys fn.node (resolve_constr_to_key (Some fn.id))
    in
      raise_entry_state obj_input_keys obj_init_keys fn.node.Ast.fn_body;
      raise_entry_state fn_input_keys fn_init_keys fn.node.Ast.fn_body;
      inner.Walk.visit_obj_fn_pre obj ident fn
  in

  let visit_obj_drop_pre obj b =
    let (obj_input_keys, obj_init_keys) =
      obj_keys obj.node (resolve_constr_to_key (Some obj.id))
    in
      raise_entry_state obj_input_keys obj_init_keys b;
      inner.Walk.visit_obj_drop_pre obj b
  in

  let visit_callable_pre id dst_slot_ids lv args =
    let defn_ty = lval_ty cx lv in
      begin
        match defn_ty with
            Ast.TY_fn (tsig,_) ->
              let formal_constrs = tsig.Ast.sig_input_constrs in
              let names = atoms_to_names args in
              let constrs =
                Array.map (apply_names_to_constr names) formal_constrs
              in
              let constr_keys =
                Array.map (resolve_constr_to_key None) constrs
              in
              let arg_init_keys =
                Array.concat
                  (Array.to_list
                     (Array.map
                        (fun arg ->
                           slot_inits (atom_slots cx arg))
                        args))
              in
                raise_pre_post_cond id arg_init_keys;
                raise_pre_post_cond id constr_keys
          | _ -> ()
      end;
      begin
        let postcond = slot_inits dst_slot_ids in
          raise_postcondition id postcond
      end
  in

  let raise_dst_init_precond_if_writing_through sid lval =
    match lval with
        Ast.LVAL_base _ -> ()
      | Ast.LVAL_ext _ ->
          let precond = slot_inits (lval_slots cx lval) in
            raise_precondition sid precond;
  in

  let visit_stmt_pre s =
    begin
      match s.node with
          Ast.STMT_check (constrs, _) ->
            let postcond = Array.map (resolve_constr_to_key None) constrs in
              raise_postcondition s.id postcond

        | Ast.STMT_recv (dst, src) ->
            let precond = slot_inits (lval_slots cx src) in
            let postcond = slot_inits (lval_slots cx dst) in
              raise_pre_post_cond s.id precond;
              raise_dst_init_precond_if_writing_through s.id dst;
              raise_postcondition s.id postcond

        | Ast.STMT_send (dst, src) ->
            let precond = Array.append
              (slot_inits (lval_slots cx dst))
              (slot_inits (lval_slots cx src))
            in
              raise_pre_post_cond s.id precond;

        | Ast.STMT_new_rec (dst, entries, base) ->
            let base_slots =
              begin
                match base with
                    None -> [| |]
                  | Some lval -> lval_slots cx lval
              end
            in
            let precond = slot_inits
              (Array.append (rec_inputs_slots cx entries) base_slots)
            in
            let postcond = slot_inits (lval_slots cx dst) in
              raise_dst_init_precond_if_writing_through s.id dst;
              raise_pre_post_cond s.id precond;
              raise_postcondition s.id postcond

        | Ast.STMT_new_tup (dst, modes_atoms) ->
            let precond = slot_inits
              (tup_inputs_slots cx modes_atoms)
            in
            let postcond = slot_inits (lval_slots cx dst) in
              raise_dst_init_precond_if_writing_through s.id dst;
              raise_pre_post_cond s.id precond;
              raise_postcondition s.id postcond

        | Ast.STMT_new_vec (dst, _, atoms) ->
            let precond = slot_inits (atoms_slots cx atoms) in
            let postcond = slot_inits (lval_slots cx dst) in
              raise_dst_init_precond_if_writing_through s.id dst;
              raise_pre_post_cond s.id precond;
              raise_postcondition s.id postcond

        | Ast.STMT_new_str (dst, _) ->
            let postcond = slot_inits (lval_slots cx dst) in
              raise_dst_init_precond_if_writing_through s.id dst;
              raise_postcondition s.id postcond

        | Ast.STMT_new_port dst ->
            let postcond = slot_inits (lval_slots cx dst) in
              raise_dst_init_precond_if_writing_through s.id dst;
              raise_postcondition s.id postcond

        | Ast.STMT_new_chan (dst, port) ->
            let precond = slot_inits (lval_option_slots cx port) in
            let postcond = slot_inits (lval_slots cx dst) in
              raise_dst_init_precond_if_writing_through s.id dst;
              raise_pre_post_cond s.id precond;
              raise_postcondition s.id postcond

        | Ast.STMT_new_box (dst, _, src) ->
            let precond = slot_inits (atom_slots cx src) in
            let postcond = slot_inits (lval_slots cx dst) in
              raise_dst_init_precond_if_writing_through s.id dst;
              raise_pre_post_cond s.id precond;
              raise_postcondition s.id postcond

        | Ast.STMT_copy (dst, src) ->
            let precond = slot_inits (expr_slots cx src) in
            let postcond = slot_inits (lval_slots cx dst) in
              raise_dst_init_precond_if_writing_through s.id dst;
              raise_pre_post_cond s.id precond;
              raise_postcondition s.id postcond

        | Ast.STMT_copy_binop (dst, _, src) ->
            let dst_init = slot_inits (lval_slots cx dst) in
            let src_init = slot_inits (atom_slots cx src) in
            let precond = Array.append dst_init src_init in
              raise_pre_post_cond s.id precond;

        | Ast.STMT_spawn (dst, _, _, lv, args)
        | Ast.STMT_call (dst, lv, args) ->
            raise_dst_init_precond_if_writing_through s.id dst;
            visit_callable_pre s.id (lval_slots cx dst) lv args

        | Ast.STMT_bind (dst, lv, args_opt) ->
            let args = arr_map_partial args_opt (fun a -> a) in
              raise_dst_init_precond_if_writing_through s.id dst;
              visit_callable_pre s.id (lval_slots cx dst) lv args

        | Ast.STMT_ret (Some at) ->
            let precond = slot_inits (atom_slots cx at) in
              raise_pre_post_cond s.id precond

        | Ast.STMT_put (Some at) ->
            let precond = slot_inits (atom_slots cx at) in
              raise_pre_post_cond s.id precond

        | Ast.STMT_join lval ->
            let precond = slot_inits (lval_slots cx lval) in
              raise_pre_post_cond s.id precond

        | Ast.STMT_log atom | Ast.STMT_log_err atom ->
            let precond = slot_inits (atom_slots cx atom) in
              raise_pre_post_cond s.id precond

        | Ast.STMT_check_expr expr ->
            let precond = slot_inits (expr_slots cx expr) in
              raise_pre_post_cond s.id precond
        | Ast.STMT_while sw ->
            let (_, expr) = sw.Ast.while_lval in
            let precond = slot_inits (expr_slots cx expr) in
              raise_precondition sw.Ast.while_body.id precond;
              raise_postcondition sw.Ast.while_body.id precond

        | Ast.STMT_alt_tag at ->
            let precond = slot_inits (lval_slots cx at.Ast.alt_tag_lval) in
            let visit_arm { node = (pat, block); id = _ } =
              (* FIXME (issue #34): propagate tag-carried constrs here. *)
              let rec get_slots pat =
                match pat with
                    Ast.PAT_slot header_slot -> [| header_slot |]
                  | Ast.PAT_tag (_, pats) ->
                      Array.concat (List.map get_slots (Array.to_list pats))
                  | _ -> [| |]
              in
              let header_slots = get_slots pat in
              let (input_keys, init_keys) =
                entry_keys header_slots [| |] (resolve_constr_to_key None)
              in
              raise_entry_state input_keys init_keys block
            in
            raise_pre_post_cond s.id precond;
            Array.iter visit_arm at.Ast.alt_tag_arms

        | Ast.STMT_for_each fe ->
            let (si, _) = fe.Ast.for_each_slot in
            let (callee, args) = fe.Ast.for_each_call in
              visit_callable_pre
                fe.Ast.for_each_body.id [| si.id |] callee args

        | Ast.STMT_for fo ->
            let (si, _) = fo.Ast.for_slot in
            let lval = fo.Ast.for_seq in
            let precond = slot_inits (lval_slots cx lval) in
            let block_entry_state = [| Constr_init si.id |] in
              raise_pre_post_cond s.id precond;
              raise_postcondition fo.Ast.for_body.id block_entry_state

        | _ -> ()
    end;
    inner.Walk.visit_stmt_pre s
  in
    { inner with
        Walk.visit_mod_item_pre = visit_mod_item_pre;
        Walk.visit_obj_fn_pre = visit_obj_fn_pre;
        Walk.visit_obj_drop_pre = visit_obj_drop_pre;
        Walk.visit_stmt_pre = visit_stmt_pre }
;;

let lset_add (x:node_id) (xs:node_id list) : node_id list =
  if List.mem x xs
  then xs
  else x::xs
;;

let lset_remove (x:node_id) (xs:node_id list) : node_id list =
  List.filter (fun a -> not (a = x)) xs
;;

let lset_union (xs:node_id list) (ys:node_id list) : node_id list =
  List.fold_left (fun ns n -> lset_add n ns) xs ys
;;

let lset_diff (xs:node_id list) (ys:node_id list) : node_id list =
  List.fold_left (fun ns n -> lset_remove n ns) xs ys
;;

let lset_fmt lset =
  "[" ^
    (String.concat ", "
       (List.map
          (fun n -> string_of_int (int_of_node n)) lset)) ^
    "]"
;;

let show_node cx graph s i =
  iflog cx
    (fun _ ->
       log cx "node '%s' = %d -> %s"
         s (int_of_node i) (lset_fmt (Hashtbl.find graph i)))
;;

let add_flow_edges (graph:node_graph) (n:node_id) (dsts:node_id list) : unit =
  if Hashtbl.mem graph n
  then
    let existing = Hashtbl.find graph n in
      Hashtbl.replace graph n (lset_union existing dsts)
  else
    Hashtbl.add graph n dsts
;;

let rec build_flow_graph_for_stmt
    (graph:node_graph)
    (predecessors:node_id list)
    (s:Ast.stmt)
    : node_id list =

  let connect ps qs =
    List.iter
      (fun pred -> add_flow_edges graph pred qs)
      ps
  in

  let seq ps (ss:Ast.stmt array) =
    build_flow_graph_for_stmts graph ps ss
  in

  let blk ps b =
    connect ps [b.id];
    seq [b.id] b.node
  in

  let first ss =
    if Array.length ss = 0
    then []
    else [ss.(0).id]
  in

    connect [s.id] [];
    let outs =
      match s.node with

        | Ast.STMT_while sw ->
            let (pre_loop_stmts, _) = sw.Ast.while_lval in
            let body = sw.Ast.while_body in
            let preloop_end = seq [s.id] pre_loop_stmts in
              connect predecessors [s.id];
              connect (blk preloop_end body) (first pre_loop_stmts);
              preloop_end

        | Ast.STMT_for sf ->
            let body_end = blk [s.id] sf.Ast.for_body in
              connect predecessors [s.id];
              connect body_end (first sf.Ast.for_body.node);
              body_end

        | Ast.STMT_for_each sfe ->
            let head_end = blk [s.id] sfe.Ast.for_each_head in
            let body_end = blk head_end sfe.Ast.for_each_body in
              connect predecessors [s.id];
              connect body_end (first sfe.Ast.for_each_head.node);
              body_end

        | Ast.STMT_if sif ->
            connect predecessors [s.id];
            (blk [s.id] sif.Ast.if_then) @
              (match sif.Ast.if_else with
                   None -> [s.id]
                 | Some els -> blk [s.id] els)

        | Ast.STMT_alt_tag sat ->
            connect predecessors [s.id];
            Array.fold_left
              (fun ends {node=(_, b); id=_} -> (blk [s.id] b) @ ends)
              [] sat.Ast.alt_tag_arms

        | Ast.STMT_block b ->
            blk predecessors b

        | Ast.STMT_fail
        | Ast.STMT_ret _ ->
            connect predecessors [s.id];
            []

        | _ ->
            connect predecessors [s.id];
            [s.id]
    in
      connect outs [];
      outs

and build_flow_graph_for_stmts
    (graph:node_graph)
    (predecessors:node_id list)
    (ss:Ast.stmt array)
    : node_id list =
  Array.fold_left (build_flow_graph_for_stmt graph) predecessors ss
;;


let graph_building_visitor
    (cx:ctxt)
    (tables_stack:typestate_tables Stack.t)
    (inner:Walk.visitor)
    : Walk.visitor =

  let tables _ = Stack.top tables_stack in
  let graph _ = (tables()).ts_graph in
  let blk b =
    add_flow_edges (graph()) b.id [];
    ignore (build_flow_graph_for_stmts (graph()) [b.id] b.node)
  in

  let visit_mod_item_pre n p i =
    begin
      match i.node.Ast.decl_item with
          Ast.MOD_ITEM_fn fn -> blk fn.Ast.fn_body
        | _ -> ()
    end;
    inner.Walk.visit_mod_item_pre n p i
  in

  let visit_obj_fn_pre obj ident fn =
    blk fn.node.Ast.fn_body;
    inner.Walk.visit_obj_fn_pre obj ident fn
  in

  let visit_obj_drop_pre obj b =
    blk b;
    inner.Walk.visit_obj_drop_pre obj b
  in

  let visit_block_pre b =
    if Hashtbl.mem cx.ctxt_block_is_loop_body b.id
    then blk b;
    inner.Walk.visit_block_pre b
  in

    { inner with
        Walk.visit_mod_item_pre = visit_mod_item_pre;
        Walk.visit_obj_fn_pre = visit_obj_fn_pre;
        Walk.visit_obj_drop_pre = visit_obj_drop_pre;
        Walk.visit_block_pre = visit_block_pre }

;;

let find_roots
    (cx:ctxt)
    (graph:(node_id, (node_id list)) Hashtbl.t)
    : (node_id,unit) Hashtbl.t =
  let roots = Hashtbl.create 0 in
    Hashtbl.iter (fun src _ -> Hashtbl.replace roots src ()) graph;
    Hashtbl.iter (fun _ dsts ->
                    List.iter (fun d -> Hashtbl.remove roots d) dsts) graph;
    iflog cx
      (fun _ -> Hashtbl.iter
         (fun k _ -> log cx "root: %d" (int_of_node k)) roots);
    roots
;;

let run_dataflow (cx:ctxt) (ts:typestate_tables) : unit =
  let graph = ts.ts_graph in
  let idref = ts.ts_maxid in
  let roots = find_roots cx graph in
  let nodes = Queue.create () in

  let progress = ref true in
  let iter = ref 0 in
  let total = ref 0 in
  let written = Hashtbl.create 0 in
  let scheduled = Hashtbl.create 0 in
  let next_nodes = Queue.create () in
  let schedule n =
    if Hashtbl.mem scheduled n
    then ()
    else
      begin
        Queue.push n next_nodes;
        Hashtbl.add scheduled n ()
      end
  in

  let fmt_constr_bitv bitv =
    String.concat ", "
      (List.map
         (fun i ->
            fmt_constr_key cx
              (Hashtbl.find ts.ts_constrs (Constr i)))
         (Bits.to_list bitv))
  in

  let set_bits dst src =
    if Bits.copy dst src
    then (progress := true;
          iflog cx (fun _ -> log cx "made progress setting bits"))
  in

  let intersect_bits node dst src =
    if Bits.intersect dst src
    then (progress := true;
          schedule node;
          iflog cx (fun _ -> log cx
                      "made progress intersecting bits"))
  in

  let tmp_diff = (Bits.create (!idref) false) in
  let tmp_poststate = (Bits.create (!idref) false) in
    Hashtbl.iter (fun n _ -> schedule n) roots;
    while !progress do
      incr iter;
      progress := false;
      Queue.clear nodes;
      Queue.transfer next_nodes nodes;
      Hashtbl.clear scheduled;
      iflog cx (fun _ ->
                  log cx "";
                  log cx "--------------------";
                  log cx "dataflow pass %d" (!iter));
      Queue.iter
        begin
          fun node ->
            let prestate = Hashtbl.find ts.ts_prestates node in
            let precond = Hashtbl.find ts.ts_preconditions node in
            let postcond = Hashtbl.find ts.ts_postconditions node in
            let poststate = Hashtbl.find ts.ts_poststates node in

              incr total;
              Bits.clear tmp_poststate;
              ignore (Bits.union tmp_poststate prestate);
              ignore (Bits.union tmp_poststate precond);
              ignore (Bits.union tmp_poststate postcond);

              ignore (Bits.copy tmp_diff precond);
              ignore (Bits.difference tmp_diff postcond);
              ignore (Bits.difference tmp_poststate tmp_diff);

              iflog cx
                begin
                  fun _ ->
                    log cx "stmt %d: '%s'" (int_of_node node)
                      begin
                        match htab_search cx.ctxt_all_stmts node with
                            None ->
                              begin
                                match htab_search cx.ctxt_all_blocks node with
                                    None -> "??"
                                  | Some b ->
                                      Fmt.fmt_to_str Ast.fmt_block b
                              end
                          | Some stmt -> Fmt.fmt_to_str Ast.fmt_stmt stmt
                      end;
                    log cx "stmt %d:" (int_of_node node);

                    log cx "    prestate %s" (fmt_constr_bitv prestate);
                    log cx "    precond %s" (fmt_constr_bitv precond);
                    log cx "    postcond %s" (fmt_constr_bitv postcond);
                    log cx "    poststate %s" (fmt_constr_bitv poststate);
                    log cx
                      "    precond - postcond %s" (fmt_constr_bitv tmp_diff);
                    log cx
                      "    new poststate %s" (fmt_constr_bitv tmp_poststate)
                end;

              set_bits poststate tmp_poststate;

              Hashtbl.replace written node ();
              let successors = Hashtbl.find graph node in
              let i = int_of_node node in
                iflog cx (fun _ -> log cx
                            "out-edges for %d: %s" i (lset_fmt successors));
                List.iter
                  begin
                    fun succ ->
                      let succ_prestates =
                        Hashtbl.find ts.ts_prestates succ
                      in
                        if Hashtbl.mem written succ
                        then
                          intersect_bits succ succ_prestates poststate
                        else
                          begin
                            progress := true;
                            schedule succ;
                            set_bits succ_prestates poststate
                          end
                  end
                  successors
        end
        nodes
    done
;;

let dataflow_visitor
    (cx:ctxt)
    (tables_stack:typestate_tables Stack.t)
    (inner:Walk.visitor)
    : Walk.visitor =

  let tables _ = Stack.top tables_stack in

  let visit_mod_item_pre n p i =
    run_dataflow cx (tables());
    inner.Walk.visit_mod_item_pre n p i
  in

  let visit_obj_fn_pre obj ident fn =
    run_dataflow cx (tables());
    inner.Walk.visit_obj_fn_pre obj ident fn
  in

  let visit_obj_drop_pre obj b =
    run_dataflow cx (tables());
    inner.Walk.visit_obj_drop_pre obj b
  in

  let visit_block_pre b =
    if Hashtbl.mem cx.ctxt_block_is_loop_body b.id
    then run_dataflow cx (tables());
    inner.Walk.visit_block_pre b
  in

    { inner with
        Walk.visit_mod_item_pre = visit_mod_item_pre;
        Walk.visit_obj_fn_pre = visit_obj_fn_pre;
        Walk.visit_obj_drop_pre = visit_obj_drop_pre;
        Walk.visit_block_pre = visit_block_pre }
;;


let typestate_verify_visitor
    (cx:ctxt)
    (tables_stack:typestate_tables Stack.t)
    (inner:Walk.visitor)
    : Walk.visitor =

  let tables _ = Stack.top tables_stack in

  let check_states id =
    let ts = tables () in
    let prestate = Hashtbl.find ts.ts_prestates id in
    let precond = Hashtbl.find ts.ts_preconditions id in
      List.iter
        (fun i ->
           if not (Bits.get prestate i)
           then
             let ckey = Hashtbl.find ts.ts_constrs (Constr i) in
             let constr_str = fmt_constr_key cx ckey in
               err (Some id)
                 "Unsatisfied precondition constraint %s"
                 constr_str)
        (Bits.to_list precond)
  in

  let visit_stmt_pre s =
    check_states s.id;
    inner.Walk.visit_stmt_pre s
  in

  let visit_block_pre b =
    check_states b.id;
    inner.Walk.visit_block_pre b
  in

    { inner with
        Walk.visit_stmt_pre = visit_stmt_pre;
        Walk.visit_block_pre = visit_block_pre }
;;


let lifecycle_visitor
    (cx:ctxt)
    (tables_stack:typestate_tables Stack.t)
    (inner:Walk.visitor)
    : Walk.visitor =

  (*
   * This visitor doesn't *calculate* part of the typestate; it uses
   * the typestates calculated in earlier passes to extract "summaries"
   * of slot-lifecycle events into the ctxt tables
   * ctxt_copy_stmt_is_init and ctxt_post_stmt_slot_drops. These are
   * used later on in translation.
   *)

  let tables _ = Stack.top tables_stack in

  let (live_block_slots:(node_id, unit) Hashtbl.t) = Hashtbl.create 0 in
  let (frame_blocks:frame_block_slots_stack) = Stack.create () in

  let (implicit_init_block_slots:(node_id,node_id list) Hashtbl.t) =
    Hashtbl.create 0
  in

  let push_slot sl =
    Stack.push sl (Stack.top (Stack.top frame_blocks))
  in

  let mark_slot_live sl =
    Hashtbl.replace live_block_slots sl ()
  in


  let visit_block_pre b =

    let s = Stack.create() in
      begin
        match Stack.top loop_blocks with
            Some loop -> Stack.push s loop
          | None -> ()
      end;
      Stack.push s (Stack.top frame_blocks);
      begin
        match htab_search implicit_init_block_slots b.id with
            None -> ()
          | Some slots ->
            List.iter
              (fun slot ->
                 push_slot slot;
                 mark_slot_live slot)
              slots
      end;
      inner.Walk.visit_block_pre b
  in

  let note_stmt_drops stmt slots =
    iflog cx
      begin
        fun _ ->
          log cx "implicit drop of %d slots after stmt %a: "
            (List.length slots)
            Ast.sprintf_stmt stmt;
          List.iter (fun s -> log cx "drop: %a"
                       Ast.sprintf_slot_key
                       (Hashtbl.find cx.ctxt_slot_keys s))
            slots
      end;
    htab_put cx.ctxt_post_stmt_slot_drops stmt.id slots
  in

  let note_block_drops bid slots =
    iflog cx
      begin
        fun _ ->
          log cx "implicit drop of %d slots after block %d: "
            (List.length slots)
            (int_of_node bid);
          List.iter (fun s -> log cx "drop: %a"
                       Ast.sprintf_slot_key
                       (Hashtbl.find cx.ctxt_slot_keys s))
            slots
      end;
    htab_put cx.ctxt_post_block_slot_drops bid slots
  in

  let filter_live_block_slots slots =
    List.filter (fun i -> Hashtbl.mem live_block_slots i) slots
  in

  let visit_block_post b =
    inner.Walk.visit_block_post b;
    begin
      match Stack.top loop_blocks with
          Some loop ->
            ignore (Stack.pop loop);
            if Stack.is_empty loop
            then ignore (Stack.pop loop_blocks);
        | None -> ()
    end;
    let block_slots = Stack.pop (Stack.top frame_blocks) in
      (* The blk_slots stack we have has accumulated slots in
       * declaration order as we walked the block; the top of the
       * stack is the last-declared slot. We want to generate
       * slot-drop obligations here for the slots in top-down order
       * (starting with the last-declared) but only hitting those
       * slots that actually got initialized (went live) at some
       * point in the block.
       *)
    let slots = stk_elts_from_top block_slots in
    let live = filter_live_block_slots slots in
      note_block_drops b.id live
  in

  let visit_stmt_pre s =
    begin
      let mark_lval_live lv_dst =
        let dst_slots = lval_slots cx lv_dst in
          Array.iter mark_slot_live dst_slots;
      in
        match s.node with
            Ast.STMT_copy (lv_dst, _)
          | Ast.STMT_call (lv_dst, _, _)
          | Ast.STMT_spawn (lv_dst, _, _, _, _)
          | Ast.STMT_recv (lv_dst, _)
          | Ast.STMT_bind (lv_dst, _, _)
          | Ast.STMT_new_rec (lv_dst, _, _)
          | Ast.STMT_new_tup (lv_dst, _)
          | Ast.STMT_new_vec (lv_dst, _, _)
          | Ast.STMT_new_str (lv_dst, _)
          | Ast.STMT_new_port lv_dst
          | Ast.STMT_new_chan (lv_dst, _)
          | Ast.STMT_new_box (lv_dst, _, _) ->
              let ts = tables () in
              let prestate = Hashtbl.find ts.ts_prestates s.id in
              let poststate = Hashtbl.find ts.ts_poststates s.id in
              let dst_slots = lval_slots cx lv_dst in
              let is_initializing slot =
                let cid =
                  Hashtbl.find ts.ts_constr_ids (Constr_init slot)
                in
                let i = int_of_constr cid in
                  (not (Bits.get prestate i)) && (Bits.get poststate i)
              in
              let initializing =
                List.exists is_initializing (Array.to_list dst_slots)
              in
                if initializing
                then
                  begin
                    iflog cx
                      begin
                        fun _ ->
                          log cx "noting lval %a init at stmt %a"
                            Ast.sprintf_lval lv_dst Ast.sprintf_stmt s
                      end;
                    Hashtbl.replace cx.ctxt_stmt_is_init s.id ();
                    mark_lval_live lv_dst
                  end;

          | Ast.STMT_decl (Ast.DECL_slot (_, sloti)) ->
              push_slot sloti.id

          | Ast.STMT_for f ->
              log cx "noting implicit init for slot %d in for-block %d"
                (int_of_node (fst f.Ast.for_slot).id)
                (int_of_node (f.Ast.for_body.id));
              Hashtbl.replace cx.ctxt_stmt_is_init s.id ();
              htab_put implicit_init_block_slots
                f.Ast.for_body.id
                [ (fst f.Ast.for_slot).id ]

          | Ast.STMT_for_each f ->
              log cx "noting implicit init for slot %d in for_each-block %d"
                (int_of_node (fst f.Ast.for_each_slot).id)
                (int_of_node (f.Ast.for_each_body.id));
              Hashtbl.replace cx.ctxt_stmt_is_init s.id ();
              htab_put implicit_init_block_slots
                f.Ast.for_each_body.id
                [ (fst f.Ast.for_each_slot).id ]

          | Ast.STMT_while sw ->
              (* Collect any header-locals. *)
              Array.iter
                begin
                  fun stmt ->
                    match stmt.node with
                        Ast.STMT_decl (Ast.DECL_slot (_, slot)) ->
                          begin
                            match
                              htab_search cx.ctxt_while_header_slots s.id
                            with
                                None ->
                                  Hashtbl.add cx.ctxt_while_header_slots
                                    s.id [slot.id]
                              | Some slots ->
                                  Hashtbl.replace cx.ctxt_while_header_slots
                                    s.id (slot.id :: slots)
                          end
                      | _ -> ()
                end
                (fst sw.Ast.while_lval);

              iflog cx (fun _ -> log cx "entering a loop");
              Stack.push (Some (Stack.create ()))  loop_blocks;

          | Ast.STMT_alt_tag { Ast.alt_tag_arms = arms;
                               Ast.alt_tag_lval = _ } ->
              let note_slot block slot_id =
                log cx
                  "noting implicit init for slot %d in pattern-alt block %d"
                  (int_of_node slot_id)
                  (int_of_node block.id);
              in
              let rec all_pat_slot_ids block pat =
                match pat with
                    Ast.PAT_slot ({ id = slot_id; node = _ }, _) ->
                      [ slot_id ]
                  | Ast.PAT_tag (_, pats) ->
                      List.concat
                        (Array.to_list
                           (Array.map (all_pat_slot_ids block) pats))
                  | Ast.PAT_lit _
                  | Ast.PAT_wild -> []
              in
                Array.iter
                  begin
                    fun { node = (pat, block); id = _ } ->
                      let slot_ids = all_pat_slot_ids block pat in
                        List.iter (note_slot block) slot_ids;
                        htab_put implicit_init_block_slots
                          block.id
                          slot_ids
                  end
                  arms
          | _ -> ()
    end;
    inner.Walk.visit_stmt_pre s
  in

  let visit_stmt_post s =
    inner.Walk.visit_stmt_post s;

    let handle_outward_jump_stmt block_stack =
      let blocks = stk_elts_from_top block_stack in
          let slots = List.concat (List.map stk_elts_from_top blocks) in
          let live = filter_live_block_slots slots in
            note_stmt_drops s live
    in

      match s.node with
          Ast.STMT_ret _
        | Ast.STMT_be _ ->
            handle_outward_jump_stmt (Stack.top frame_blocks)

        | Ast.STMT_break ->
            begin
              match (Stack.top loop_blocks) with
                  Some loop -> handle_outward_jump_stmt loop
                | None ->
                    err (Some s.id) "break statement outside of a loop"
            end
        | _ -> ()
  in

  let enter_frame _ =
    Stack.push (Stack.create()) frame_blocks;
    Stack.push None loop_blocks
  in

  let leave_frame _ =
    ignore (Stack.pop frame_blocks);
    match Stack.pop loop_blocks with
        Some _ -> bug () "leave_frame should not end a loop"
      | None -> ()
  in

  let visit_mod_item_pre n p i =
    enter_frame();
    inner.Walk.visit_mod_item_pre n p i
  in

  let visit_mod_item_post n p i =
    inner.Walk.visit_mod_item_post n p i;
    leave_frame()
  in

  let visit_obj_fn_pre obj ident fn =
    enter_frame();
    inner.Walk.visit_obj_fn_pre obj ident fn
  in

  let visit_obj_fn_post obj ident fn =
    inner.Walk.visit_obj_fn_post obj ident fn;
    leave_frame()
  in

  let visit_obj_drop_pre obj b =
    enter_frame();
    inner.Walk.visit_obj_drop_pre obj b
  in

  let visit_obj_drop_post obj b =
    inner.Walk.visit_obj_drop_post obj b;
    leave_frame()
  in

    { inner with
        Walk.visit_block_pre = visit_block_pre;
        Walk.visit_block_post = visit_block_post;
        Walk.visit_stmt_pre = visit_stmt_pre;
        Walk.visit_stmt_post = visit_stmt_post;

        Walk.visit_mod_item_pre = visit_mod_item_pre;
        Walk.visit_mod_item_post = visit_mod_item_post;
        Walk.visit_obj_fn_pre = visit_obj_fn_pre;
        Walk.visit_obj_fn_post = visit_obj_fn_post;
        Walk.visit_obj_drop_pre = visit_obj_drop_pre;
        Walk.visit_obj_drop_post = visit_obj_drop_post;

    }
;;

let process_crate
    (cx:ctxt)
    (crate:Ast.crate)
    : unit =
  let (scopes:(scope list) ref) = ref [] in
  let (tables_stack:typestate_tables Stack.t) = Stack.create () in
  let (all_tables:item_tables) = Hashtbl.create 0 in
  let table_managed = tables_managing_visitor all_tables tables_stack in
  let setup_passes =
    [|
      (table_managed
         (scope_stack_managing_visitor scopes
            (constr_id_assigning_visitor cx tables_stack scopes
               Walk.empty_visitor)));
      (table_managed
         (bitmap_assigning_visitor cx tables_stack
            Walk.empty_visitor));
      (table_managed
         (scope_stack_managing_visitor scopes
            (condition_assigning_visitor cx tables_stack scopes
               Walk.empty_visitor)));
      (table_managed
         (graph_building_visitor cx tables_stack
            Walk.empty_visitor));
    |]
  in
  let dataflow_passes =
    [|
      (table_managed
         (dataflow_visitor cx tables_stack
            Walk.empty_visitor))
    |]
  in
  let verify_passes =
    [|
      (table_managed
         (typestate_verify_visitor cx tables_stack
            Walk.empty_visitor))
    |]
  in
  let aux_passes =
    [|
      (table_managed
         (lifecycle_visitor cx tables_stack
            Walk.empty_visitor))
    |]
  in
  let log_flag = cx.ctxt_sess.Session.sess_log_typestate in
    run_passes cx "typestate setup" setup_passes log_flag log crate;
    run_passes cx
      "typestate dataflow" dataflow_passes log_flag log crate;
    run_passes cx "typestate verify" verify_passes log_flag log crate;
    run_passes cx "typestate aux" aux_passes log_flag log crate
;;


(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
