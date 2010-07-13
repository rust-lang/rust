open Semant;;
open Common;;


let log cx = Session.log "typestate"
  cx.ctxt_sess.Session.sess_log_typestate
  cx.ctxt_sess.Session.sess_log_out
;;

let iflog cx thunk =
  if cx.ctxt_sess.Session.sess_log_typestate
  then thunk ()
  else ()
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
        Some (_, cid) ->
          if referent_is_item cx cid
          then
            begin
              match Hashtbl.find cx.ctxt_all_item_types cid with
                  Ast.TY_fn (_, taux) ->
                    begin
                      if taux.Ast.fn_effect = Ast.PURE
                      then cid
                      else err (Some cid) "impure function used in constraint"
                    end
                | _ -> bug () "bad type of predicate"
            end
          else
            bug () "slot used as predicate"
      | None -> bug () "predicate not found"
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
                        None -> bug () "constraint-arg not found"
                      | Some (_, aid) ->
                          if referent_is_slot cx aid
                          then
                            if type_has_state
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
                        if referent_is_slot cx id
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

let constr_id_assigning_visitor
    (cx:ctxt)
    (scopes:(scope list) ref)
    (idref:int ref)
    (inner:Walk.visitor)
    : Walk.visitor =

  let resolve_constr_to_key
      (formal_base:node_id)
      (constr:Ast.constr)
      : constr_key =
    determine_constr_key cx (!scopes) (Some formal_base) constr
  in

  let note_constr_key key =
    if not (Hashtbl.mem cx.ctxt_constr_ids key)
    then
      begin
        let cid = Constr (!idref) in
          iflog cx
            (fun _ -> log cx "assigning constr id #%d to constr %s"
               (!idref) (fmt_constr_key cx key));
          incr idref;
          htab_put cx.ctxt_constrs cid key;
          htab_put cx.ctxt_constr_ids key cid;
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
            let referent = lval_to_referent cx (lval_base_id lv) in
            let referent_ty = lval_ty cx lv in
              begin
                match referent_ty with
                    Ast.TY_fn (tsig,_) ->
                      let constrs = tsig.Ast.sig_input_constrs in
                      let names = atoms_to_names args in
                      let constrs' =
                        Array.map (apply_names_to_constr names) constrs
                      in
                        Array.iter (visit_constr_pre (Some referent)) constrs'

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
        Walk.visit_slot_identified_pre = visit_slot_identified_pre;
        Walk.visit_stmt_pre = visit_stmt_pre;
        Walk.visit_constr_pre = visit_constr_pre }
;;

let bitmap_assigning_visitor
    (cx:ctxt)
    (idref:int ref)
    (inner:Walk.visitor)
    : Walk.visitor =
  let visit_stmt_pre s =
    iflog cx (fun _ -> log cx "building %d-entry bitmap for node %d"
                (!idref) (int_of_node s.id));
    htab_put cx.ctxt_preconditions s.id (Bits.create (!idref) false);
    htab_put cx.ctxt_postconditions s.id (Bits.create (!idref) false);
    htab_put cx.ctxt_prestates s.id (Bits.create (!idref) false);
    htab_put cx.ctxt_poststates s.id (Bits.create (!idref) false);
    inner.Walk.visit_stmt_pre s
  in
  let visit_block_pre b =
    iflog cx (fun _ -> log cx "building %d-entry bitmap for node %d"
                (!idref) (int_of_node b.id));
    htab_put cx.ctxt_preconditions b.id (Bits.create (!idref) false);
    htab_put cx.ctxt_postconditions b.id (Bits.create (!idref) false);
    htab_put cx.ctxt_prestates b.id (Bits.create (!idref) false);
    htab_put cx.ctxt_poststates b.id (Bits.create (!idref) false);
    inner.Walk.visit_block_pre b
  in
    { inner with
        Walk.visit_stmt_pre = visit_stmt_pre;
        Walk.visit_block_pre = visit_block_pre }
;;

let condition_assigning_visitor
    (cx:ctxt)
    (scopes:(scope list) ref)
    (inner:Walk.visitor)
    : Walk.visitor =

  let raise_bits (bitv:Bits.t) (keys:constr_key array) : unit =
    Array.iter
      (fun key ->
         let cid = Hashtbl.find cx.ctxt_constr_ids key in
         let i = int_of_constr cid in
           iflog cx (fun _ -> log cx "setting bit %d, constraint %s"
                       i (fmt_constr_key cx key));
           Bits.set bitv (int_of_constr cid) true)
      keys
  in

  let slot_inits ss = Array.map (fun s -> Constr_init s) ss in

  let raise_postcondition (id:node_id) (keys:constr_key array) : unit =
    let bitv = Hashtbl.find cx.ctxt_postconditions id in
      raise_bits bitv keys
  in

  let raise_precondition (id:node_id) (keys:constr_key array) : unit =
    let bitv = Hashtbl.find cx.ctxt_preconditions id in
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
    let referent_ty = lval_ty cx lv in
      begin
        match referent_ty with
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
              raise_postcondition s.id postcond

        | Ast.STMT_send (dst, src) ->
            let precond = Array.append
              (slot_inits (lval_slots cx dst))
              (slot_inits (lval_slots cx src))
            in
              raise_pre_post_cond s.id precond;

        | Ast.STMT_init_rec (dst, entries, base) ->
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
              raise_pre_post_cond s.id precond;
              raise_postcondition s.id postcond

        | Ast.STMT_init_tup (dst, modes_atoms) ->
            let precond = slot_inits
              (tup_inputs_slots cx modes_atoms)
            in
            let postcond = slot_inits (lval_slots cx dst) in
              raise_pre_post_cond s.id precond;
              raise_postcondition s.id postcond

        | Ast.STMT_init_vec (dst, atoms) ->
            let precond = slot_inits (atoms_slots cx atoms) in
            let postcond = slot_inits (lval_slots cx dst) in
              raise_pre_post_cond s.id precond;
              raise_postcondition s.id postcond

        | Ast.STMT_init_str (dst, _) ->
            let postcond = slot_inits (lval_slots cx dst) in
              raise_postcondition s.id postcond

        | Ast.STMT_init_port dst ->
            let postcond = slot_inits (lval_slots cx dst) in
              raise_postcondition s.id postcond

        | Ast.STMT_init_chan (dst, port) ->
            let precond = slot_inits (lval_option_slots cx port) in
            let postcond = slot_inits (lval_slots cx dst) in
              raise_pre_post_cond s.id precond;
              raise_postcondition s.id postcond

        | Ast.STMT_init_box (dst, src) ->
            let precond = slot_inits (atom_slots cx src) in
            let postcond = slot_inits (lval_slots cx dst) in
              raise_pre_post_cond s.id precond;
              raise_postcondition s.id postcond

        | Ast.STMT_copy (dst, src) ->
            let precond = slot_inits (expr_slots cx src) in
            let postcond = slot_inits (lval_slots cx dst) in
              raise_pre_post_cond s.id precond;
              raise_postcondition s.id postcond

        | Ast.STMT_copy_binop (dst, _, src) ->
            let dst_init = slot_inits (lval_slots cx dst) in
            let src_init = slot_inits (atom_slots cx src) in
            let precond = Array.append dst_init src_init in
              raise_pre_post_cond s.id precond;

        | Ast.STMT_spawn (dst, _, lv, args)
        | Ast.STMT_call (dst, lv, args) ->
            visit_callable_pre s.id (lval_slots cx dst) lv args

        | Ast.STMT_bind (dst, lv, args_opt) ->
            let args = arr_map_partial args_opt (fun a -> a) in
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

        | Ast.STMT_log atom ->
            let precond = slot_inits (atom_slots cx atom) in
              raise_pre_post_cond s.id precond

        | Ast.STMT_check_expr expr ->
            let precond = slot_inits (expr_slots cx expr) in
              raise_pre_post_cond s.id precond

        | Ast.STMT_while sw ->
            let (_, expr) = sw.Ast.while_lval in
            let precond = slot_inits (expr_slots cx expr) in
              raise_pre_post_cond s.id precond

        | Ast.STMT_alt_tag at ->
            let precond = slot_inits (lval_slots cx at.Ast.alt_tag_lval) in
            let visit_arm { node = (pat, block) } =
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

type node_graph = (node_id, (node_id list)) Hashtbl.t;;
type sibling_map = (node_id, node_id) Hashtbl.t;;

let graph_sequence_building_visitor
    (cx:ctxt)
    (graph:node_graph)
    (sibs:sibling_map)
    (inner:Walk.visitor)
    : Walk.visitor =

  (* Flow each stmt to its sequence-successor. *)
  let visit_stmts stmts =
    let len = Array.length stmts in
      for i = 0 to len - 2
      do
        let stmt = stmts.(i) in
        let next = stmts.(i+1) in
          log cx "sequential stmt edge %d -> %d"
            (int_of_node stmt.id) (int_of_node next.id);
          htab_put graph stmt.id [next.id];
          htab_put sibs stmt.id next.id;
      done;
      (* Flow last node to nowhere. *)
      if len > 0
      then htab_put graph stmts.(len-1).id []
  in

  let visit_stmt_pre s =
    (* Sequence the prelude nodes on special stmts. *)
    begin
      match s.node with
          Ast.STMT_while sw ->
            let (stmts, _) = sw.Ast.while_lval in
              visit_stmts stmts
        | _ -> ()
    end;
    inner.Walk.visit_stmt_pre s
  in

  let visit_block_pre b =
    visit_stmts b.node;
    inner.Walk.visit_block_pre b
  in

    { inner with
        Walk.visit_stmt_pre = visit_stmt_pre;
        Walk.visit_block_pre = visit_block_pre }
;;

let add_flow_edges (graph:node_graph) (n:node_id) (dsts:node_id list) : unit =
  let existing = Hashtbl.find graph n in
    Hashtbl.replace graph n (lset_union existing dsts)
;;

let remove_flow_edges
    (graph:node_graph)
    (n:node_id)
    (dsts:node_id list)
    : unit =
  let existing = Hashtbl.find graph n in
    Hashtbl.replace graph n (lset_diff existing dsts)
;;


let last_id (nodes:('a identified) array) : node_id =
  let len = Array.length nodes in
    nodes.(len-1).id
;;

let last_id_or_block_id (block:Ast.block) : node_id =
  let len = Array.length block.node in
    if len = 0
    then block.id
    else last_id block.node
;;

let graph_general_block_structure_building_visitor
    (cx:ctxt)
    (graph:node_graph)
    (sibs:sibling_map)
    (inner:Walk.visitor)
    : Walk.visitor =

  let stmts = Stack.create () in

  let visit_stmt_pre s =
    Stack.push s stmts;
    inner.Walk.visit_stmt_pre s
  in

  let visit_stmt_post s =
    inner.Walk.visit_stmt_post s;
    ignore (Stack.pop stmts)
  in

  let show_node = show_node cx graph in

  let visit_block_pre b =
    begin
      let len = Array.length b.node in
      let _ = htab_put graph b.id
        (if len > 0 then [b.node.(0).id] else [])
      in

      (*
       * If block has len, 
       * then flow block to block.node.(0) and block.node.(len-1) to dsts
       * else flow block to dsts
       * 
       * so AST:
       * 
       *   block#n{ stmt#0 ... stmt#k };
       *   stmt#j;
       * 
       * turns into graph:
       * 
       *   block#n -> stmt#0 -> ... -> stmt#k -> stmt#j
       * 
       *)
        if Stack.is_empty stmts
        then ()
        else
          let s = Stack.top stmts in
            add_flow_edges graph s.id [b.id];
            match htab_search sibs s.id with
                None -> ()
              | Some sib_id ->
                  if len > 0
                  then
                    add_flow_edges graph (last_id b.node) [sib_id]
                  else
                    add_flow_edges graph b.id [sib_id]
    end;
    show_node "block" b.id;
    inner.Walk.visit_block_pre b
  in

    { inner with
        Walk.visit_stmt_pre = visit_stmt_pre;
        Walk.visit_stmt_post = visit_stmt_post;
        Walk.visit_block_pre = visit_block_pre }
;;


let graph_special_block_structure_building_visitor
    (cx:ctxt)
    (graph:(node_id, (node_id list)) Hashtbl.t)
    (inner:Walk.visitor)
    : Walk.visitor =

  let visit_stmt_pre s =
    begin
      match s.node with

        | Ast.STMT_if sif ->
            let cond_id = s.id in
            let then_id = sif.Ast.if_then.id in
            let then_end_id = last_id_or_block_id sif.Ast.if_then in
            let show_node = show_node cx graph in
              show_node "initial cond" cond_id;
              show_node "initial then" then_id;
              show_node "initial then_end" then_end_id;
              begin
                match sif.Ast.if_else with
                    None ->
                      let succ = Hashtbl.find graph then_end_id in
                        Hashtbl.replace graph cond_id (then_id :: succ);
                        (* Kill residual messed-up block wiring.*)
                        remove_flow_edges graph then_end_id [then_id];
                        show_node "cond" cond_id;
                        show_node "then" then_id;
                        show_node "then_end" then_end_id;

                  | Some e ->
                      let else_id = e.id in
                      let else_end_id = last_id_or_block_id e in
                      let succ = Hashtbl.find graph else_end_id in
                        show_node "initial else" else_id;
                        show_node "initial else_end" else_end_id;
                        Hashtbl.replace graph cond_id [then_id; else_id];
                        Hashtbl.replace graph then_end_id succ;
                        Hashtbl.replace graph else_end_id succ;
                        (* Kill residual messed-up block wiring.*)
                        remove_flow_edges graph then_end_id [then_id];
                        remove_flow_edges graph else_id [then_id];
                        remove_flow_edges graph else_end_id [then_id];
                        show_node "cond" cond_id;
                        show_node "then" then_id;
                        show_node "then_end" then_end_id;
                        show_node "else" else_id;
                        show_node "else_end" else_end_id;
              end;

        | Ast.STMT_while sw ->
            (* There are a bunch of rewirings to do on 'while' nodes. *)

            begin
              let dsts = Hashtbl.find graph s.id in
              let body = sw.Ast.while_body in
              let succ_stmts =
                List.filter (fun x -> not (x = body.id)) dsts
              in

              let (pre_loop_stmts, _) = sw.Ast.while_lval in
              let loop_head_id =
                (* Splice loop prelude into flow graph, save loop-head
                 * node.
                 *)
                let slen = Array.length pre_loop_stmts in
                  if slen > 0
                  then
                    begin
                      let pre_loop_begin = pre_loop_stmts.(0).id in
                      let pre_loop_end = last_id pre_loop_stmts in
                        remove_flow_edges graph s.id [body.id];
                        add_flow_edges graph s.id [pre_loop_begin];
                        add_flow_edges graph pre_loop_end [body.id];
                        pre_loop_end
                    end
                  else
                    body.id
              in

                (* Always flow s into the loop prelude; prelude may end
                 * loop.
                 *)
                remove_flow_edges graph s.id succ_stmts;
                add_flow_edges graph loop_head_id succ_stmts;

                (* Flow loop-end to loop-head. *)
                let loop_end = last_id_or_block_id body in
                  add_flow_edges graph loop_end [loop_head_id]
            end

        | Ast.STMT_alt_tag at ->
            let dsts = Hashtbl.find graph s.id in
            let arm_blocks =
              let arm_block_id { node = (_, block) } = block.id in
              Array.to_list (Array.map arm_block_id at.Ast.alt_tag_arms)
            in
            let succ_stmts =
              List.filter (fun x -> not (List.mem x arm_blocks)) dsts
            in
              remove_flow_edges graph s.id succ_stmts

        | _ -> ()
    end;
    inner.Walk.visit_stmt_post s
  in
    { inner with
        Walk.visit_stmt_pre = visit_stmt_pre }
;;

let find_roots
    (graph:(node_id, (node_id list)) Hashtbl.t)
    : (node_id,unit) Hashtbl.t =
  let roots = Hashtbl.create 0 in
    Hashtbl.iter (fun src _ -> Hashtbl.replace roots src ()) graph;
    Hashtbl.iter (fun _ dsts ->
                    List.iter (fun d -> Hashtbl.remove roots d) dsts) graph;
    roots
;;

let run_dataflow cx idref graph : unit =
  let roots = find_roots graph in
  let nodes = Queue.create () in
  let progress = ref true in
  let fmt_constr_bitv bitv =
    String.concat ", "
      (List.map
         (fun i ->
            fmt_constr_key cx
              (Hashtbl.find cx.ctxt_constrs (Constr i)))
         (Bits.to_list bitv))
  in
  let set_bits dst src =
    if Bits.copy dst src
    then (progress := true;
          iflog cx (fun _ -> log cx "made progress setting bits"))
  in
  let intersect_bits dst src =
    if Bits.intersect dst src
    then (progress := true;
          iflog cx (fun _ -> log cx
                      "made progress intersecting bits"))
  in
  let iter = ref 0 in
  let written = Hashtbl.create 0 in
  let tmp_diff = (Bits.create (!idref) false) in
  let tmp_poststate = (Bits.create (!idref) false) in
    Hashtbl.iter (fun n _ -> Queue.push n nodes) roots;
    while !progress do
      incr iter;
      progress := false;
      iflog cx (fun _ ->
                  log cx "";
                  log cx "--------------------";
                  log cx "dataflow pass %d" (!iter));
      Queue.iter
        begin
          fun node ->
            let prestate = Hashtbl.find cx.ctxt_prestates node in
            let precond = Hashtbl.find cx.ctxt_preconditions node in
            let postcond = Hashtbl.find cx.ctxt_postconditions node in
            let poststate = Hashtbl.find cx.ctxt_poststates node in

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
                      (match htab_search cx.ctxt_all_stmts node with
                           None -> "??"
                         | Some stmt -> Fmt.fmt_to_str Ast.fmt_stmt stmt);
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
                      Hashtbl.find cx.ctxt_prestates succ
                    in
                      if Hashtbl.mem written succ
                      then
                        begin
                          intersect_bits succ_prestates poststate;
                          Hashtbl.replace written succ ()
                        end
                      else
                        begin
                          progress := true;
                          Queue.push succ nodes;
                          set_bits succ_prestates poststate
                      end
                end
                successors
        end
        nodes
    done
;;

let typestate_verify_visitor
    (cx:ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =
  let visit_stmt_pre s =
    let prestate = Hashtbl.find cx.ctxt_prestates s.id in
    let precond = Hashtbl.find cx.ctxt_preconditions s.id in
      List.iter
        (fun i ->
           if not (Bits.get prestate i)
           then
             let ckey = Hashtbl.find cx.ctxt_constrs (Constr i) in
             let constr_str = fmt_constr_key cx ckey in
               err (Some s.id)
                 "Unsatisfied precondition constraint %s at stmt %d: %s"
                 constr_str
                 (int_of_node s.id)
                 (Fmt.fmt_to_str Ast.fmt_stmt
                    (Hashtbl.find cx.ctxt_all_stmts s.id)))
        (Bits.to_list precond);
      inner.Walk.visit_stmt_pre s
  in
    { inner with
        Walk.visit_stmt_pre = visit_stmt_pre }
;;

let lifecycle_visitor
    (cx:ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =

  (*
   * This visitor doesn't *calculate* part of the typestate; it uses
   * the typestates calculated in earlier passes to extract "summaries"
   * of slot-lifecycle events into the ctxt tables
   * ctxt_copy_stmt_is_init and ctxt_post_stmt_slot_drops. These are
   * used later on in translation.
   *)

  let (live_block_slots:(node_id, unit) Hashtbl.t) = Hashtbl.create 0 in
  let (block_slots:(node_id Stack.t) Stack.t) = Stack.create () in

  let (implicit_init_block_slots:(node_id,node_id) Hashtbl.t) =
    Hashtbl.create 0
  in

  let push_slot sl =
    Stack.push sl (Stack.top block_slots)
  in

  let mark_slot_init sl =
    Hashtbl.replace live_block_slots sl ()
  in


  let visit_block_pre b =
    Stack.push (Stack.create()) block_slots;
    begin
      match htab_search implicit_init_block_slots b.id with
          None -> ()
        | Some slot ->
            push_slot slot;
            mark_slot_init slot
    end;
    inner.Walk.visit_block_pre b
  in

  let note_drops stmt slots =
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

  let visit_block_post b =
    inner.Walk.visit_block_post b;
    let blk_slots = Stack.pop block_slots in
    let stmts = b.node in
    let len = Array.length stmts in
      if len > 0
      then
        begin
          let s = stmts.(len-1) in
            match s.node with
                Ast.STMT_ret _
              | Ast.STMT_be _ ->
                  () (* Taken care of in visit_stmt_post below. *)
              | _ ->
                (* The blk_slots stack we have has accumulated slots in
                 * declaration order as we walked the block; the top of the
                 * stack is the last-declared slot. We want to generate
                 * slot-drop obligations here for the slots in top-down order
                 * (starting with the last-declared) but only hitting those
                 * slots that actually got initialized (went live) at some
                 * point in the block.
                 *)
                let slots = stk_elts_from_top blk_slots in
                let live =
                  List.filter
                    (fun i -> Hashtbl.mem live_block_slots i)
                    slots
                in
                  note_drops s live
        end;
  in

  let visit_stmt_pre s =
    begin
      let init_lval lv_dst =
        let dst_slots = lval_slots cx lv_dst in
          Array.iter mark_slot_init dst_slots;
      in
        match s.node with
            Ast.STMT_copy (lv_dst, _)
          | Ast.STMT_call (lv_dst, _, _)
          | Ast.STMT_spawn (lv_dst, _, _, _)
          | Ast.STMT_recv (lv_dst, _)
          | Ast.STMT_bind (lv_dst, _, _) ->
              let prestate = Hashtbl.find cx.ctxt_prestates s.id in
              let poststate = Hashtbl.find cx.ctxt_poststates s.id in
              let dst_slots = lval_slots cx lv_dst in
              let is_initializing slot =
                let cid =
                  Hashtbl.find cx.ctxt_constr_ids (Constr_init slot)
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
                    Hashtbl.replace cx.ctxt_copy_stmt_is_init s.id ();
                    init_lval lv_dst
                  end;

          | Ast.STMT_decl (Ast.DECL_slot (_, sloti)) ->
              push_slot sloti.id

          | Ast.STMT_init_rec (lv_dst, _, _)
          | Ast.STMT_init_tup (lv_dst, _)
          | Ast.STMT_init_vec (lv_dst, _)
          | Ast.STMT_init_str (lv_dst, _)
          | Ast.STMT_init_port lv_dst
          | Ast.STMT_init_chan (lv_dst, _)
          | Ast.STMT_init_box (lv_dst, _) ->
              init_lval lv_dst

          | Ast.STMT_for f ->
              log cx "noting implicit init for slot %d in for-block %d"
                (int_of_node (fst f.Ast.for_slot).id)
                (int_of_node (f.Ast.for_body.id));
              htab_put implicit_init_block_slots
                f.Ast.for_body.id
                (fst f.Ast.for_slot).id

          | Ast.STMT_for_each f ->
              log cx "noting implicit init for slot %d in for_each-block %d"
                (int_of_node (fst f.Ast.for_each_slot).id)
                (int_of_node (f.Ast.for_each_body.id));
              htab_put implicit_init_block_slots
                f.Ast.for_each_body.id
                (fst f.Ast.for_each_slot).id


          | _ -> ()
    end;
    inner.Walk.visit_stmt_pre s
  in

  let visit_stmt_post s =
    inner.Walk.visit_stmt_post s;
    match s.node with
        Ast.STMT_ret _
      | Ast.STMT_be _ ->
          let stks = stk_elts_from_top block_slots in
          let slots = List.concat (List.map stk_elts_from_top stks) in
          let live =
            List.filter
              (fun i -> Hashtbl.mem live_block_slots i)
              slots
          in
            note_drops s live
      | _ -> ()
  in

    { inner with
        Walk.visit_block_pre = visit_block_pre;
        Walk.visit_block_post = visit_block_post;
        Walk.visit_stmt_pre = visit_stmt_pre;
        Walk.visit_stmt_post = visit_stmt_post
    }
;;

let process_crate
    (cx:ctxt)
    (crate:Ast.crate)
    : unit =
  let path = Stack.create () in
  let (scopes:(scope list) ref) = ref [] in
  let constr_id = ref 0 in
  let (graph:(node_id, (node_id list)) Hashtbl.t) = Hashtbl.create 0 in
  let sibs = Hashtbl.create 0 in
  let setup_passes =
    [|
      (scope_stack_managing_visitor scopes
         (constr_id_assigning_visitor cx scopes constr_id
            Walk.empty_visitor));
      (bitmap_assigning_visitor cx constr_id
         Walk.empty_visitor);
      (scope_stack_managing_visitor scopes
         (condition_assigning_visitor cx scopes
            Walk.empty_visitor));
      (graph_sequence_building_visitor cx graph sibs
         Walk.empty_visitor);
      (graph_general_block_structure_building_visitor cx graph sibs
         Walk.empty_visitor);
      (graph_special_block_structure_building_visitor cx graph
         Walk.empty_visitor);
    |]
  in
  let verify_passes =
    [|
      (scope_stack_managing_visitor scopes
         (typestate_verify_visitor cx
            Walk.empty_visitor))
    |]
  in
  let aux_passes =
    [|
      (lifecycle_visitor cx
         Walk.empty_visitor)
    |]
  in
    run_passes cx "typestate setup" path setup_passes (log cx "%s") crate;
    run_dataflow cx constr_id graph;
    run_passes cx "typestate verify" path verify_passes (log cx "%s") crate;
    run_passes cx "typestate aux" path aux_passes (log cx "%s") crate
;;


(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
