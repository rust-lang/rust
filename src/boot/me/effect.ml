open Semant;;
open Common;;

let log cx = Session.log "effect"
  cx.ctxt_sess.Session.sess_log_effect
  cx.ctxt_sess.Session.sess_log_out
;;

let iflog cx thunk =
  if cx.ctxt_sess.Session.sess_log_effect
  then thunk ()
  else ()
;;

let mutability_checking_visitor
    (cx:ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =
  (* 
   * This visitor enforces the following rules:
   * 
   * - A channel type carrying a mutable type is illegal.
   * 
   * - Writing to an immutable slot is illegal.
   * 
   * - Forming a mutable alias to an immutable slot is illegal.
   * 
   *)
  let visit_ty_pre t =
    match t with
        Ast.TY_chan t' when type_has_state t' ->
          err None "channel of mutable type: %a " Ast.sprintf_ty t'
      | _ -> ()
  in

  let check_write id dst =
    let dst_slot = lval_slot cx dst in
      if (dst_slot.Ast.slot_mutable or
            (Hashtbl.mem cx.ctxt_copy_stmt_is_init id))
      then ()
      else err (Some id) "writing to non-mutable slot"
  in
    (* FIXME (issue #75): enforce the no-write-alias-to-immutable-slot
     * rule.
     *)
  let visit_stmt_pre s =
    begin
      match s.node with
          Ast.STMT_copy (dst, _) -> check_write s.id dst
        | Ast.STMT_copy_binop (dst, _, _) -> check_write s.id dst
        | Ast.STMT_call (dst, _, _) -> check_write s.id dst
        | Ast.STMT_recv (dst, _) -> check_write s.id dst
        | _ -> ()
    end;
    inner.Walk.visit_stmt_pre s
  in

    { inner with
        Walk.visit_ty_pre = visit_ty_pre;
        Walk.visit_stmt_pre = visit_stmt_pre }
;;

let function_effect_propagation_visitor
    (item_effect:(node_id, Ast.effect) Hashtbl.t)
    (cx:ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =
  (* 
   * This visitor calculates the effect of each function according to
   * its statements:
   * 
   *    - Communication lowers to 'io'
   *    - Native calls lower to 'unsafe'
   *    - Calling a function with effect e lowers to e.
   *)
  let curr_fn = Stack.create () in

  let visit_mod_item_pre n p i =
    begin
      match i.node.Ast.decl_item with
          Ast.MOD_ITEM_fn _ -> Stack.push i.id curr_fn
        | _ -> ()
    end;
    inner.Walk.visit_mod_item_pre n p i
  in

  let visit_mod_item_post n p i =
    inner.Walk.visit_mod_item_post n p i;
    match i.node.Ast.decl_item with
        Ast.MOD_ITEM_fn _ -> ignore (Stack.pop curr_fn)
      | _ -> ()
  in

  let visit_obj_fn_pre o i fi =
    Stack.push fi.id curr_fn;
    inner.Walk.visit_obj_fn_pre o i fi
  in

  let visit_obj_fn_post o i fi =
    inner.Walk.visit_obj_fn_post o i fi;
    ignore (Stack.pop curr_fn)
  in

  let visit_obj_drop_pre o b =
    Stack.push b.id curr_fn;
    inner.Walk.visit_obj_drop_pre o b
  in

  let visit_obj_drop_post o b =
    inner.Walk.visit_obj_drop_post o b;
    ignore (Stack.pop curr_fn);
  in

  let lower_to s ne =
    let fn_id = Stack.top curr_fn in
    let e =
      match htab_search item_effect fn_id with
          None -> Ast.PURE
        | Some e -> e
    in
    let ne = lower_effect_of ne e in
      if ne <> e
      then
        begin
          iflog cx
            begin
              fun _ ->
                let name = Hashtbl.find cx.ctxt_all_item_names fn_id in
                  log cx "lowering calculated effect on '%a': '%a' -> '%a'"
                    Ast.sprintf_name name
                    Ast.sprintf_effect e
                    Ast.sprintf_effect ne;
                  log cx "at stmt %a" Ast.sprintf_stmt s
            end;
          Hashtbl.replace item_effect fn_id ne
        end;
  in

  let visit_stmt_pre s =
    begin
      match s.node with
          Ast.STMT_send _
        | Ast.STMT_recv _ -> lower_to s Ast.IO

        | Ast.STMT_call (_, fn, _) ->
            let lower_to_callee_ty t =
              match t with
                  Ast.TY_fn (_, taux) ->
                    lower_to s taux.Ast.fn_effect;
                | _ -> bug () "non-fn callee"
            in
              if lval_is_slot cx fn
              then
                let t = lval_slot cx fn in
                  lower_to_callee_ty (slot_ty t)
              else
                begin
                  let item = lval_item cx fn in
                  let t = Hashtbl.find cx.ctxt_all_item_types item.id in
                    lower_to_callee_ty t;
                    match htab_search cx.ctxt_required_items item.id with
                        None -> ()
                      | Some (REQUIRED_LIB_rust _, _) -> ()
                      | Some _ -> lower_to s Ast.UNSAFE
                end
        | _ -> ()
    end;
    inner.Walk.visit_stmt_pre s
  in

    { inner with
        Walk.visit_mod_item_pre = visit_mod_item_pre;
        Walk.visit_mod_item_post = visit_mod_item_post;
        Walk.visit_obj_fn_pre = visit_obj_fn_pre;
        Walk.visit_obj_fn_post = visit_obj_fn_post;
        Walk.visit_obj_drop_pre = visit_obj_drop_pre;
        Walk.visit_obj_drop_post = visit_obj_drop_post;
        Walk.visit_stmt_pre = visit_stmt_pre }
;;

let binding_effect_propagation_visitor
    ((*cx*)_:ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =
  (* This visitor lowers the effect of an object or binding according
   * to its slots: holding a 'state' slot lowers any obj item, or
   * bind-stmt LHS, to 'state'.
   * 
   * Binding (or implicitly just making a native 1st-class) makes the LHS
   * unsafe.
   *)
  inner
;;

let effect_checking_visitor
    (item_auth:(node_id, Ast.effect) Hashtbl.t)
    (item_effect:(node_id, Ast.effect) Hashtbl.t)
    (cx:ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =
  (*
   * This visitor checks that each type, item and obj declares
   * effects consistent with what we calculated.
   *)
  let auth_stack = Stack.create () in
  let visit_mod_item_pre n p i =
    begin
      match htab_search item_auth i.id with
          None -> ()
        | Some e ->
            let curr =
              if Stack.is_empty auth_stack
              then Ast.PURE
              else Stack.top auth_stack
            in
            let next = lower_effect_of e curr in
              Stack.push next auth_stack;
              iflog cx
                begin
                  fun _ ->
                    let name = Hashtbl.find cx.ctxt_all_item_names i.id in
                      log cx
                        "entering '%a', adjusting auth effect: '%a' -> '%a'"
                        Ast.sprintf_name name
                        Ast.sprintf_effect curr
                        Ast.sprintf_effect next
                end
    end;
    begin
      match i.node.Ast.decl_item with
          Ast.MOD_ITEM_fn f ->
            let e =
              match htab_search item_effect i.id with
                None -> Ast.PURE
              | Some e -> e
            in
            let fe = f.Ast.fn_aux.Ast.fn_effect in
            let ae =
              if Stack.is_empty auth_stack
              then None
              else Some (Stack.top auth_stack)
            in
              if e <> fe && (ae <> (Some e))
              then
                begin
                  let name = Hashtbl.find cx.ctxt_all_item_names i.id in
                    err (Some i.id)
                      "%a claims effect '%a' but calculated effect is '%a'%s"
                      Ast.sprintf_name name
                      Ast.sprintf_effect fe
                      Ast.sprintf_effect e
                      begin
                        match ae with
                            Some ae when ae <> fe ->
                              Printf.sprintf " (auth effect is '%a')"
                                Ast.sprintf_effect ae
                          | _ -> ""
                      end
                end
        | _ -> ()
    end;
    inner.Walk.visit_mod_item_pre n p i
  in
  let visit_mod_item_post n p i =
    inner.Walk.visit_mod_item_post n p i;
    match htab_search item_auth i.id with
        None -> ()
      | Some _ ->
          let curr = Stack.pop auth_stack in
          let next =
            if Stack.is_empty auth_stack
            then Ast.PURE
            else Stack.top auth_stack
          in
          iflog cx
            begin
              fun _ ->
                let name = Hashtbl.find cx.ctxt_all_item_names i.id in
                  log cx
                    "leaving '%a', restoring auth effect: '%a' -> '%a'"
                    Ast.sprintf_name name
                    Ast.sprintf_effect curr
                    Ast.sprintf_effect next
            end
  in
    { inner with
        Walk.visit_mod_item_pre = visit_mod_item_pre;
        Walk.visit_mod_item_post = visit_mod_item_post; }
;;


let process_crate
    (cx:ctxt)
    (crate:Ast.crate)
    : unit =
  let path = Stack.create () in
  let item_auth = Hashtbl.create 0 in
  let item_effect = Hashtbl.create 0 in
  let passes =
    [|
      (mutability_checking_visitor cx
         Walk.empty_visitor);
      (function_effect_propagation_visitor item_effect cx
         Walk.empty_visitor);
      (binding_effect_propagation_visitor cx
         Walk.empty_visitor);
      (effect_checking_visitor item_auth item_effect cx
         Walk.empty_visitor);
    |]
  in
  let root_scope = [ SCOPE_crate crate ] in
  let auth_effect name eff =
    match lookup_by_name cx root_scope name with
        None -> ()
      | Some (_, id) ->
          if referent_is_item cx id
          then htab_put item_auth id eff
          else err (Some id) "auth clause in crate refers to non-item"
  in
    Hashtbl.iter auth_effect crate.node.Ast.crate_auth;
    run_passes cx "effect" path passes (log cx "%s") crate
;;

(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
