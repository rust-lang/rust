open Semant;;
open Common;;

let log cx = Session.log "effect"
  (should_log cx cx.ctxt_sess.Session.sess_log_effect)
  cx.ctxt_sess.Session.sess_log_out
;;

let iflog cx thunk =
  if (should_log cx cx.ctxt_sess.Session.sess_log_effect)
  then thunk ()
  else ()
;;

let effect_calculating_visitor
    (item_effect:(node_id, Ast.effect) Hashtbl.t)
    (cx:ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =
  (* 
   * This visitor calculates the effect of each function according to
   * its statements:
   * 
   *    - Communication statements lower to 'impure'
   *    - Writing to anything other than a local slot lowers to 'impure'
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
          None -> Ast.EFF_pure
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

  let note_write s dst =
    (* FIXME (issue #182): this is too aggressive; won't permit writes to
     * interior components of records or tuples. It should at least do that,
     * possibly handle escape analysis on the pointee for things like vecs as
     * well.  *)
    if lval_base_is_slot cx dst
    then
      let base_slot = lval_base_slot cx dst in
        match dst, base_slot.Ast.slot_mode with
            (Ast.LVAL_base _, Ast.MODE_local) -> ()
          | _ -> lower_to s Ast.EFF_impure
  in

  let visit_stmt_pre s =
    begin
      match s.node with
          Ast.STMT_send _
        | Ast.STMT_recv _ -> lower_to s Ast.EFF_impure

        | Ast.STMT_call (lv_dst, fn, _) ->
            note_write s lv_dst;
            let lower_to_callee_ty t =
              match simplified_ty t with
                  Ast.TY_fn (_, taux) ->
                    lower_to s taux.Ast.fn_effect;
                | _ -> bug () "non-fn callee"
            in
              if lval_base_is_slot cx fn
              then
                lower_to_callee_ty (lval_ty cx fn)
              else
                begin
                  let item = lval_item cx fn in
                  let t = Hashtbl.find cx.ctxt_all_item_types item.id in
                    lower_to_callee_ty t;
                    match htab_search cx.ctxt_required_items item.id with
                        None -> ()
                      | Some (REQUIRED_LIB_rust _, _) -> ()
                      | Some _ -> lower_to s Ast.EFF_unsafe
                end

        | Ast.STMT_copy (lv_dst, _)
        | Ast.STMT_spawn (lv_dst, _, _, _, _)
        | Ast.STMT_bind (lv_dst, _, _)
        | Ast.STMT_new_rec (lv_dst, _, _)
        | Ast.STMT_new_tup (lv_dst, _)
        | Ast.STMT_new_vec (lv_dst, _, _)
        | Ast.STMT_new_str (lv_dst, _)
        | Ast.STMT_new_port lv_dst
        | Ast.STMT_new_chan (lv_dst, _)
        | Ast.STMT_new_box (lv_dst, _, _) ->
            note_write s lv_dst

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


let effect_checking_visitor
    (item_auth:(node_id, Ast.effect) Hashtbl.t)
    (item_effect:(node_id, Ast.effect) Hashtbl.t)
    (cx:ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =
  (*
   * This visitor checks that each fn declares
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
              then Ast.EFF_pure
              else Stack.top auth_stack
            in
              Stack.push e auth_stack;
              iflog cx
                begin
                  fun _ ->
                    let name = Hashtbl.find cx.ctxt_all_item_names i.id in
                      log cx
                        "entering '%a', adjusting auth effect: '%a' -> '%a'"
                        Ast.sprintf_name name
                        Ast.sprintf_effect curr
                        Ast.sprintf_effect e
                end
    end;
    let report_mismatch declared_effect calculated_effect =
      let name = Hashtbl.find cx.ctxt_all_item_names i.id in
        err (Some i.id)
          "%a claims effect '%a' but calculated effect is '%a'%s"
          Ast.sprintf_name name
          Ast.sprintf_effect declared_effect
          Ast.sprintf_effect calculated_effect
          begin
            if Stack.is_empty auth_stack
            then ""
            else
              Printf.sprintf " (auth effects are '%s')"
                (stk_fold
                   auth_stack
                   (fun e s ->
                      if s = ""
                      then
                        Printf.sprintf "%a"
                          Ast.sprintf_effect e
                      else
                        Printf.sprintf "%s, %a" s
                          Ast.sprintf_effect e) "")
          end
    in
    begin
      match i.node.Ast.decl_item with
          Ast.MOD_ITEM_fn f
            when htab_search cx.ctxt_required_items i.id = None ->
            let calculated_effect =
              match htab_search item_effect i.id with
                None -> Ast.EFF_pure
              | Some e -> e
            in
            let declared_effect = f.Ast.fn_aux.Ast.fn_effect in
              if calculated_effect <> declared_effect
              then
                (* Something's fishy in this case. If the calculated effect
                 * is equal to one auth'ed by an enclosing scope -- not just
                 * a lower one -- we accept this mismatch; otherwise we
                 * complain.
                 * 
                 * FIXME: this choice of "what constitutes an error" in
                 * auth/effect mismatches is subjective and could do
                 * with some discussion.  *)
                begin
                  match
                    stk_search auth_stack
                      (fun e ->
                         if e = calculated_effect then Some e else None)
                  with
                      Some _ -> ()
                    | None ->
                        report_mismatch declared_effect calculated_effect
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
            then Ast.EFF_pure
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
  let item_auth = Hashtbl.create 0 in
  let item_effect = Hashtbl.create 0 in
  let passes =
    [|
      (effect_calculating_visitor item_effect cx
         Walk.empty_visitor);
      (effect_checking_visitor item_auth item_effect cx
         Walk.empty_visitor);
    |]
  in
  let root_scope = [ SCOPE_crate crate ] in
  let auth_effect name eff =
    match lookup_by_name cx [] root_scope name with
        RES_failed _ -> ()
      | RES_ok (_, id) ->
          if defn_id_is_item cx id
          then htab_put item_auth id eff
          else err (Some id) "auth clause in crate refers to non-item"
  in
    Hashtbl.iter auth_effect crate.node.Ast.crate_auth;
    run_passes cx "effect" passes
      cx.ctxt_sess.Session.sess_log_effect log crate
;;

(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
