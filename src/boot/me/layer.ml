open Semant;;
open Common;;

let log cx = Session.log "layer"
  (should_log cx cx.ctxt_sess.Session.sess_log_layer)
  cx.ctxt_sess.Session.sess_log_out
;;

let iflog cx thunk =
  if (should_log cx cx.ctxt_sess.Session.sess_log_layer)
  then thunk ()
  else ()
;;


let state_layer_checking_visitor
    (cx:ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =
  (* 
   * This visitor enforces the following rules:
   * 
   * - A channel type carrying a state type is illegal.
   * 
   * - Writing to an immutable slot is illegal.
   * 
   * - Forming a mutable alias to an immutable slot is illegal.
   * 
   *)
  let visit_ty_pre t =
    match t with
        Ast.TY_chan t' when type_has_state cx t' ->
          err None "channel of state type: %a " Ast.sprintf_ty t'
      | _ -> ()
  in

  let check_write s dst =
    let is_init = Hashtbl.mem cx.ctxt_stmt_is_init s.id in
    let dst_ty = lval_ty cx dst in
    let is_mutable =
      match dst_ty with
          Ast.TY_mutable _ -> true
        | _ -> false
    in
      iflog cx
        (fun _ -> log cx "checking %swrite to %slval #%d = %a of type %a"
           (if is_init then "initializing " else "")
           (if is_mutable then "mutable " else "")
           (int_of_node (lval_base_id dst))
           Ast.sprintf_lval dst
           Ast.sprintf_ty dst_ty);
      if (is_mutable or is_init)
      then ()
      else err (Some s.id)
        "writing to immutable type %a in statement %a"
        Ast.sprintf_ty dst_ty Ast.sprintf_stmt s
  in
    (* FIXME (issue #75): enforce the no-write-alias-to-immutable-slot
     * rule.
     *)
  let visit_stmt_pre s =
    begin
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
              check_write s lv_dst
        | _ -> ()
    end;
    inner.Walk.visit_stmt_pre s
  in

    { inner with
        Walk.visit_ty_pre = visit_ty_pre;
        Walk.visit_stmt_pre = visit_stmt_pre }
;;

let process_crate
    (cx:ctxt)
    (crate:Ast.crate)
    : unit =
  let passes =
    [|
      (state_layer_checking_visitor cx
         Walk.empty_visitor);
    |]
  in
    run_passes cx "layer" passes
      cx.ctxt_sess.Session.sess_log_layer log crate
;;

(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
