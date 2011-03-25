open Semant;;
open Common;;

let log cx = Session.log "alias"
  (should_log cx cx.ctxt_sess.Session.sess_log_alias)
  cx.ctxt_sess.Session.sess_log_out
;;

let alias_analysis_visitor
    (cx:ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =
  let curr_stmt = Stack.create () in

  let alias_slot (slot_id:node_id) : unit =
    begin
      log cx "noting slot #%d as aliased" (int_of_node slot_id);
      Hashtbl.replace cx.ctxt_slot_aliased slot_id ()
    end
  in

  let alias lval =
    let defn_id = lval_base_defn_id cx lval in
      if (defn_id_is_slot cx defn_id)
      then alias_slot defn_id
  in

  let alias_atom at =
    match at with
        Ast.ATOM_lval lv -> alias lv
      | _ -> () (* Aliasing a literal is harmless, if weird. *)
  in

  let alias_call_args dst callee args =
    alias dst;
    let callee_ty = lval_ty cx callee in
      match callee_ty with
          Ast.TY_fn (tsig,_) ->
            Array.iteri
              begin
                fun i slot ->
                  match slot.Ast.slot_mode with
                      Ast.MODE_alias  ->
                        alias_atom args.(i)
                    | _ -> ()
              end
              tsig.Ast.sig_input_slots
        | _ -> ()
  in

  let check_no_alias_bindings
      (fn:Ast.lval)
      (args:(Ast.atom option) array)
      : unit =
    let fty = match lval_ty cx fn with
        Ast.TY_fn tfn -> tfn
      | _ -> err (Some (lval_base_id fn)) "binding non-fn"
    in
    let arg_slots = (fst fty).Ast.sig_input_slots in
      Array.iteri
        begin
          fun i arg ->
            match arg with
                None -> ()
              | Some _ ->
                  match arg_slots.(i).Ast.slot_mode with
                      Ast.MODE_local -> ()
                    | Ast.MODE_alias ->
                        err (Some (lval_base_id fn)) "binding alias slot"
        end
        args
  in

  let visit_stmt_pre s =
    Stack.push s.id curr_stmt;
    begin
      try
        match s.node with
            (* FIXME (issue #26): actually all these *existing* cases
             * can probably go now that we're using Trans.aliasing to
             * form short-term spill-based aliases. Only aliases that
             * survive 'into' a sub-block (those formed during iteration)
             * need to be handled in this module.  *)
            Ast.STMT_call (dst, callee, args)
          | Ast.STMT_spawn (dst, _, _, callee, args)
            -> alias_call_args dst callee args

          | Ast.STMT_bind (_, fn, args) ->
              check_no_alias_bindings fn args

          | Ast.STMT_send (_, src) -> alias src
          | Ast.STMT_recv (dst, _) -> alias dst
          | Ast.STMT_new_port (dst) -> alias dst
          | Ast.STMT_new_chan (dst, _) -> alias dst
          | Ast.STMT_new_vec (dst, _, _) -> alias dst
          | Ast.STMT_new_str (dst, _) -> alias dst
          | Ast.STMT_for_each sfe ->
              let (slot, _) = sfe.Ast.for_each_slot in
                alias_slot slot.id
          | _ -> () (* FIXME (issue #29): plenty more to handle here. *)
      with
          Semant_err (None, msg) ->
            raise (Semant_err ((Some s.id), msg))
    end;
    inner.Walk.visit_stmt_pre s
  in
  let visit_stmt_post s =
    inner.Walk.visit_stmt_post s;
    ignore (Stack.pop curr_stmt);
  in

  let visit_lval_pre lv =
    let slot_id = lval_base_defn_id cx lv in
      if (not (Stack.is_empty curr_stmt)) && (defn_id_is_slot cx slot_id)
      then
        begin
          let slot_depth = get_slot_depth cx slot_id in
          let stmt_depth = get_stmt_depth cx (Stack.top curr_stmt) in
            if slot_depth <> stmt_depth
            then
              begin
                let _ = assert (slot_depth < stmt_depth) in
                  alias_slot slot_id
              end
        end
  in

    { inner with
        Walk.visit_stmt_pre = visit_stmt_pre;
        Walk.visit_stmt_post = visit_stmt_post;
        Walk.visit_lval_pre = visit_lval_pre
    }
;;

let process_crate
    (cx:ctxt)
    (crate:Ast.crate)
    : unit =
  let passes =
    [|
      (alias_analysis_visitor cx
         Walk.empty_visitor);
    |]
  in
    run_passes cx "alias" passes
      cx.ctxt_sess.Session.sess_log_alias log crate
;;

(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
