open Common;;
open Semant;;

let log cx =
  Session.log
    "simplify"
    (should_log cx cx.Semant.ctxt_sess.Session.sess_log_simplify)
    cx.Semant.ctxt_sess.Session.sess_log_out

let iflog cx thunk =
  if (should_log cx cx.Semant.ctxt_sess.Session.sess_log_simplify)
  then thunk ()
  else ()
;;


let plval_const_marking_visitor
    (cx:Semant.ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =
  let visit_pexp_pre pexp =
    begin
      match pexp.node with
          Ast.PEXP_lval pl ->
            begin
              let id = lval_base_id_to_defn_base_id cx pexp.id in
              let is_const =
                if defn_id_is_item cx id
                then match (get_item cx id).Ast.decl_item with
                    Ast.MOD_ITEM_const _ -> true
                  | _ -> false
                else false
              in
                iflog cx (fun _ -> log cx "plval %a refers to %s"
                            Ast.sprintf_plval pl
                            (if is_const then "const item" else "non-const"));
                htab_put cx.ctxt_plval_const pexp.id is_const
            end
        | _ -> ()
    end;
    inner.Walk.visit_pexp_pre pexp
  in

  let visit_pexp_post p =
    inner.Walk.visit_pexp_post p;
    iflog cx (fun _ -> log cx "pexp %a is %s"
                Ast.sprintf_pexp p
                (if pexp_is_const cx p
                 then "constant"
                 else "non-constant"))
  in

    { inner with
        Walk.visit_pexp_pre = visit_pexp_pre;
        Walk.visit_pexp_post = visit_pexp_post;
    }
;;


let pexp_simplifying_visitor
    (_:Semant.ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =

  let walk_atom at =
    match at with
        Ast.ATOM_pexp _ ->
          begin
            (* FIXME: move desugaring code from frontend to here. *)
            ()
          end
      | _ -> ()
  in

  let visit_stmt_pre s =
    begin
      match s.node with
          Ast.STMT_copy (_, Ast.EXPR_atom a) -> walk_atom a
        | _ -> ()
    end;
    inner.Walk.visit_stmt_pre s;
  in
    { inner with
        Walk.visit_stmt_pre = visit_stmt_pre;
    }
;;


let process_crate (cx:Semant.ctxt) (crate:Ast.crate) : unit =

  let passes =
    [|
      (plval_const_marking_visitor cx Walk.empty_visitor);
      (pexp_simplifying_visitor cx Walk.empty_visitor)
    |]
  in
  let log_flag = cx.Semant.ctxt_sess.Session.sess_log_simplify in
    Semant.run_passes cx "simplify" passes log_flag log crate
;;

(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)

