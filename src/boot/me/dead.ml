(* 
 * A simple dead-code analysis that rejects code following unconditional
 * 'ret' or 'be'. 
 *)

open Semant;;
open Common;;

let log cx = Session.log "dead"
  cx.ctxt_sess.Session.sess_log_dead
  cx.ctxt_sess.Session.sess_log_out
;;

let dead_code_visitor
    ((*cx*)_:ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =

  (* FIXME: create separate table for each fn body for less garbage *)
  let must_exit = Hashtbl.create 100 in

  let all_must_exit ids =
    arr_for_all (fun _ id -> Hashtbl.mem must_exit id) ids
  in

  let visit_block_post block =
    let stmts = block.node in
    let len = Array.length stmts in
      if len > 0 then
        Array.iteri
          begin
            fun i s ->
              if (i < (len - 1)) && (Hashtbl.mem must_exit s.id) then
                err (Some stmts.(i + 1).id) "dead statement"
          end
          stmts;
      inner.Walk.visit_block_post block
  in

  let visit_stmt_post s =
    begin
      match s.node with
        | Ast.STMT_block block ->
            if Hashtbl.mem must_exit block.id then
              Hashtbl.add must_exit s.id ()

        | Ast.STMT_while { Ast.while_body = body }
        | Ast.STMT_do_while { Ast.while_body = body }
        | Ast.STMT_for_each { Ast.for_each_body = body }
        | Ast.STMT_for { Ast.for_body = body } ->
            if (Hashtbl.mem must_exit body.id) then
              Hashtbl.add must_exit s.id ()

        | Ast.STMT_if { Ast.if_then = b1; Ast.if_else = Some b2 } ->
            if (Hashtbl.mem must_exit b1.id) && (Hashtbl.mem must_exit b2.id)
            then Hashtbl.add must_exit s.id ()

        | Ast.STMT_if _ -> ()

        | Ast.STMT_ret _
        | Ast.STMT_be _ ->
            Hashtbl.add must_exit s.id ()

        | Ast.STMT_alt_tag { Ast.alt_tag_arms = arms } ->
            let arm_ids =
              Array.map (fun { node = (_, block) } -> block.id) arms
            in
              if all_must_exit arm_ids
              then Hashtbl.add must_exit s.id ()

        | Ast.STMT_alt_type { Ast.alt_type_arms = arms;
                              Ast.alt_type_else = alt_type_else } ->
            let arm_ids = Array.map (fun (_, _, block) -> block.id) arms in
            let else_ids =
              begin
                match alt_type_else with
                    Some stmt -> [| stmt.id |]
                  | None -> [| |]
              end
            in
              if all_must_exit (Array.append arm_ids else_ids) then
                Hashtbl.add must_exit s.id ()

        (* FIXME: figure this one out *)
        | Ast.STMT_alt_port _ -> ()

        | _ -> ()
    end;
    inner.Walk.visit_stmt_post s

  in
    { inner with
        Walk.visit_block_post = visit_block_post;
        Walk.visit_stmt_post = visit_stmt_post }
;;

let process_crate
    (cx:ctxt)
    (crate:Ast.crate)
    : unit =
  let path = Stack.create () in
  let passes =
    [|
      (dead_code_visitor cx
         Walk.empty_visitor)
    |]
  in

    run_passes cx "dead" path passes (log cx "%s") crate;
    ()
;;


(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
