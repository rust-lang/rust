
open Common;;
open Token;;
open Parser;;

(* Item grammar. *)

let default_exports =
  let e = Hashtbl.create 0 in
    Hashtbl.add e Ast.EXPORT_all_decls ();
    e
;;

let empty_view = { Ast.view_imports = Hashtbl.create 0;
                   Ast.view_exports = default_exports }
;;

let rec parse_expr (ps:pstate) : (Ast.stmt array * Ast.expr) =
  let pexp = ctxt "expr" Pexp.parse_pexp ps in
    if ps.pstate_sess.Session.sess_use_pexps
    then ([||], Ast.EXPR_atom (Ast.ATOM_pexp pexp))
    else Pexp.desugar_expr ps pexp

and parse_prim_expr (ps:pstate) : Ast.expr =
  let pexp = ctxt "expr" Pexp.parse_pexp ps in
  let (stmts, expr) = Pexp.desugar_expr ps pexp in
    if Array.length stmts = 0
    then expr
    else raise (Parse_err (ps, "expected primitive expression"))

and parse_expr_atom (ps:pstate) : (Ast.stmt array * Ast.atom) =
  let pexp = ctxt "expr" Pexp.parse_pexp ps in
    if ps.pstate_sess.Session.sess_use_pexps
    then ([||], Ast.ATOM_pexp pexp)
    else Pexp.desugar_expr_atom ps pexp

and parse_expr_atom_list
    (bra:token)
    (ket:token)
    (ps:pstate)
    : (Ast.stmt array * Ast.atom array) =
  arj1st (bracketed_zero_or_more bra ket (Some COMMA)
            (ctxt "expr-atom list" parse_expr_atom) ps)

and parse_expr_init (lv:Ast.lval) (ps:pstate) : (Ast.stmt array) =
  let apos = lexpos ps in
  let pexp = ctxt "expr" Pexp.parse_pexp ps in
  let bpos = lexpos ps in
    if ps.pstate_sess.Session.sess_use_pexps
    then [|
      span ps apos bpos
        (Ast.STMT_copy (lv, Ast.EXPR_atom (Ast.ATOM_pexp pexp)))
    |]
    else Pexp.desugar_expr_init ps lv pexp

and parse_lval (ps:pstate) : (Ast.stmt array * Ast.lval) =
  let apos = lexpos ps in
  let pexp = ctxt "lval" Pexp.parse_pexp ps in
  let bpos = lexpos ps in
    if ps.pstate_sess.Session.sess_use_pexps
    then
      let (_, tmp, decl_stmt) = build_tmp ps slot_auto apos bpos in
      let copy_stmt =
        span ps apos bpos
          (Ast.STMT_copy (tmp, Ast.EXPR_atom (Ast.ATOM_pexp pexp)))
      in
        ([| decl_stmt; copy_stmt |], (clone_lval ps tmp))
    else Pexp.desugar_lval ps pexp

and parse_identified_slot_and_ident
    (aliases_ok:bool)
    (ps:pstate)
    : (Ast.slot identified * Ast.ident) =
  let slot =
    ctxt "identified slot and ident: slot"
      (Pexp.parse_identified_slot aliases_ok) ps
  in
  let ident =
    ctxt "identified slot and ident: ident" Pexp.parse_ident ps
  in
    (slot, ident)

and parse_zero_or_more_identified_slot_ident_pairs
    (aliases_ok:bool)
    (ps:pstate)
    : (((Ast.slot identified) * Ast.ident) array) =
  ctxt "zero+ slots and idents"
    (paren_comma_list
       (parse_identified_slot_and_ident aliases_ok)) ps

and parse_block (ps:pstate) : Ast.block =
  let apos = lexpos ps in
  let stmts =
    arj (ctxt "block: stmts"
           (bracketed_zero_or_more LBRACE RBRACE
              None parse_stmts) ps)
  in
  let bpos = lexpos ps in
    span ps apos bpos stmts

and parse_block_stmt (ps:pstate) : Ast.stmt =
  let apos = lexpos ps in
  let block = parse_block ps in
  let bpos = lexpos ps in
    span ps apos bpos (Ast.STMT_block block)

and parse_init
    (lval:Ast.lval)
    (ps:pstate)
    : Ast.stmt array =
  let apos = lexpos ps in
  let stmts =
    match peek ps with
        EQ ->
          bump ps;
          parse_expr_init lval ps
      | LARROW ->
          bump ps;
          let (stmts, rhs) = ctxt "init: port" parse_lval ps in
          let bpos = lexpos ps in
          let stmt = Ast.STMT_recv (lval, rhs) in
            Array.append stmts [| (span ps apos bpos stmt) |]
      | _ -> arr []
  in
  let _ = expect ps SEMI in
    stmts

and parse_slot_and_ident_and_init
    (ps:pstate)
    : (Ast.stmt array * Ast.slot * Ast.ident) =
  let apos = lexpos ps in
  let (slot, ident) =
    ctxt "slot, ident and init: slot and ident"
      (Pexp.parse_slot_and_ident false) ps
  in
  let bpos = lexpos ps in
  let lval = Ast.LVAL_base (span ps apos bpos (Ast.BASE_ident ident)) in
  let stmts = ctxt "slot, ident and init: init" (parse_init lval) ps in
    (stmts, slot, ident)

and parse_auto_slot_and_init
    (ps:pstate)
    : (Ast.stmt array * Ast.slot * Ast.ident) =
  let apos = lexpos ps in
  let ident = Pexp.parse_ident ps in
  let bpos = lexpos ps in
  let lval = Ast.LVAL_base (span ps apos bpos (Ast.BASE_ident ident)) in
  let stmts = ctxt "slot, ident and init: init" (parse_init lval) ps in
    (stmts, slot_auto, ident)


and parse_stmts (ps:pstate) : Ast.stmt array =
  let arr = parse_stmts_including_none ps in
    if (Array.length arr) == 0 then
      raise (err "statement does nothing" ps);
    arr

(*
 * We have no way to parse a single Ast.stmt; any incoming syntactic statement
 * may desugar to N>1 real Ast.stmts
 *)

and parse_stmts_including_none (ps:pstate) : Ast.stmt array =
  let apos = lexpos ps in

  let ensure_mutable slot =
    match slot.Ast.slot_ty with
        None -> slot
      | Some (Ast.TY_mutable _) -> slot
      | Some t -> { slot with Ast.slot_ty = Some (Ast.TY_mutable t) }
  in

  let rec name_to_lval (apos:pos) (bpos:pos) (name:Ast.name)
      : Ast.lval =
    match name with
        Ast.NAME_base nb ->
          Ast.LVAL_base (span ps apos bpos nb)
      | Ast.NAME_ext (n, nc) ->
          Ast.LVAL_ext (name_to_lval apos bpos n, Ast.COMP_named nc)
  in

    match peek ps with

        LOG ->
          bump ps;
          let (stmts, atom) = ctxt "stmts: log value" parse_expr_atom ps in
            expect ps SEMI;
            spans ps stmts apos (Ast.STMT_log atom)
      | LOG_ERR ->
          bump ps;
          let (stmts, atom) = ctxt "stmts: log value" parse_expr_atom ps in
            expect ps SEMI;
            spans ps stmts apos (Ast.STMT_log_err atom)
      | BREAK ->
          bump ps;
          expect ps SEMI;
          [| span ps apos (lexpos ps) Ast.STMT_break |]
      | CONT ->
          bump ps;
          expect ps SEMI;
          [| span ps apos (lexpos ps) Ast.STMT_cont |]
      | ASSERT ->
          bump ps;
          let (stmts, expr) =
             ctxt "stmts: check value" parse_expr ps
          in
              expect ps SEMI;
              spans ps stmts apos (Ast.STMT_check_expr expr)
(* leaving check as it is; adding assert as a synonym for the "old" check *)
      | CHECK ->
          bump ps;
          begin

            let rec carg_path_to_lval (bpos:pos) (path:Ast.carg_path)
                : Ast.lval =
              match path with
                  Ast.CARG_base Ast.BASE_formal ->
                    raise (err "converting formal constraint-arg to atom" ps)
                | Ast.CARG_base (Ast.BASE_named nb) ->
                    Ast.LVAL_base (span ps apos bpos nb)
                | Ast.CARG_ext (pth, nc) ->
                    Ast.LVAL_ext (carg_path_to_lval bpos pth,
                                  Ast.COMP_named nc)
            in

            let carg_to_atom (bpos:pos) (carg:Ast.carg)
                : Ast.atom =
              match carg with
                  Ast.CARG_lit lit ->
                    Ast.ATOM_literal (span ps apos bpos lit)
                | Ast.CARG_path pth ->
                    Ast.ATOM_lval (carg_path_to_lval bpos pth)
            in

            let synthesise_check_call (bpos:pos) (constr:Ast.constr)
                : (Ast.lval * (Ast.atom array)) =
              let lval = name_to_lval apos bpos constr.Ast.constr_name in
              let args =
                Array.map (carg_to_atom bpos) constr.Ast.constr_args
              in
                (lval, args)
            in

            let synthesise_check_calls (bpos:pos) (constrs:Ast.constrs)
                : Ast.check_calls =
              Array.map (synthesise_check_call bpos) constrs
            in

              match peek ps with
                  LPAREN ->
                    bump ps;
                    let (stmts, expr) =
                      ctxt "stmts: check value" parse_expr ps
                    in
                      expect ps RPAREN;
                      expect ps SEMI;
                      spans ps stmts apos (Ast.STMT_check_expr expr)

                | IF ->
                    bump ps;
                    expect ps LPAREN;
                    let constrs = Pexp.parse_constrs ps in
                      expect ps RPAREN;
                      let block = parse_block ps in
                      let bpos = lexpos ps in
                      let calls = synthesise_check_calls bpos constrs in
                        [| span ps apos bpos
                             (Ast.STMT_check_if (constrs, calls, block))
                        |]

                | _ ->
                    let constrs = Pexp.parse_constrs ps in
                      expect ps SEMI;
                      let bpos = lexpos ps in
                      let calls = synthesise_check_calls bpos constrs in
                        [| span ps apos bpos
                             (Ast.STMT_check (constrs, calls))
                        |]
          end

      | ALT ->
          bump ps;
          let rec parse_pat ps  =
            match peek ps with
                QUES ->
                  let apos = lexpos ps in
                    bump ps;
                    let name = Pexp.parse_name ps in
                    let bpos = lexpos ps in
                      begin
                        match name with
                            Ast.NAME_base (Ast.BASE_ident ident) ->
                              let slot =
                                { Ast.slot_mode = Ast.MODE_local;
                                  Ast.slot_ty = None }
                              in
                                Left
                                  (Ast.PAT_slot ((span ps apos bpos slot),
                                                 ident))
                          |_ -> raise (unexpected ps)
                      end

              | IDENT _ ->
                  let apos = lexpos ps in
                  let name = Pexp.parse_name ps in
                  let bpos = lexpos ps in
                  let lv = name_to_lval apos bpos name in
                  let parse_pat ps = either_get_left (parse_pat ps) in
                  let args =
                    match peek ps with
                        LPAREN ->
                          let args = paren_comma_list parse_pat ps in
                            if Array.length args = 0
                            then raise (err "empty pattern argument list" ps)
                            else args
                      | _ -> [| |]
                  in
                    Left (Ast.PAT_tag (lv, args))

              | LIT_INT _
              | LIT_UINT _
              | LIT_CHAR _
              | LIT_BOOL _ ->
                  Left (Ast.PAT_lit (Pexp.parse_lit ps))

              | UNDERSCORE -> bump ps; Left (Ast.PAT_wild)

              | tok -> raise (Parse_err (ps,
                                         "Expected pattern but found '" ^
                                           (string_of_tok tok) ^ "'"))
          in
          let rec parse_arms ps parse_case =
            match peek ps with
                CASE ->
                  bump ps;
                  let case = parse_case ps in
                  let blk = parse_block ps in
                  let combine_and_span case =
                    (span ps apos (lexpos ps) (case, blk)) in
                  let is_default = either_has_right case in
                    if is_default then
                      let arm = combine_and_span (either_get_right case) in
                        ([], Some arm)
                    else
                      let rec_result = parse_arms ps parse_case in
                      let arm = combine_and_span (either_get_left case) in
                        (arm::(fst rec_result), (snd rec_result))
              | _ -> ([], None)
          in
          let parse_alt_block ps str parse_case make_stmt =
            let br_parse_case = bracketed LPAREN RPAREN parse_case in
            let arms = (ctxt (String.concat " " ["alt"; str; "arms"])
                          (fun ps -> parse_arms ps br_parse_case) ps) in
              make_stmt (fst arms) (snd arms)
          in
          let which_alt = match peek ps with
              TYPE -> "type" | LPAREN -> "tag" | _ -> raise (unexpected ps)
          in
          let (stmts, lval) = if which_alt = "type" then bump ps;
            bracketed LPAREN RPAREN parse_lval ps
          in
          let make_alt_tag_stmt val_arms dflt_arm =
            assert (not (bool_of_option dflt_arm));
            spans ps stmts apos begin
              Ast.STMT_alt_tag {
                Ast.alt_tag_lval = lval;
                Ast.alt_tag_arms = Array.of_list val_arms;
              }
            end
          in
          let make_alt_type_stmt val_arms dflt_arm =
            spans ps stmts apos begin
              Ast.STMT_alt_type {
                Ast.alt_type_lval = lval;
                Ast.alt_type_arms = Array.of_list val_arms;
                Ast.alt_type_else = option_map (fun x -> snd x.node) dflt_arm;
              }
            end
          in
          let parse_slot_and_ident ps =
            match peek ps with
                UNDERSCORE -> Right ()
              | _ -> Left (pair_rev (Pexp.parse_slot_and_ident false ps))

          in
          let parse_alt_tag_block ps =
            parse_alt_block ps
              "tag"
              parse_pat
              make_alt_tag_stmt
          in
          let parse_alt_type_block ps =
            parse_alt_block ps
              "type"
              parse_slot_and_ident
              make_alt_type_stmt
          in
          let parse_alt_block2 ps =
            match which_alt with
                "type" -> parse_alt_type_block ps
              | "tag" -> parse_alt_tag_block ps
              | _ -> assert false
          in
            bracketed LBRACE RBRACE parse_alt_block2 ps
      | IF ->
          let final_else = ref None in
          let rec parse_stmt_if _ =
            bump ps;
            let (stmts, expr) =
              ctxt "stmts: if cond"
                (bracketed LPAREN RPAREN parse_expr) ps
            in
            let then_block = ctxt "stmts: if-then" parse_block ps in
              begin
                match peek ps with
                    ELSE ->
                      begin
                        bump ps;
                        match peek ps with
                            IF ->
                              let nested_if = parse_stmt_if () in
                              let bpos = lexpos ps in
                                final_else :=
                                  Some (span ps apos bpos nested_if)
                          | _ ->
                              final_else :=
                                Some (ctxt "stmts: if-else" parse_block ps)
                      end
                  | _ -> ()
              end;
              let res =
                spans ps stmts apos
                  (Ast.STMT_if
                     { Ast.if_test = expr;
                       Ast.if_then = then_block;
                       Ast.if_else = !final_else; })
              in
                final_else := None;
                res
          in
            parse_stmt_if()

      | FOR ->
          bump ps;
          begin
            match peek ps with
                EACH ->
                  bump ps;
                  let inner ps : ((Ast.slot identified * Ast.ident)
                                  * Ast.stmt array
                                  * (Ast.lval * Ast.atom array)) =
                    let slot = (parse_identified_slot_and_ident true ps) in
                    let _    = (expect ps IN) in
                    let (stmts1, iter) = (rstr true parse_lval) ps in
                    let (stmts2, args) =
                      parse_expr_atom_list LPAREN RPAREN ps
                    in
                      (slot, Array.append stmts1 stmts2, (iter, args))
                  in
                  let (slot, stmts, call) = ctxt "stmts: foreach head"
                    (bracketed LPAREN RPAREN inner) ps
                  in
                  let body_block =
                    ctxt "stmts: foreach body" parse_block ps
                  in
                  let bpos = lexpos ps in
                  let head_block =
                    (*
                     * Slightly weird, but we put an extra nesting level of
                     * block here to separate the part that lives in our frame
                     * (the iter slot) from the part that lives in the callee
                     * frame (the body block).
                     *)
                    span ps apos bpos [|
                      span ps apos bpos (Ast.STMT_block body_block);
                    |]
                  in
                    Array.append stmts
                      [| span ps apos bpos
                           (Ast.STMT_for_each
                              { Ast.for_each_slot = slot;
                                Ast.for_each_call = call;
                                Ast.for_each_head = head_block;
                                Ast.for_each_body = body_block; }) |]
              | _ ->
                  let inner ps =
                    let slot = (parse_identified_slot_and_ident false ps) in
                    let _    = (expect ps IN) in
                      (slot, (parse_lval ps))
                  in
                  let (slot, (stmts, lval)) =
                    ctxt "stmts: for head" (bracketed LPAREN RPAREN inner) ps
                  in
                  let body_block = ctxt "stmts: for body" parse_block ps in
                  let bpos = lexpos ps in
                    Array.append stmts
                      [| span ps apos bpos
                           (Ast.STMT_for
                            { Ast.for_slot = slot;
                              Ast.for_seq = lval;
                              Ast.for_body = body_block; }) |]
          end

      | WHILE ->
          bump ps;
          let (stmts, test) =
            ctxt "stmts: while cond" (bracketed LPAREN RPAREN parse_expr) ps
          in
          let body_block = ctxt "stmts: while body" parse_block ps in
          let bpos = lexpos ps in
            [| span ps apos bpos
                 (Ast.STMT_while
                    { Ast.while_lval = (stmts, test);
                      Ast.while_body = body_block; }) |]

      | PUT ->
          begin
            bump ps;
            match peek ps with
                EACH ->
                  bump ps;
                  let (lstmts, lval) =
                    ctxt "put each: lval" (rstr true parse_lval) ps
                  in
                  let (astmts, args) =
                    ctxt "put each: args"
                      (parse_expr_atom_list LPAREN RPAREN) ps
                  in
                  let bpos = lexpos ps in
                  let be =
                    span ps apos bpos (Ast.STMT_put_each (lval, args))
                  in
                    expect ps SEMI;
                    Array.concat [ lstmts; astmts; [| be |] ]

              | _ ->
                  begin
                    let (stmts, e) =
                      match peek ps with
                          SEMI -> (arr [], None)
                        | _ ->
                            let (stmts, expr) =
                              ctxt "stmts: put expr" parse_expr_atom ps
                            in
                              expect ps SEMI;
                              (stmts, Some expr)
                    in
                      spans ps stmts apos (Ast.STMT_put e)
                  end
          end

      | RET ->
          bump ps;
          let (stmts, e) =
            match peek ps with
                SEMI -> (bump ps; (arr [], None))
              | _ ->
                  let (stmts, expr) =
                    ctxt "stmts: ret expr" parse_expr_atom ps
                  in
                    expect ps SEMI;
                    (stmts, Some expr)
          in
            spans ps stmts apos (Ast.STMT_ret e)

      | BE ->
          bump ps;
          let (lstmts, lval) = ctxt "be: lval" (rstr true parse_lval) ps in
          let (astmts, args) =
            ctxt "be: args" (parse_expr_atom_list LPAREN RPAREN) ps
          in
          let bpos = lexpos ps in
          let be = span ps apos bpos (Ast.STMT_be (lval, args)) in
            expect ps SEMI;
            Array.concat [ lstmts; astmts; [| be |] ]

      | LBRACE -> [| ctxt "stmts: block" parse_block_stmt ps |]

      | LET ->
          bump ps;
          let (stmts, slot, ident) =
            ctxt "stmt slot" parse_slot_and_ident_and_init ps in
          let slot = ensure_mutable slot in
          let bpos = lexpos ps in
          let decl = Ast.DECL_slot (Ast.KEY_ident ident,
                                    (span ps apos bpos slot))
          in
            Array.concat [[| span ps apos bpos (Ast.STMT_decl decl) |]; stmts]

      | AUTO ->
          bump ps;
          let (stmts, slot, ident) =
            ctxt "stmt slot" parse_auto_slot_and_init ps in
          let slot = ensure_mutable slot in
          let bpos = lexpos ps in
          let decl = Ast.DECL_slot (Ast.KEY_ident ident,
                                    (span ps apos bpos slot))
          in
            Array.concat [[| span ps apos bpos (Ast.STMT_decl decl) |]; stmts]

      | YIELD ->
          bump ps;
          expect ps SEMI;
          let bpos = lexpos ps in
            [| span ps apos bpos Ast.STMT_yield |]

      | FAIL ->
          bump ps;
          expect ps SEMI;
          let bpos = lexpos ps in
            [| span ps apos bpos Ast.STMT_fail |]

      | JOIN ->
          bump ps;
          let (stmts, lval) = ctxt "stmts: task expr" parse_lval ps in
            expect ps SEMI;
            spans ps stmts apos (Ast.STMT_join lval)


       | STATE | GC | NATIVE
       | MOD | OBJ | TAG | TYPE | FN | USE ->
          let items = ctxt "stmt: decl" parse_mod_item ps in
          let bpos = lexpos ps in
            Array.map
              begin
                fun (ident, item) ->
                  let decl = Ast.DECL_mod_item (ident, item) in
                    span ps apos bpos (Ast.STMT_decl decl)
              end
              items

      | token ->
          if token = SPAWN then
            prerr_endline ("warning: \"spawn\" with unused result spawns a " ^
              "task that immediately dies");

          let (lstmts, lval) = ctxt "stmt: lval" parse_lval ps in
          let stmts =
            match peek ps with

                SEMI -> (bump ps; [| |])

              | EQ -> parse_init lval ps

              | OPEQ binop_token ->
                  bump ps;
                  let (stmts, rhs) =
                    ctxt "stmt: opeq rhs" parse_expr_atom ps
                  in
                  let binop =
                    match binop_token with
                        PLUS    -> Ast.BINOP_add
                      | MINUS   -> Ast.BINOP_sub
                      | STAR    -> Ast.BINOP_mul
                      | SLASH   -> Ast.BINOP_div
                      | PERCENT -> Ast.BINOP_mod
                      | AND     -> Ast.BINOP_and
                      | OR      -> Ast.BINOP_or
                      | CARET   -> Ast.BINOP_xor
                      | LSL     -> Ast.BINOP_lsl
                      | LSR     -> Ast.BINOP_lsr
                      | ASR     -> Ast.BINOP_asr
                      | _       -> raise (err "unknown opeq token" ps)
                  in
                    expect ps SEMI;
                    spans ps stmts apos
                      (Ast.STMT_copy_binop (lval, binop, rhs))

              | LARROW ->
                  bump ps;
                  let (stmts, rhs) = ctxt "stmt: recv rhs" parse_lval ps in
                  let _ = expect ps SEMI in
                    spans ps stmts apos (Ast.STMT_recv (lval, rhs))

              | SEND ->
                  bump ps;
                  let (stmts, rhs) =
                    ctxt "stmt: send rhs" parse_expr_atom ps
                  in
                  let _ = expect ps SEMI in
                  let bpos = lexpos ps in
                  let (src, copy) = match rhs with
                      Ast.ATOM_lval lv -> (lv, [| |])
                    | _ ->
                        let (_, tmp, tempdecl) =
                          build_tmp ps slot_auto apos bpos
                        in
                        let copy = span ps apos bpos
                          (Ast.STMT_copy (tmp, Ast.EXPR_atom rhs)) in
                            ((clone_lval ps tmp), [| tempdecl; copy |])
                  in
                  let send =
                    span ps apos bpos
                      (Ast.STMT_send (lval, src))
                  in
                    Array.concat [ stmts; copy; [| send |] ]

              | _ -> raise (unexpected ps)
          in
          Array.append lstmts stmts


and parse_ty_param (iref:int ref) (ps:pstate) : Ast.ty_param identified =
  let apos = lexpos ps in
  let s = Pexp.parse_layer ps in
  let ident = Pexp.parse_ident ps in
  let i = !iref in
  let bpos = lexpos ps in
    incr iref;
    span ps apos bpos (ident, (i, s))

and parse_ty_params (ps:pstate)
    : (Ast.ty_param identified) array =
  match peek ps with
      LBRACKET ->
        bracketed_zero_or_more LBRACKET RBRACKET (Some COMMA)
          (parse_ty_param (ref 0)) ps
    | _ -> arr []

and parse_ident_and_params (ps:pstate) (cstr:string)
    : (Ast.ident * (Ast.ty_param identified) array) =
  let ident = ctxt ("mod " ^ cstr ^ " item: ident") Pexp.parse_ident ps in
  let params =
    ctxt ("mod " ^ cstr ^ " item: type params") parse_ty_params ps
  in
    (ident, params)

and parse_inputs
    (ps:pstate)
    : ((Ast.slot identified * Ast.ident) array * Ast.constrs)  =
  let slots =
    match peek ps with
        LPAREN -> ctxt "inputs: input idents and slots"
          (parse_zero_or_more_identified_slot_ident_pairs true) ps
      | _ -> raise (unexpected ps)
  in
  let constrs =
    match peek ps with
        COLON -> (bump ps; ctxt "inputs: constrs" Pexp.parse_constrs ps)
      | _ -> [| |]
  in
  let rec rewrite_carg_path cp =
    match cp with
        Ast.CARG_base (Ast.BASE_named (Ast.BASE_ident ident)) ->
          begin
            let res = ref cp in
              for i = 0 to (Array.length slots) - 1
              do
                let (_, ident') = slots.(i) in
                  if ident' = ident
                  then res := Ast.CARG_ext (Ast.CARG_base Ast.BASE_formal,
                                            Ast.COMP_idx i)
                  else ()
              done;
              !res
          end
      | Ast.CARG_base _ -> cp
      | Ast.CARG_ext (cp, ext) ->
          Ast.CARG_ext (rewrite_carg_path cp, ext)
  in
    (* Rewrite constrs with input tuple as BASE_formal. *)
    Array.iter
      begin
        fun constr ->
          let args = constr.Ast.constr_args in
            Array.iteri
              begin
                fun i carg ->
                  match carg with
                      Ast.CARG_path cp ->
                        args.(i) <- Ast.CARG_path (rewrite_carg_path cp)
                    | _ -> ()
              end
              args
      end
      constrs;
    (slots, constrs)


and parse_in_and_out
    (ps:pstate)
    : ((Ast.slot identified * Ast.ident) array
       * Ast.constrs
       * Ast.slot identified) =
  let (inputs, constrs) = parse_inputs ps in
  let output =
    match peek ps with
        RARROW ->
          bump ps;
          ctxt "fn in and out: output slot"
            (Pexp.parse_identified_slot true) ps
      | _ ->
          let apos = lexpos ps in
            span ps apos apos slot_nil
  in
    (inputs, constrs, output)


(* parse_fn starts at the first lparen of the sig. *)
and parse_fn
    (is_iter:bool)
    (ps:pstate)
    : Ast.fn =
    let (inputs, constrs, output) =
      ctxt "fn: in_and_out" parse_in_and_out ps
    in
    let body = ctxt "fn: body" parse_block ps in
      { Ast.fn_input_slots = inputs;
        Ast.fn_input_constrs = constrs;
        Ast.fn_output_slot = output;
        Ast.fn_aux = { Ast.fn_is_iter = is_iter; };
        Ast.fn_body = body; }

and parse_meta_input (ps:pstate) : (Ast.ident * string option) =
  let lab = (ctxt "meta input: label" Pexp.parse_ident ps) in
    match peek ps with
        EQ ->
          bump ps;
          let v =
            match peek ps with
                UNDERSCORE -> bump ps; None
              | LIT_STR s -> bump ps; Some s
              | _ -> raise (unexpected ps)
          in
            (lab, v)
      | _ -> raise (unexpected ps)

and parse_meta_pat (ps:pstate) : Ast.meta_pat =
  bracketed_zero_or_more LPAREN RPAREN
    (Some COMMA) parse_meta_input ps

and parse_meta (ps:pstate) : Session.meta =
  Array.map
    begin
      fun (id,v) ->
        match v with
            None ->
              raise (err ("wildcard found in meta "
                          ^ "pattern where value expected") ps)
          | Some v -> (id,v)
    end
    (parse_meta_pat ps)

and parse_optional_meta_pat (ps:pstate) (ident:Ast.ident) : Ast.meta_pat =
  match peek ps with
      LPAREN -> parse_meta_pat ps
    | _ -> [| ("name", Some ident) |]


and parse_obj_item
    (ps:pstate)
    (apos:pos)
    (layer:Ast.layer)
    : (Ast.ident * Ast.mod_item) =
  expect ps OBJ;
  let (ident, params) = parse_ident_and_params ps "obj" in
  let (state, constrs) = (ctxt "obj state" parse_inputs ps) in
  let drop = ref None in
    expect ps LBRACE;
    let fns = Hashtbl.create 0 in
      while (not (peek ps = RBRACE))
      do
        let apos = lexpos ps in
          match peek ps with
              FN | ITER ->
                let is_iter = (peek ps) = ITER in
                  bump ps;
                  let ident = ctxt "obj fn: ident" Pexp.parse_ident ps in
                  let fn = ctxt "obj fn: fn" (parse_fn is_iter) ps in
                  let bpos = lexpos ps in
                    htab_put fns ident (span ps apos bpos fn)
            | DROP ->
                bump ps;
                drop := Some (parse_block ps)
            | RBRACE -> ()
            | _ -> raise (unexpected ps)
      done;
      expect ps RBRACE;
      let bpos = lexpos ps in
      let obj = { Ast.obj_state = state;
                  Ast.obj_layer = layer;
                  Ast.obj_constrs = constrs;
                  Ast.obj_fns = fns;
                  Ast.obj_drop = !drop }
      in
        (ident,
         span ps apos bpos
           (decl params (Ast.MOD_ITEM_obj obj)))

and parse_tag_item
    (ps:pstate)
    (apos:pos)
    (layer:Ast.layer)
    : (Ast.ident * Ast.mod_item) array =
  expect ps TAG;
  let (ident, params) = parse_ident_and_params ps "tag" in
  let tag_id = next_opaque_id ps in
  let i = ref 0 in
  let parse_tag_ctor ps =
    let apos = lexpos ps in
    let ident = Pexp.parse_ident ps in
    let hdr =
      let j = ref 0 in
      let parse_ctor_slot ps =
        let apos = lexpos ps in
        let t = Pexp.parse_ty ps in
        let s = { Ast.slot_mode = Ast.MODE_local;
                  Ast.slot_ty = Some t }
        in
        let bpos = lexpos ps in
          incr j;
          ((span ps apos bpos s), "_" ^ string_of_int (!j))
      in
      let res =
        match peek ps with
            LPAREN ->
              let slots = paren_comma_list parse_ctor_slot ps in
                if Array.length slots = 0
                then
                  raise (err ("empty argument list to tag constructor") ps)
                else slots

          | _ -> [| |]
      in
        expect ps SEMI;
        res
    in
    let n = !i in
    let bpos = lexpos ps in
    let params =
      Array.map (fun p -> Parser.clone_span ps p p.node) params
    in
      incr i;
      (ident,
       span ps apos bpos
         (decl params
            (Ast.MOD_ITEM_tag (hdr, tag_id, n))))
  in
  let constructors =
    bracketed_one_or_more LBRACE RBRACE
      None (ctxt "tag: ctor" parse_tag_ctor) ps
  in
  let bpos = lexpos ps in
  let ty =
    Ast.TY_tag
      { Ast.tag_id = tag_id;
        Ast.tag_args =
          Array.map
            (fun p -> Ast.TY_param (snd p.node))
            params }
  in
  let ty_item =
    (ident,
     span ps apos bpos
       (decl params (Ast.MOD_ITEM_type (layer, ty))))
  in
    Array.append [| ty_item |] constructors

and parse_type_item
    (ps:pstate)
    (apos:pos)
    (layer:Ast.layer)
    : (Ast.ident * Ast.mod_item) =
  expect ps TYPE;
  let (ident, params) = parse_ident_and_params ps "type" in
  let _ = expect ps EQ in
  let ty = ctxt "mod type item: ty" Pexp.parse_ty ps in
  let _ = expect ps SEMI in
  let bpos = lexpos ps in
  let item = Ast.MOD_ITEM_type (layer, ty) in
    (ident, span ps apos bpos (decl params item))

and parse_mod_item (ps:pstate)
    : (Ast.ident * Ast.mod_item) array =
  let apos = lexpos ps in
  let parse_lib_name ident =
    match peek ps with
        EQ ->
          begin
            bump ps;
            let do_tok t =
              bump ps;
              match t with
                  LIT_STR s -> s
                | _ -> raise (unexpected ps)
            in
              match peek ps with
                  IDENT i ->
                    do_tok (ps.pstate_get_cenv_tok ps i)
                | t ->
                    do_tok t
          end
      | _ -> ps.pstate_infer_lib_name ident
  in


    match peek ps with

        STATE | GC
      | TYPE | OBJ | TAG | FN | ITER ->
          let layer = Pexp.parse_layer ps in
            begin
              match peek ps with
                  OBJ ->
                    [| parse_obj_item ps apos layer |]
                | TAG ->
                    parse_tag_item ps apos layer
                | TYPE ->
                    [| parse_type_item ps apos layer |]
                | _ ->
                    if layer <> Ast.LAYER_value
                    then raise (err "layer specified for fn or iter" ps);
                    let is_iter = (peek ps) = ITER in
                      bump ps;
                      let (ident, params) = parse_ident_and_params ps "fn" in
                      let fn =
                        ctxt "mod fn item: fn" (parse_fn is_iter) ps
                      in
                      let bpos = lexpos ps in
                        [| (ident,
                            span ps apos bpos
                              (decl params (Ast.MOD_ITEM_fn fn))) |]
            end

      | CONST ->
          bump ps;
          let ty = Pexp.parse_ty ps in
          let ident = Pexp.parse_ident ps in
            expect ps EQ;
            let expr = parse_prim_expr ps in
              expect ps SEMI;
              let bpos = lexpos ps in
                [| (ident, span ps apos bpos
                      (decl [||] (Ast.MOD_ITEM_const (ty, Some expr)))) |]

      | MOD ->
          bump ps;
          let (ident, params) = parse_ident_and_params ps "mod" in
            expect ps LBRACE;
            let items = parse_mod_items ps RBRACE in
            let bpos = lexpos ps in
              [| (ident,
                  span ps apos bpos
                    (decl params (Ast.MOD_ITEM_mod items))) |]

      | NATIVE ->
          begin
            bump ps;
            let conv =
              match peek ps with
                  LIT_STR s ->
                    bump ps;
                    begin
                      match string_to_conv s with
                          None -> raise (unexpected ps)
                        | Some c -> c
                    end
                | _ -> CONV_cdecl
            in
              expect ps MOD;
              let ident = Pexp.parse_ident ps in
              let path = parse_lib_name ident in
              let items = parse_native_mod_items_from_signature ps in
              let bpos = lexpos ps in
              let rlib = REQUIRED_LIB_c { required_libname = path;
                                          required_prefix = ps.pstate_depth }
              in
              let item = decl [||] (Ast.MOD_ITEM_mod items) in
              let item = span ps apos bpos item in
                note_required_mod ps {lo=apos; hi=bpos} conv rlib item;
                [| (ident, item) |]
          end
      | _ -> raise (unexpected ps)

and parse_native_mod_header_from_signature (ps:pstate) : Ast.mod_view =
  let exports = Hashtbl.create 0 in
    while (peek ps = EXPORT)
    do
      bump ps;
      parse_export ps exports;
      expect ps SEMI;
    done;
    if (Hashtbl.length exports) = 0
    then htab_put exports Ast.EXPORT_all_decls ();
    {empty_view with Ast.view_exports = exports}

and parse_native_mod_items_from_signature
    (ps:pstate)
    : (Ast.mod_view * Ast.mod_items) =
  expect ps LBRACE;
  let view  = parse_native_mod_header_from_signature ps in
  let items = Hashtbl.create 0 in
    while not (peek ps = RBRACE)
    do
      Array.iter
        (fun (ident, item) ->
           htab_put items ident item)
        (ctxt "mod items from sig: mod item"
           parse_native_mod_item_from_signature ps)
    done;
    expect ps RBRACE;
    (view,items)

and parse_native_mod_item_from_signature (ps:pstate)
    : (Ast.ident * Ast.mod_item) array =
  let apos = lexpos ps in
    match peek ps with
        MOD ->
          bump ps;
          let (ident, params) = parse_ident_and_params ps "mod signature" in
          let items = parse_native_mod_items_from_signature ps in
          let bpos = lexpos ps in
            [| (ident,
                span ps apos bpos (decl params (Ast.MOD_ITEM_mod items))) |]

      | FN | ITER ->
          let is_iter = (peek ps) = ITER in
            bump ps;
            let (ident, params) = parse_ident_and_params ps "fn signature" in
            let (inputs, constrs, output) = parse_in_and_out ps in
            let bpos = lexpos ps in
            let body = span ps apos bpos [| |] in
            let fn =
              Ast.MOD_ITEM_fn
                { Ast.fn_input_slots = inputs;
                  Ast.fn_input_constrs = constrs;
                  Ast.fn_output_slot = output;
                  Ast.fn_aux = { Ast.fn_is_iter = is_iter; };
                  Ast.fn_body = body; }
            in
            let node = span ps apos bpos (decl params fn) in
              begin
                match peek ps with
                    EQ ->
                      bump ps;
                      begin
                        match peek ps with
                            LIT_STR s ->
                              bump ps;
                              htab_put ps.pstate_required_syms node.id s
                          | _ -> raise (unexpected ps)
                      end;
                  | _ -> ()
              end;
              expect ps SEMI;
              [| (ident, node) |]

    | TYPE ->
        bump ps;
        let (ident, params) = parse_ident_and_params ps "type type" in
        let t =
          match peek ps with
              SEMI -> Ast.TY_native (next_opaque_id ps)
            | _ -> Pexp.parse_ty ps
        in
          expect ps SEMI;
          let bpos = lexpos ps in
            [| (ident, span ps apos bpos
                  (decl params (Ast.MOD_ITEM_type (Ast.LAYER_value, t)))) |]

    | _ -> raise (unexpected ps)

and note_required_mod
    (ps:pstate)
    (sp:span)
    (conv:nabi_conv)
    (rlib:required_lib)
    (item:Ast.mod_item)
    : unit =
  iflog ps
    begin
      fun _ -> log ps "marking item #%d as required" (int_of_node item.id)
    end;
  htab_put ps.pstate_required item.id (rlib, conv);
  if not (Hashtbl.mem ps.pstate_sess.Session.sess_spans item.id)
  then Hashtbl.add ps.pstate_sess.Session.sess_spans item.id sp;
  match item.node.Ast.decl_item with
      Ast.MOD_ITEM_mod (_, items) ->
        Hashtbl.iter
          begin
            fun _ sub ->
              note_required_mod ps sp conv rlib sub
          end
          items
    | _ -> ()


and parse_import
    (ps:pstate)
    (imports:(Ast.ident, Ast.name) Hashtbl.t)
    : unit =
  let import a n =
    let a = match a with
        None ->
          begin
            match n with
                Ast.NAME_ext (_, Ast.COMP_ident i)
              | Ast.NAME_ext (_, Ast.COMP_app (i, _))
              | Ast.NAME_base (Ast.BASE_ident i)
              | Ast.NAME_base (Ast.BASE_app (i, _)) -> i
              | _ -> raise (Parse_err (ps, "bad import specification"))
          end
      | Some i -> i
    in
      Hashtbl.add imports a n
  in
    match peek ps with
        IDENT i ->
          begin
            bump ps;
            match peek ps with
                EQ ->
                  (*
                   * import x = ...
                   *)
                  bump ps;
                  import (Some i) (Pexp.parse_name ps)
              | _ ->
                  (*
                   * import x...
                   *)
                  import None (Pexp.parse_name_ext ps
                                 (Ast.NAME_base
                                    (Ast.BASE_ident i)))
          end
      | _ ->
          import None (Pexp.parse_name ps)


and parse_export
    (ps:pstate)
    (exports:(Ast.export, unit) Hashtbl.t)
    : unit =
  let e =
    match peek ps with
        STAR -> bump ps; Ast.EXPORT_all_decls
      | IDENT i -> bump ps; Ast.EXPORT_ident i
      | _ -> raise (unexpected ps)
  in
    Hashtbl.add exports e ()

and parse_use
    (ps:pstate)
    : (Ast.ident * Ast.mod_item) array =
  bump ps;
  let ident = ctxt "use mod: ident" Pexp.parse_ident ps in
  let meta =
    ctxt "use mod: meta" parse_optional_meta_pat ps ident
  in
  let apos = lexpos ps in
  let bpos = lexpos ps in
  let id = (span ps apos bpos ()).id in
  let (path, items) = ps.pstate_get_mod meta id ps.pstate_crate_cache in
  let bpos = lexpos ps in
    expect ps SEMI;
    let rlib =
      REQUIRED_LIB_rust { required_libname = path;
                          required_prefix = ps.pstate_depth }
    in
      iflog ps
        begin
          fun _ ->
            log ps "extracted mod from %s (binding to %s)"
              path ident;
            log ps "%a" Ast.sprintf_mod_items items;
        end;
      let item = decl [||] (Ast.MOD_ITEM_mod (empty_view, items)) in
      let item = span ps apos bpos item in
        note_required_mod ps {lo=apos; hi=bpos} CONV_rust rlib item;
        [| (ident, item) |]

and parse_item_decl ps items fn =
  let add (id, item) =
    if Hashtbl.mem items id then
      raise (Parse_err
        (ps, (Printf.sprintf "item name already in use: '%s'" id)));
    Hashtbl.add items id item
  in
  Array.iter add (fn ps)

and parse_mod_header (ps:pstate)
                     : (Ast.mod_view * Ast.mod_items) =
  let imports = Hashtbl.create 0 in
  let exports = Hashtbl.create 0 in
  let items   = Hashtbl.create 4 in
  let rec loop () =
    match peek ps with
        IMPORT ->
          bump ps;
          parse_import ps imports;
          expect ps SEMI;
          loop ()
      | EXPORT ->
          bump ps;
          parse_export ps exports;
          expect ps SEMI;
          loop ()
      | USE ->
          parse_item_decl ps items parse_use;
          loop ()
      | _ -> ()
  in
    loop ();
    if (Hashtbl.length exports) = 0
    then Hashtbl.add exports Ast.EXPORT_all_decls ();
    let view = { Ast.view_imports = imports;
                 Ast.view_exports = exports }
    in
      (view, items)

and parse_mod_items
    (ps:pstate)
    (terminal:token)
    : (Ast.mod_view * Ast.mod_items) =
  ps.pstate_depth <- ps.pstate_depth + 1;
  let (view, items) = parse_mod_header ps in
    while (not (peek ps = terminal))
    do
      parse_item_decl ps items parse_mod_item;
    done;
    expect ps terminal;
    ps.pstate_depth <- ps.pstate_depth - 1;
    (view, items)
;;


(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
