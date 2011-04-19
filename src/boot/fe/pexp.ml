
open Common;;
open Token;;
open Parser;;

(* NB: pexps (parser-expressions) are only used transiently during
 * parsing, static-evaluation and syntax-expansion.  They're desugared
 * into the general "item" AST and/or evaluated as part of the
 * outermost "cexp" expressions. Expressions that can show up in source
 * correspond to this loose grammar and have a wide-ish flexibility in
 * *theoretical* composition; only subsets of those compositions are
 * legal in various AST contexts.
 * 
 * Desugaring on the fly is unfortunately complicated enough to require
 * -- or at least "make much more convenient" -- this two-pass
 * routine.
 *)

(* Pexp grammar. Includes names, idents, types, constrs, binops and unops,
   etc. *)

let parse_ident (ps:pstate) : Ast.ident =
  match peek ps with
      IDENT id -> (bump ps; id)
    (* Decay IDX tokens to identifiers if they occur ousdide name paths. *)
    | IDX i -> (bump ps; string_of_tok (IDX i))
    | _ -> raise (unexpected ps)
;;

(* Enforces the restricted pexp grammar when applicable (e.g. after "bind") *)
let check_rstr_start (ps:pstate) : 'a =
  if (ps.pstate_rstr) then
    match peek ps with
        IDENT _ | LPAREN -> ()
      | _ -> raise (unexpected ps)
;;

let rec parse_name_component (ps:pstate) : Ast.name_component =
  match peek ps with
      IDENT id ->
        (bump ps;
         match peek ps with
             LBRACKET ->
               let tys =
                 ctxt "name_component: apply"
                   (bracketed_one_or_more LBRACKET RBRACKET
                      (Some COMMA) parse_ty) ps
               in
                 Ast.COMP_app (id, tys)
           | _ -> Ast.COMP_ident id)

    | IDX i ->
        bump ps;
        Ast.COMP_idx i
    | _ -> raise (unexpected ps)

and parse_name_base (ps:pstate) : Ast.name_base =
  match peek ps with
      IDENT i ->
        (bump ps;
         match peek ps with
             LBRACKET ->
               let tys =
                 ctxt "name_base: apply"
                   (bracketed_one_or_more LBRACKET RBRACKET
                      (Some COMMA) parse_ty) ps
               in
                 Ast.BASE_app (i, tys)
           | _ -> Ast.BASE_ident i)
    | _ -> raise (unexpected ps)

and parse_name_ext (ps:pstate) (base:Ast.name) : Ast.name =
  match peek ps with
      DOT ->
        bump ps;
        let comps = one_or_more DOT parse_name_component ps in
          Array.fold_left (fun x y -> Ast.NAME_ext (x, y)) base comps
    | _ -> base


and parse_name (ps:pstate) : Ast.name =
  let base = Ast.NAME_base (parse_name_base ps) in
  let name = parse_name_ext ps base in
    if Ast.sane_name name
    then name
    else raise (err "malformed name" ps)

and parse_carg_base (ps:pstate) : Ast.carg_base =
  match peek ps with
      STAR -> bump ps; Ast.BASE_formal
    | _ -> Ast.BASE_named (parse_name_base ps)

and parse_carg (ps:pstate) : Ast.carg =
  match peek ps with
      IDENT _ | STAR ->
        begin
          let base = Ast.CARG_base (parse_carg_base ps) in
          let path =
            match peek ps with
                DOT ->
                  bump ps;
                  let comps = one_or_more DOT parse_name_component ps in
                    Array.fold_left
                      (fun x y -> Ast.CARG_ext (x, y)) base comps
              | _ -> base
          in
            Ast.CARG_path path
        end
    | _ ->
        Ast.CARG_lit (parse_lit ps)


and parse_constraint (ps:pstate) : Ast.constr =
  match peek ps with

      (*
       * NB: A constraint *looks* a lot like an EXPR_call, but is restricted
       * syntactically: the constraint name needs to be a name (not an lval)
       * and the constraint args all need to be cargs, which are similar to
       * names but can begin with the 'formal' base anchor '*'.
       *)

      IDENT _ ->
        let n = ctxt "constraint: name" parse_name ps in
        let args = ctxt "constraint: args"
          (bracketed_zero_or_more
             LPAREN RPAREN (Some COMMA)
             parse_carg) ps
        in
          { Ast.constr_name = n;
            Ast.constr_args = args }
    | _ -> raise (unexpected ps)


and parse_constrs (ps:pstate) : Ast.constrs =
  ctxt "state: constraints" (one_or_more COMMA parse_constraint) ps

and parse_optional_trailing_constrs (ps:pstate) : Ast.constrs =
  match peek ps with
      COLON -> (bump ps; parse_constrs ps)
    | _ -> [| |]

and parse_layer (ps:pstate) : Ast.layer =
  match peek ps with
      STATE -> bump ps; Ast.LAYER_state
    | GC -> bump ps; Ast.LAYER_gc
    |  _ -> Ast.LAYER_value

and parse_auth (ps:pstate) : Ast.auth =
  match peek ps with
    | UNSAFE -> bump ps; Ast.AUTH_unsafe
    | _ -> raise (unexpected ps)

and parse_mutability (ps:pstate) : Ast.mutability =
  match peek ps with
      MUTABLE ->
        begin
          (* HACK: ignore "mutable?" *)
          bump ps;
          match peek ps with
              QUES -> bump ps; Ast.MUT_immutable
            | _ -> Ast.MUT_mutable
        end
    | _ -> Ast.MUT_immutable

and parse_ty_fn
    (ps:pstate)
    : (Ast.ty_fn * Ast.ident option) =
  match peek ps with
      FN | ITER ->
        let is_iter = (peek ps) = ITER in
          bump ps;
          let ident =
            match peek ps with
                IDENT i -> bump ps; Some i
              | _ -> None
          in
          let in_slots =
            match peek ps with
                _ ->
                  bracketed_zero_or_more LPAREN RPAREN (Some COMMA)
                    (parse_slot_and_optional_ignored_ident true) ps
          in
          let out_slot =
            match peek ps with
                RARROW -> (bump ps; parse_slot false ps)
              | _ -> slot_nil
          in
          let constrs = parse_optional_trailing_constrs ps in
          let tsig = { Ast.sig_input_slots = in_slots;
                       Ast.sig_input_constrs = constrs;
                       Ast.sig_output_slot = out_slot; }
          in
          let taux = { Ast.fn_is_iter = is_iter; }
          in
          let tfn = (tsig, taux) in
            (tfn, ident)

    | _ -> raise (unexpected ps)

and check_dup_rec_labels ps labels =
  arr_check_dups labels
    (fun l _ ->
       raise (err (Printf.sprintf
                     "duplicate record label: %s" l) ps));


and parse_atomic_ty (ps:pstate) : Ast.ty =
  match peek ps with

      BOOL ->
        bump ps;
        Ast.TY_bool

    | INT ->
        bump ps;
        Ast.TY_int

    | UINT ->
        bump ps;
        Ast.TY_uint

    | CHAR ->
        bump ps;
        Ast.TY_char

    | STR ->
        bump ps;
        Ast.TY_str

    | ANY ->
        bump ps;
        Ast.TY_any

    | TASK ->
        bump ps;
        Ast.TY_task

    | CHAN ->
        bump ps;
        Ast.TY_chan (bracketed LBRACKET RBRACKET parse_ty ps)

    | PORT ->
        bump ps;
        Ast.TY_port (bracketed LBRACKET RBRACKET parse_ty ps)

    | VEC ->
        bump ps;
        Ast.TY_vec (bracketed LBRACKET RBRACKET parse_ty ps)

    | IDENT _ -> Ast.TY_named (parse_name ps)

    | REC ->
        bump ps;
        let parse_rec_entry ps =
          let (ty, ident) = parse_ty_and_ident ps in
            (ident, ty)
        in
        let entries = paren_comma_list parse_rec_entry ps in
        let labels = Array.map (fun (l, _) -> l) entries in
          begin
            check_dup_rec_labels ps labels;
            Ast.TY_rec entries
          end

    | TUP ->
        bump ps;
        let tys = paren_comma_list parse_ty ps in
          Ast.TY_tup tys

    | MACH m ->
        bump ps;
        Ast.TY_mach m

    | STATE | GC | UNSAFE | OBJ | FN | ITER ->
        let layer = parse_layer ps in
          begin
            match peek ps with
                OBJ ->
                  bump ps;
                  let methods = Hashtbl.create 0 in
                  let parse_method ps =
                    let (tfn, ident) = parse_ty_fn ps in
                      expect ps SEMI;
                      match ident with
                          None ->
                            raise (err (Printf.sprintf
                                          "missing method identifier") ps)
                        | Some i -> htab_put methods i tfn
                  in
                    ignore (bracketed_zero_or_more LBRACE RBRACE
                              None parse_method ps);
                    Ast.TY_obj (layer, methods)

              | FN | ITER ->
                  if layer <> Ast.LAYER_value
                  then raise (err "layer specified for fn or iter" ps);
                  Ast.TY_fn (fst (parse_ty_fn ps))
              | _ -> raise (unexpected ps)
          end

    | AT ->
        bump ps;
        Ast.TY_box (parse_ty ps)

    | MUTABLE ->
        bump ps;
        begin
          (* HACK: ignore "mutable?" *)
          match peek ps with
              QUES -> bump ps; parse_ty ps
            | _ -> Ast.TY_mutable (parse_ty ps)
        end

    | LPAREN ->
        begin
          bump ps;
          match peek ps with
              RPAREN ->
                bump ps;
                Ast.TY_nil
            | _ ->
                let t = parse_ty ps in
                  expect ps RPAREN;
                  t
        end

    | _ -> raise (unexpected ps)

and flag (ps:pstate) (tok:token) : bool =
  if peek ps = tok
  then (bump ps; true)
  else false

and parse_slot (aliases_ok:bool) (ps:pstate) : Ast.slot =
  let mode =
  match (peek ps, aliases_ok) with
      (AND, true) -> bump ps; Ast.MODE_alias
    | (AND, false) -> raise (err "alias slot in prohibited context" ps)
    | _ -> Ast.MODE_local
  in
  let ty = parse_ty ps in
    { Ast.slot_mode = mode;
      Ast.slot_ty = Some ty }

and parse_slot_and_ident
    (aliases_ok:bool)
    (ps:pstate)
    : (Ast.slot * Ast.ident) =
  let slot = ctxt "slot and ident: slot" (parse_slot aliases_ok) ps in
  let ident = ctxt "slot and ident: ident" parse_ident ps in
    (slot, ident)

and parse_ty_and_ident
    (ps:pstate)
    : (Ast.ty * Ast.ident) =
  let ty = ctxt "ty and ident: ty" parse_ty ps in
  let ident = ctxt "ty and ident: ident" parse_ident ps in
    (ty, ident)

and parse_slot_and_optional_ignored_ident
    (aliases_ok:bool)
    (ps:pstate)
    : Ast.slot =
  let slot = parse_slot aliases_ok ps in
    begin
      match peek ps with
          IDENT _ -> bump ps
        | _ -> ()
    end;
    slot

and parse_identified_slot
    (aliases_ok:bool)
    (ps:pstate)
    : Ast.slot identified =
  let apos = lexpos ps in
  let slot = parse_slot aliases_ok ps in
  let bpos = lexpos ps in
    span ps apos bpos slot

and parse_constrained_ty (ps:pstate) : Ast.ty =
  let base = ctxt "ty: base" parse_atomic_ty ps in
    match peek ps with
        COLON ->
          bump ps;
          let constrs = ctxt "ty: constrs" parse_constrs ps in
            Ast.TY_constrained (base, constrs)

      | _ -> base

and parse_ty (ps:pstate) : Ast.ty =
  parse_constrained_ty ps


and parse_rec_input (ps:pstate)
    : (Ast.ident * Ast.mutability * Ast.pexp) =
  let mutability = parse_mutability ps in
  let lab = (ctxt "rec input: label" parse_ident ps) in
    match peek ps with
        EQ ->
          bump ps;
          let pexp = ctxt "rec input: expr" parse_pexp ps in
            (lab, mutability, pexp)
      | _ -> raise (unexpected ps)


and parse_rec_body (ps:pstate) : Ast.pexp' =
  begin
    expect ps LPAREN;
    match peek ps with
        RPAREN -> Ast.PEXP_rec ([||], None)
      | WITH -> raise (err "empty record extension" ps)
      | _ ->
          let inputs = one_or_more COMMA parse_rec_input ps in
          let labels = Array.map (fun (l, _, _) -> l) inputs in
            begin
              check_dup_rec_labels ps labels;
              match peek ps with
                  RPAREN -> (bump ps; Ast.PEXP_rec (inputs, None))
                | WITH ->
                    begin
                      bump ps;
                      let base =
                        ctxt "rec input: extension base"
                          parse_pexp ps
                      in
                        expect ps RPAREN;
                        Ast.PEXP_rec (inputs, Some base)
                    end
                | _ -> raise (err "expected 'with' or ')'" ps)
            end
  end


and parse_lit (ps:pstate) : Ast.lit =
  match peek ps with
      LIT_INT i -> (bump ps; Ast.LIT_int i)
    | LIT_UINT i -> (bump ps; Ast.LIT_uint i)
    | LIT_MACH_INT (tm, i) -> (bump ps; Ast.LIT_mach_int (tm, i))
    | LIT_CHAR c -> (bump ps; Ast.LIT_char c)
    | LIT_BOOL b -> (bump ps; Ast.LIT_bool b)
    | _ -> raise (unexpected ps)


and parse_bottom_pexp (ps:pstate) : Ast.pexp =
  check_rstr_start ps;
  let apos = lexpos ps in
  match peek ps with

      AT ->
        bump ps;
        let mutability = parse_mutability ps in
        let inner = parse_pexp ps in
        let bpos = lexpos ps in
          span ps apos bpos (Ast.PEXP_box (mutability, inner))

    | TUP ->
        bump ps;
        let pexps =
          ctxt "paren pexps(s)" (rstr false parse_mutable_and_pexp_list) ps
        in
        let bpos = lexpos ps in
          span ps apos bpos (Ast.PEXP_tup pexps)

    | REC ->
          bump ps;
          let body = ctxt "rec pexp: rec body" parse_rec_body ps in
          let bpos = lexpos ps in
            span ps apos bpos body

    | VEC ->
        bump ps;
        let pexps =
          ctxt "paren pexps(s)" (rstr false parse_mutable_and_pexp_list) ps
        in
        let mutability = ref Ast.MUT_immutable in
        let pexps =
          Array.mapi
            begin
              fun i (mut, e) ->
                if i = 0
                then
                  mutability := mut
                else
                  if mut <> Ast.MUT_immutable
                  then
                    raise
                      (err "'mutable' keyword after first vec element" ps);
                e
            end
            pexps
        in
        let bpos = lexpos ps in
          span ps apos bpos (Ast.PEXP_vec (!mutability, pexps))


    | LIT_STR s ->
        bump ps;
        let bpos = lexpos ps in
          span ps apos bpos (Ast.PEXP_str s)

    | PORT ->
        begin
            bump ps;
            expect ps LPAREN;
            expect ps RPAREN;
            let bpos = lexpos ps in
              span ps apos bpos (Ast.PEXP_port)
        end

    | CHAN ->
        begin
            bump ps;
            let port =
              match peek ps with
                  LPAREN ->
                    begin
                      bump ps;
                      match peek ps with
                          RPAREN -> (bump ps; None)
                        | _ ->
                            let lv = parse_pexp ps in
                              expect ps RPAREN;
                              Some lv
                    end
                | _ -> raise (unexpected ps)
            in
            let bpos = lexpos ps in
              span ps apos bpos (Ast.PEXP_chan port)
        end

    | SPAWN ->
        bump ps;
        let domain =
          match peek ps with
              THREAD -> bump ps; Ast.DOMAIN_thread
            | _ -> Ast.DOMAIN_local
        in
          (* Spawns either have an explicit literal string for the spawned
             task's name, or the task is named as the entry call
             expression. *)
        let explicit_name =
          match peek ps with
              LIT_STR s -> bump ps; Some s
            | _ -> None
        in
        let pexp =
          ctxt "spawn [domain] [name] pexp: init call" parse_pexp ps
        in
        let bpos = lexpos ps in
        let name =
          match explicit_name with
              Some s -> s
                (* FIXME: string_of_span returns a string like
                   "./driver.rs:10:16 - 11:52", not the actual text at those
                   characters *)
            | None -> Session.string_of_span { lo = apos; hi = bpos }
        in
          span ps apos bpos (Ast.PEXP_spawn (domain, name, pexp))

    | BIND ->
        let apos = lexpos ps in
          begin
            bump ps;
            let pexp = ctxt "bind pexp: function" (rstr true parse_pexp) ps in
            let args =
              ctxt "bind args"
                (paren_comma_list parse_bind_arg) ps
            in
            let bpos = lexpos ps in
              span ps apos bpos (Ast.PEXP_bind (pexp, args))
          end

    | IDENT i ->
        begin
          bump ps;
          match peek ps with
              LBRACKET ->
                begin
                  let tys =
                    ctxt "apply-type expr"
                      (bracketed_one_or_more LBRACKET RBRACKET
                         (Some COMMA) parse_ty) ps
                  in
                  let bpos = lexpos ps in
                    span ps apos bpos
                      (Ast.PEXP_lval (Ast.PLVAL_base (Ast.BASE_app (i, tys))))
                end

            | _ ->
                begin
                  let bpos = lexpos ps in
                    span ps apos bpos
                      (Ast.PEXP_lval (Ast.PLVAL_base (Ast.BASE_ident i)))
                end
        end


    | STAR ->
        bump ps;
        let inner = parse_pexp ps in
        let bpos = lexpos ps in
          span ps apos bpos (Ast.PEXP_lval (Ast.PLVAL_ext_deref inner))

    | POUND ->
        bump ps;
        let name = parse_name ps in
        let args =
          match peek ps with
              LPAREN ->
                parse_pexp_list ps
            | _ -> [| |]
        in
        let str =
          match peek ps with
              LBRACE ->
                begin
                  bump_bracequote ps;
                  match peek ps with
                      BRACEQUOTE s -> bump ps; Some s
                    | _ -> raise (unexpected ps)
                end
            | _ -> None
        in
        let bpos = lexpos ps in
          span ps apos bpos
            (Ast.PEXP_custom (name, args, str))

    | LPAREN ->
        begin
          bump ps;
          match peek ps with
              RPAREN ->
                bump ps;
                let bpos = lexpos ps in
                  span ps apos bpos (Ast.PEXP_lit Ast.LIT_nil)
            | _ ->
                let pexp = parse_pexp ps in
                  expect ps RPAREN;
                  pexp
        end

    | _ ->
        let lit = parse_lit ps in
        let bpos = lexpos ps in
          span ps apos bpos (Ast.PEXP_lit lit)


and parse_bind_arg (ps:pstate) : Ast.pexp option =
  match peek ps with
      UNDERSCORE -> (bump ps; None)
    | _ -> Some (parse_pexp ps)


and parse_ext_pexp (ps:pstate) (pexp:Ast.pexp) : Ast.pexp =
  let apos = lexpos ps in
    match peek ps with
        LPAREN ->
          if ps.pstate_rstr
          then pexp
          else
            let args = parse_pexp_list ps in
            let bpos = lexpos ps in
            let ext = span ps apos bpos (Ast.PEXP_call (pexp, args)) in
              parse_ext_pexp ps ext

      | DOT ->
          begin
            bump ps;
            let ext =
              match peek ps with
                  LPAREN ->
                    bump ps;
                    let rhs = rstr false parse_pexp ps in
                      expect ps RPAREN;
                      let bpos = lexpos ps in
                        span ps apos bpos
                          (Ast.PEXP_lval (Ast.PLVAL_ext_pexp (pexp, rhs)))
                | _ ->
                    let rhs = parse_name_component ps in
                    let bpos = lexpos ps in
                      span ps apos bpos
                        (Ast.PEXP_lval (Ast.PLVAL_ext_name (pexp, rhs)))
            in
              parse_ext_pexp ps ext
          end

      | _ -> pexp


and parse_negation_pexp (ps:pstate) : Ast.pexp =
    let apos = lexpos ps in
      match peek ps with
          NOT ->
            bump ps;
            let rhs = ctxt "negation pexp" parse_negation_pexp ps in
            let bpos = lexpos ps in
              span ps apos bpos (Ast.PEXP_unop (Ast.UNOP_not, rhs))

        | TILDE ->
            bump ps;
            let rhs = ctxt "negation pexp" parse_negation_pexp ps in
            let bpos = lexpos ps in
              span ps apos bpos (Ast.PEXP_unop (Ast.UNOP_bitnot, rhs))

        | MINUS ->
            bump ps;
            let rhs = ctxt "negation pexp" parse_negation_pexp ps in
            let bpos = lexpos ps in
              span ps apos bpos (Ast.PEXP_unop (Ast.UNOP_neg, rhs))

        | _ ->
            let lhs = parse_bottom_pexp ps in
              parse_ext_pexp ps lhs


(* Binops are all left-associative,                *)
(* so we factor out some of the parsing code here. *)
and binop_build
    (ps:pstate)
    (name:string)
    (apos:pos)
    (rhs_parse_fn:pstate -> Ast.pexp)
    (lhs:Ast.pexp)
    (step_fn:Ast.pexp -> Ast.pexp)
    (op:Ast.binop)
    : Ast.pexp =
  bump ps;
  let rhs = (ctxt (name ^ " rhs") rhs_parse_fn ps) in
  let bpos = lexpos ps in
  let node = span ps apos bpos (Ast.PEXP_binop (op, lhs, rhs)) in
    step_fn node


and parse_factor_pexp (ps:pstate) : Ast.pexp =
  let name = "factor pexp" in
  let apos = lexpos ps in
  let lhs = ctxt (name ^ " lhs") parse_negation_pexp ps in
  let build = binop_build ps name apos parse_negation_pexp in
  let rec step accum =
    match peek ps with
        STAR    -> build accum step Ast.BINOP_mul
      | SLASH   -> build accum step Ast.BINOP_div
      | PERCENT -> build accum step Ast.BINOP_mod
      | _       -> accum
  in
    step lhs


and parse_term_pexp (ps:pstate) : Ast.pexp =
  let name = "term pexp" in
  let apos = lexpos ps in
  let lhs = ctxt (name ^ " lhs") parse_factor_pexp ps in
  let build = binop_build ps name apos parse_factor_pexp in
  let rec step accum =
    match peek ps with
        PLUS  -> build accum step Ast.BINOP_add
      | MINUS -> build accum step Ast.BINOP_sub
      | _     -> accum
  in
    step lhs


and parse_shift_pexp (ps:pstate) : Ast.pexp =
  let name = "shift pexp" in
  let apos = lexpos ps in
  let lhs = ctxt (name ^ " lhs") parse_term_pexp ps in
  let build = binop_build ps name apos parse_term_pexp in
  let rec step accum =
    match peek ps with
        LSL -> build accum step Ast.BINOP_lsl
      | LSR -> build accum step Ast.BINOP_lsr
      | ASR -> build accum step Ast.BINOP_asr
      | _   -> accum
  in
    step lhs


and parse_and_pexp (ps:pstate) : Ast.pexp =
  let name = "and pexp" in
  let apos = lexpos ps in
  let lhs = ctxt (name ^ " lhs") parse_shift_pexp ps in
  let build = binop_build ps name apos parse_shift_pexp in
  let rec step accum =
    match peek ps with
        AND -> build accum step Ast.BINOP_and
      | _   -> accum
  in
    step lhs


and parse_xor_pexp (ps:pstate) : Ast.pexp =
  let name = "xor pexp" in
  let apos = lexpos ps in
  let lhs = ctxt (name ^ " lhs") parse_and_pexp ps in
  let build = binop_build ps name apos parse_and_pexp in
  let rec step accum =
    match peek ps with
        CARET -> build accum step Ast.BINOP_xor
      | _     -> accum
  in
    step lhs


and parse_or_pexp (ps:pstate) : Ast.pexp =
  let name = "or pexp" in
  let apos = lexpos ps in
  let lhs = ctxt (name ^ " lhs") parse_xor_pexp ps in
  let build = binop_build ps name apos parse_xor_pexp in
  let rec step accum =
    match peek ps with
        OR -> build accum step Ast.BINOP_or
      | _  -> accum
  in
    step lhs


and parse_as_pexp (ps:pstate) : Ast.pexp =
  let apos = lexpos ps in
  let pexp = ctxt "as pexp" parse_or_pexp ps in
  let rec step accum =
    match peek ps with
        AS ->
          bump ps;
          let tapos = lexpos ps in
          let t = parse_ty ps in
          let bpos = lexpos ps in
          let t = span ps tapos bpos t in
          let node =
            span ps apos bpos
              (Ast.PEXP_unop ((Ast.UNOP_cast t), accum))
          in
            step node

      | _ -> accum
  in
    step pexp


and parse_relational_pexp (ps:pstate) : Ast.pexp =
  let name = "relational pexp" in
  let apos = lexpos ps in
  let lhs = ctxt (name ^ " lhs") parse_as_pexp ps in
  let build = binop_build ps name apos parse_as_pexp in
  let rec step accum =
    match peek ps with
        LT -> build accum step Ast.BINOP_lt
      | LE -> build accum step Ast.BINOP_le
      | GE -> build accum step Ast.BINOP_ge
      | GT -> build accum step Ast.BINOP_gt
      | _  -> accum
  in
    step lhs


and parse_equality_pexp (ps:pstate) : Ast.pexp =
  let name = "equality pexp" in
  let apos = lexpos ps in
  let lhs = ctxt (name ^ " lhs") parse_relational_pexp ps in
  let build = binop_build ps name apos parse_relational_pexp in
  let rec step accum =
    match peek ps with
        EQEQ -> build accum step Ast.BINOP_eq
      | NE   -> build accum step Ast.BINOP_ne
      | _    -> accum
  in
    step lhs


and parse_andand_pexp (ps:pstate) : Ast.pexp =
  let name = "andand pexp" in
  let apos = lexpos ps in
  let lhs = ctxt (name ^ " lhs") parse_equality_pexp ps in
  let rec step accum =
    match peek ps with
        ANDAND ->
          bump ps;
          let rhs = parse_equality_pexp ps in
          let bpos = lexpos ps in
          let node = span ps apos bpos (Ast.PEXP_lazy_and (accum, rhs)) in
            step node

      | _   -> accum
  in
    step lhs


and parse_oror_pexp (ps:pstate) : Ast.pexp =
  let name = "oror pexp" in
  let apos = lexpos ps in
  let lhs = ctxt (name ^ " lhs") parse_andand_pexp ps in
  let rec step accum =
    match peek ps with
        OROR ->
          bump ps;
          let rhs = parse_andand_pexp ps in
          let bpos = lexpos ps in
          let node = span ps apos bpos (Ast.PEXP_lazy_or (accum, rhs)) in
            step node

      | _  -> accum
  in
    step lhs


and parse_pexp (ps:pstate) : Ast.pexp =
  parse_oror_pexp ps

and parse_mutable_and_pexp (ps:pstate) : (Ast.mutability * Ast.pexp) =
  let mutability = parse_mutability ps in
  (mutability, parse_as_pexp ps)

and parse_pexp_list (ps:pstate) : Ast.pexp array =
  match peek ps with
      LPAREN ->
        bracketed_zero_or_more LPAREN RPAREN (Some COMMA)
          (ctxt "pexp list" parse_pexp) ps
    | _ -> raise (unexpected ps)

and parse_mutable_and_pexp_list (ps:pstate)
    : (Ast.mutability * Ast.pexp) array =
  match peek ps with
      LPAREN ->
        bracketed_zero_or_more LPAREN RPAREN (Some COMMA)
          (ctxt "mutable-and-pexp list" parse_mutable_and_pexp) ps
    | _ -> raise (unexpected ps)

;;

(* 
 * Desugarings depend on context:
 * 
 *   - If a pexp is used on the RHS of an assignment, it's turned into
 *     an initialization statement such as STMT_new_rec or such. This
 *     removes the possibility of initializing into a temp only to
 *     copy out. If the topmost pexp in such a desugaring is an atom,
 *     unop or binop, of course, it will still just emit a STMT_copy
 *     on a primitive expression.
 * 
 *   - If a pexp is used in the context where an atom is required, a 
 *     statement declaring a temporary and initializing it with the 
 *     result of the pexp is prepended, and the temporary atom is used.
 *)

let rec desugar_lval (ps:pstate) (pexp:Ast.pexp)
    : (Ast.stmt array * Ast.lval) =
  let s = Hashtbl.find ps.pstate_sess.Session.sess_spans pexp.id in
  let (apos, bpos) = (s.lo, s.hi) in
    match pexp.node with

        Ast.PEXP_lval (Ast.PLVAL_base nb) ->
            ([||], Ast.LVAL_base (span ps apos bpos nb))

      | Ast.PEXP_lval (Ast.PLVAL_ext_name (base_pexp, comp)) ->
          let (base_stmts, base_atom) = desugar_expr_atom ps base_pexp in
          let base_lval = atom_lval ps base_atom in
            (base_stmts, Ast.LVAL_ext (base_lval, Ast.COMP_named comp))

      | Ast.PEXP_lval (Ast.PLVAL_ext_pexp (base_pexp, ext_pexp)) ->
          let (base_stmts, base_atom) = desugar_expr_atom ps base_pexp in
          let (ext_stmts, ext_atom) = desugar_expr_atom ps ext_pexp in
          let base_lval = atom_lval ps base_atom in
            (Array.append base_stmts ext_stmts,
             Ast.LVAL_ext (base_lval, Ast.COMP_atom (clone_atom ps ext_atom)))

      | Ast.PEXP_lval (Ast.PLVAL_ext_deref base_pexp) ->
          let (base_stmts, base_atom) = desugar_expr_atom ps base_pexp in
          let base_lval = atom_lval ps base_atom in
            (base_stmts, Ast.LVAL_ext (base_lval, Ast.COMP_deref))

      | _ ->
          let (stmts, atom) = desugar_expr_atom ps pexp in
            (stmts, atom_lval ps atom)


and desugar_expr
    (ps:pstate)
    (pexp:Ast.pexp)
    : (Ast.stmt array * Ast.expr) =
  match pexp.node with

      Ast.PEXP_unop (op, pe) ->
        let (stmts, at) = desugar_expr_atom ps pe in
          (stmts, Ast.EXPR_unary (op, at))

    | Ast.PEXP_binop (op, lhs, rhs) ->
          let (lhs_stmts, lhs_atom) = desugar_expr_atom ps lhs in
          let (rhs_stmts, rhs_atom) = desugar_expr_atom ps rhs in
            (Array.append lhs_stmts rhs_stmts,
             Ast.EXPR_binary (op, lhs_atom, rhs_atom))

    | _ ->
        let (stmts, at) = desugar_expr_atom ps pexp in
          (stmts, Ast.EXPR_atom at)


and desugar_opt_expr_atom
    (ps:pstate)
    (po:Ast.pexp option)
    : (Ast.stmt array * Ast.atom option) =
  match po with
      None -> ([| |], None)
    | Some pexp ->
        let (stmts, atom) = desugar_expr_atom ps pexp in
          (stmts, Some atom)


and desugar_expr_atom
    (ps:pstate)
    (pexp:Ast.pexp)
    : (Ast.stmt array * Ast.atom) =
  let s = Hashtbl.find ps.pstate_sess.Session.sess_spans pexp.id in
  let (apos, bpos) = (s.lo, s.hi) in
    match pexp.node with

        Ast.PEXP_unop _
      | Ast.PEXP_binop _
      | Ast.PEXP_lazy_or _
      | Ast.PEXP_lazy_and _
      | Ast.PEXP_rec _
      | Ast.PEXP_tup _
      | Ast.PEXP_str _
      | Ast.PEXP_vec _
      | Ast.PEXP_port
      | Ast.PEXP_chan _
      | Ast.PEXP_call _
      | Ast.PEXP_bind _
      | Ast.PEXP_spawn _
      | Ast.PEXP_custom _
      | Ast.PEXP_box _ ->
          let (_, tmp, decl_stmt) = build_tmp ps slot_auto apos bpos in
          let stmts = desugar_expr_init ps tmp pexp in
            (Array.append [| decl_stmt |] stmts,
             Ast.ATOM_lval (clone_lval ps tmp))

      | Ast.PEXP_lit lit ->
          ([||], Ast.ATOM_literal (span ps apos bpos lit))

      | Ast.PEXP_lval _ ->
          let (stmts, lval) = desugar_lval ps pexp in
            (stmts, Ast.ATOM_lval lval)

and desugar_expr_atoms
    (ps:pstate)
    (pexps:Ast.pexp array)
    : (Ast.stmt array * Ast.atom array) =
  arj1st (Array.map (desugar_expr_atom ps) pexps)

and desugar_opt_expr_atoms
    (ps:pstate)
    (pexps:Ast.pexp option array)
    : (Ast.stmt array * Ast.atom option array) =
  arj1st (Array.map (desugar_opt_expr_atom ps) pexps)

and desugar_expr_init
    (ps:pstate)
    (dst_lval:Ast.lval)
    (pexp:Ast.pexp)
    : (Ast.stmt array) =
  let s = Hashtbl.find ps.pstate_sess.Session.sess_spans pexp.id in
  let (apos, bpos) = (s.lo, s.hi) in

  (* Helpers. *)
  let ss x = span ps apos bpos x in
  let cp v = Ast.STMT_copy (clone_lval ps dst_lval, v) in
  let aa x y = Array.append x y in
  let ac xs = Array.concat xs in

    match pexp.node with

        Ast.PEXP_lit _
      | Ast.PEXP_lval _ ->
          let (stmts, atom) = desugar_expr_atom ps pexp in
            aa stmts [| ss (cp (Ast.EXPR_atom atom)) |]

      | Ast.PEXP_binop (op, lhs, rhs) ->
          let (lhs_stmts, lhs_atom) = desugar_expr_atom ps lhs in
          let (rhs_stmts, rhs_atom) = desugar_expr_atom ps rhs in
          let copy_stmt =
            ss (cp (Ast.EXPR_binary (op, lhs_atom, rhs_atom)))
          in
            ac [ lhs_stmts; rhs_stmts; [| copy_stmt |] ]

      (* x = a && b ==> if (a) { x = b; } else { x = false; } *)

      | Ast.PEXP_lazy_and (lhs, rhs) ->
          let (lhs_stmts, lhs_atom) = desugar_expr_atom ps lhs in
          let (rhs_stmts, rhs_atom) = desugar_expr_atom ps rhs in
          let sthen =
            ss (aa rhs_stmts [| ss (cp (Ast.EXPR_atom rhs_atom)) |])
          in
          let selse =
            ss [| ss (cp (Ast.EXPR_atom
                            (Ast.ATOM_literal (ss (Ast.LIT_bool false))))) |]
          in
          let sif =
            ss (Ast.STMT_if { Ast.if_test = Ast.EXPR_atom lhs_atom;
                              Ast.if_then = sthen;
                              Ast.if_else = Some selse })
          in
            aa lhs_stmts [| sif |]

      (* x = a || b ==> if (a) { x = true; } else { x = b; } *)

      | Ast.PEXP_lazy_or (lhs, rhs) ->
          let (lhs_stmts, lhs_atom) = desugar_expr_atom ps lhs in
          let (rhs_stmts, rhs_atom) = desugar_expr_atom ps rhs in
          let sthen =
            ss [| ss (cp (Ast.EXPR_atom
                            (Ast.ATOM_literal (ss (Ast.LIT_bool true))))) |]
          in
          let selse =
            ss (aa rhs_stmts [| ss (cp (Ast.EXPR_atom rhs_atom)) |])
          in
          let sif =
            ss (Ast.STMT_if { Ast.if_test = Ast.EXPR_atom lhs_atom;
                              Ast.if_then = sthen;
                              Ast.if_else = Some selse })
          in
            aa lhs_stmts [| sif |]


      | Ast.PEXP_unop (op, rhs) ->
          let (rhs_stmts, rhs_atom) = desugar_expr_atom ps rhs in
          let expr = Ast.EXPR_unary (op, rhs_atom) in
          let copy_stmt = ss (cp expr) in
            aa rhs_stmts [| copy_stmt |]

      | Ast.PEXP_call (fn, args) ->
          let (fn_stmts, fn_atom) = desugar_expr_atom ps fn in
          let (arg_stmts, arg_atoms) = desugar_expr_atoms ps args in
          let fn_lval = atom_lval ps fn_atom in
          let call_stmt = ss (Ast.STMT_call (dst_lval, fn_lval, arg_atoms)) in
            ac [ fn_stmts; arg_stmts; [| call_stmt |] ]

      | Ast.PEXP_bind (fn, args) ->
          let (fn_stmts, fn_atom) = desugar_expr_atom ps fn in
          let (arg_stmts, arg_atoms) = desugar_opt_expr_atoms ps args in
          let fn_lval = atom_lval ps fn_atom in
          let bind_stmt = ss (Ast.STMT_bind (dst_lval, fn_lval, arg_atoms)) in
            ac [ fn_stmts; arg_stmts; [| bind_stmt |] ]

      | Ast.PEXP_spawn (domain, name, sub) ->
          begin
            match sub.node with
                Ast.PEXP_call (fn, args) ->
                  let (fn_stmts, fn_atom) = desugar_expr_atom ps fn in
                  let (arg_stmts, arg_atoms) = desugar_expr_atoms ps args in
                  let fn_lval = atom_lval ps fn_atom in
                  let spawn_stmt =
                    ss (Ast.STMT_spawn
                          (dst_lval, domain, name, fn_lval, arg_atoms))
                  in
                    ac [ fn_stmts; arg_stmts; [| spawn_stmt |] ]
              | _ -> raise (err "non-call spawn" ps)
          end

      | Ast.PEXP_rec (args, base) ->
          let (arg_stmts, entries) =
            arj1st
              begin
                Array.map
                  begin
                    fun (ident, mutability, pexp) ->
                      let (stmts, atom) =
                        desugar_expr_atom ps pexp
                      in
                        (stmts, (ident, mutability, atom))
                  end
                  args
              end
          in
            begin
              match base with
                  Some base ->
                    let (base_stmts, base_lval) = desugar_lval ps base in
                    let rec_stmt =
                      ss (Ast.STMT_new_rec
                            (dst_lval, entries, Some base_lval))
                    in
                      ac [ arg_stmts; base_stmts; [| rec_stmt |] ]
                | None ->
                    let rec_stmt =
                      ss (Ast.STMT_new_rec (dst_lval, entries, None))
                    in
                      aa arg_stmts [| rec_stmt |]
            end

      | Ast.PEXP_tup args ->
          let muts = Array.to_list (Array.map fst args) in
          let (arg_stmts, arg_atoms) =
            desugar_expr_atoms ps (Array.map snd args)
          in
          let arg_atoms = Array.to_list arg_atoms in
          let tup_args = Array.of_list (List.combine muts arg_atoms) in
          let stmt = ss (Ast.STMT_new_tup (dst_lval, tup_args)) in
            aa arg_stmts [| stmt |]

      | Ast.PEXP_str s ->
          let stmt = ss (Ast.STMT_new_str (dst_lval, s)) in
            [| stmt |]

      | Ast.PEXP_vec (mutability, args) ->
          let (arg_stmts, arg_atoms) = desugar_expr_atoms ps args in
          let stmt =
            ss (Ast.STMT_new_vec (dst_lval, mutability, arg_atoms))
          in
            aa arg_stmts [| stmt |]

      | Ast.PEXP_port ->
          [| ss (Ast.STMT_new_port dst_lval) |]

      | Ast.PEXP_chan pexp_opt ->
          let (port_stmts, port_opt) =
            match pexp_opt with
                None -> ([||], None)
              | Some port_pexp ->
                  begin
                    let (port_stmts, port_atom) =
                      desugar_expr_atom ps port_pexp
                    in
                    let port_lval = atom_lval ps port_atom in
                      (port_stmts, Some port_lval)
                  end
          in
          let chan_stmt =
            ss
              (Ast.STMT_new_chan (dst_lval, port_opt))
          in
            aa port_stmts [| chan_stmt |]

      | Ast.PEXP_box (mutability, arg) ->
          let (arg_stmts, arg_mode_atom) =
            desugar_expr_atom ps arg
          in
          let stmt =
            ss (Ast.STMT_new_box (dst_lval, mutability, arg_mode_atom))
          in
            aa arg_stmts [| stmt |]

      | Ast.PEXP_custom (n, a, b) ->
          expand_pexp_custom ps apos bpos dst_lval n a b

(* 
 * FIXME: This is a crude approximation of the syntax-extension system,
 * for purposes of prototyping and/or hard-wiring any extensions we
 * wish to use in the bootstrap compiler. The eventual aim is to permit
 * loading rust crates to process extensions, but this will likely
 * require a rust-based frontend, or an ocaml-FFI-based connection to
 * rust crates. At the moment we have neither.
 *)

and expand_pexp_custom
    (ps:pstate)
    (apos:pos)
    (bpos:pos)
    (dst_lval:Ast.lval)
    (name:Ast.name)
    (pexp_args:Ast.pexp array)
    (body:string option)
    : (Ast.stmt array) =
  let nstr = Fmt.fmt_to_str Ast.fmt_name name in
    match (nstr, (Array.length pexp_args), body) with

        ("shell", 0, Some cmd) ->
          let c = Unix.open_process_in cmd in
          let b = Buffer.create 32 in
          let rec r _ =
            try
              Buffer.add_char b (input_char c);
              r ()
            with
                End_of_file ->
                  ignore (Unix.close_process_in c);
                  Buffer.contents b
          in
            [| span ps apos bpos
                 (Ast.STMT_new_str (dst_lval, r())) |]

      | ("fmt", nargs, None) ->
            if nargs = 0
            then raise (err "malformed #fmt call" ps)
            else
              begin
                match pexp_args.(0).node with
                    Ast.PEXP_str s ->
                      let (arg_stmts, args) =
                        desugar_expr_atoms ps
                          (Array.sub pexp_args 1 (nargs-1))
                      in

                      let pieces = Extfmt.parse_fmt_string s in
                      let fmt_stmts =
                        fmt_pieces_to_stmts
                          ps apos bpos dst_lval pieces args
                      in
                        Array.append arg_stmts fmt_stmts
                  | _ ->
                      raise (err "malformed #fmt call" ps)
              end

      | _ ->
          raise (err ("unknown syntax extension: " ^ nstr) ps)

and fmt_pieces_to_stmts
    (ps:pstate)
    (apos:pos)
    (bpos:pos)
    (dst_lval:Ast.lval)
    (pieces:Extfmt.piece array)
    (args:Ast.atom array)
    : (Ast.stmt array) =

  let stmts = Queue.create () in

  let make_new_tmp _ =
    let (_, tmp, decl_stmt) = build_tmp ps slot_auto apos bpos in
      Queue.add decl_stmt stmts;
      tmp
  in

  let make_new_str s =
    let tmp = make_new_tmp () in
    let init_stmt =
      span ps apos bpos (Ast.STMT_new_str (clone_lval ps tmp, s))
    in
      Queue.add init_stmt stmts;
      tmp
  in

  let make_append dst_lval src_atom =
    let stmt =
      span ps apos bpos
        (Ast.STMT_copy_binop
           ((clone_lval ps dst_lval), Ast.BINOP_add, src_atom))
    in
      Queue.add stmt stmts
  in

  let make_append_lval dst_lval src_lval =
      make_append dst_lval (Ast.ATOM_lval (clone_lval ps src_lval))
  in

  let rec make_lval' path =
    match path with
        [n] ->
          Ast.LVAL_base (span ps apos bpos (Ast.BASE_ident n))

      | x :: xs ->
          Ast.LVAL_ext (make_lval' xs,
                        Ast.COMP_named (Ast.COMP_ident x))

      | [] -> (bug () "make_lval on empty list in #fmt")
  in

  let make_lval path = make_lval' (List.rev path) in

  let make_call dst path args =
    let callee = make_lval path in
    let stmt =
      span ps apos bpos (Ast.STMT_call (dst, callee, args ))
    in
      Queue.add stmt stmts
  in

  let ulit i =
    Ast.ATOM_literal (span ps apos bpos (Ast.LIT_uint (Int64.of_int i)))
  in

  let n = ref 0 in
  let tmp_lval = make_new_str "" in
  let final_stmt =
    span ps apos bpos
      (Ast.STMT_copy
         (clone_lval ps dst_lval,
          Ast.EXPR_atom (Ast.ATOM_lval tmp_lval)))
  in
    Array.iter
      begin
        fun piece ->
          match piece with
              Extfmt.PIECE_string s ->
                let s_lval = make_new_str s in
                  make_append_lval tmp_lval s_lval

            | Extfmt.PIECE_conversion conv ->
                if not
                  ((conv.Extfmt.conv_parameter = None) &&
                     (conv.Extfmt.conv_flags = []) &&
                     (conv.Extfmt.conv_width = Extfmt.COUNT_implied) &&
                     (conv.Extfmt.conv_precision = Extfmt.COUNT_implied))
                then
                  raise (err "conversion not supported in #fmt string" ps);
                if !n >= Array.length args
                then raise (err "too many conversions in #fmt string" ps);
                let arg = args.(!n) in
                  incr n;
                  match conv.Extfmt.conv_ty with
                      Extfmt.TY_str ->
                        make_append tmp_lval arg

                    | Extfmt.TY_int Extfmt.SIGNED ->
                        let t = make_new_tmp () in
                          make_call t
                            ["std"; "_int"; "to_str" ] [| arg; ulit 10 |];

                          make_append_lval tmp_lval t

                    | Extfmt.TY_int Extfmt.UNSIGNED ->
                        let t = make_new_tmp () in
                          make_call t
                            ["std"; "_uint"; "to_str" ] [| arg; ulit 10 |];
                          make_append_lval tmp_lval t

                    | _ ->
                        raise (err "conversion not supported in #fmt" ps);
      end
      pieces;
    Queue.add final_stmt stmts;
    queue_to_arr stmts;


and atom_lval (_:pstate) (at:Ast.atom) : Ast.lval =
  match at with
      Ast.ATOM_lval lv -> lv
    | Ast.ATOM_literal _
    | Ast.ATOM_pexp _ -> bug () "Pexp.atom_lval on non-ATOM_lval"
;;




(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
