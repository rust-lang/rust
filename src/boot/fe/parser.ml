
open Common;;
open Token;;

(* Fundamental parser types and actions *)

type get_mod_fn = (Ast.meta_pat
                   -> node_id
                     -> (crate_id, Ast.mod_items) Hashtbl.t
                       -> (filename * Ast.mod_items))
;;

type pstate =
    { mutable pstate_peek : token;
      mutable pstate_ctxt : (string * pos) list;
      mutable pstate_rstr : bool;
      mutable pstate_depth: int;
      pstate_lexbuf       : Lexing.lexbuf;
      pstate_file         : filename;
      pstate_sess         : Session.sess;
      pstate_crate_cache  : (crate_id, Ast.mod_items) Hashtbl.t;
      pstate_get_mod      : get_mod_fn;
      pstate_get_cenv_tok : pstate -> Ast.ident -> token;
      pstate_infer_lib_name : (Ast.ident -> filename);
      pstate_required       : (node_id, (required_lib * nabi_conv)) Hashtbl.t;
      pstate_required_syms  : (node_id, string) Hashtbl.t; }
;;

let log (ps:pstate) = Session.log "parse"
  ps.pstate_sess.Session.sess_log_parse
  ps.pstate_sess.Session.sess_log_out
;;

let iflog ps thunk =
  if ps.pstate_sess.Session.sess_log_parse
  then thunk ()
  else ()
;;

let make_parser
    (crate_cache:(crate_id, Ast.mod_items) Hashtbl.t)
    (sess:Session.sess)
    (get_mod:get_mod_fn)
    (get_cenv_tok:pstate -> Ast.ident -> token)
    (infer_lib_name:Ast.ident -> filename)
    (required:(node_id, (required_lib * nabi_conv)) Hashtbl.t)
    (required_syms:(node_id, string) Hashtbl.t)
    (fname:string)
    : pstate =
  let lexbuf = Lexing.from_channel (open_in fname) in
  let spos = { lexbuf.Lexing.lex_start_p with Lexing.pos_fname = fname } in
  let cpos = { lexbuf.Lexing.lex_curr_p with Lexing.pos_fname = fname } in
    lexbuf.Lexing.lex_start_p <- spos;
    lexbuf.Lexing.lex_curr_p <- cpos;
    let first = Lexer.token lexbuf in
    let ps =
      { pstate_peek = first;
        pstate_ctxt = [];
        pstate_rstr = false;
        pstate_depth = 0;
        pstate_lexbuf = lexbuf;
        pstate_file = fname;
        pstate_sess = sess;
        pstate_crate_cache = crate_cache;
        pstate_get_mod = get_mod;
        pstate_get_cenv_tok = get_cenv_tok;
        pstate_infer_lib_name = infer_lib_name;
        pstate_required = required;
        pstate_required_syms = required_syms; }
    in
      iflog ps (fun _ -> log ps "made parser for: %s\n%!" fname);
      ps
;;

exception Parse_err of (pstate * string)
;;

let lexpos (ps:pstate) : pos =
  let p = ps.pstate_lexbuf.Lexing.lex_start_p in
    (p.Lexing.pos_fname,
     p.Lexing.pos_lnum ,
     (p.Lexing.pos_cnum) - (p.Lexing.pos_bol))
;;

let next_node_id (ps:pstate) : node_id =
  let r = ps.pstate_sess.Session.sess_node_id_counter in
  let id = !r in
    r := Node ((int_of_node id)+1);
    id
;;

let next_opaque_id (ps:pstate) : opaque_id =
  let r = ps.pstate_sess.Session.sess_opaque_id_counter in
  let id = !r in
    r := Opaque ((int_of_opaque id)+1);
    id
;;

let span
    (ps:pstate)
    (apos:pos)
    (bpos:pos)
    (x:'a)
    : 'a identified =
  let span = { lo = apos; hi = bpos } in
  let id = next_node_id ps in
    iflog ps (fun _ -> log ps "span for node #%d: %s"
                (int_of_node id) (Session.string_of_span span));
    htab_put ps.pstate_sess.Session.sess_spans id span;
    { node = x; id = id }
;;

let decl p i =
  { Ast.decl_params = p;
    Ast.decl_item = i }
;;

let spans
    (ps:pstate)
    (things:('a identified) array)
    (apos:pos)
    (thing:'a)
    : ('a identified) array =
  Array.append things [| (span ps apos (lexpos ps) thing) |]
;;

(* 
 * The point of this is to make a new node_id entry for a node that is a
 * "copy" of an lval returned from somewhere else. For example if you create
 * a temp, the lval it returns can only be used in *one* place, for the
 * node_id denotes the place that lval is first used; subsequent uses of
 * 'the same' reference must clone_lval it into a new node_id. Otherwise
 * there is trouble.
 *)

let clone_span
    (ps:pstate)
    (oldnode:'a identified)
    (newthing:'b)
    : 'b identified =
  let s = Hashtbl.find ps.pstate_sess.Session.sess_spans oldnode.id in
    span ps s.lo s.hi newthing
;;

let rec clone_lval (ps:pstate) (lval:Ast.lval) : Ast.lval =
  match lval with
      Ast.LVAL_base nb ->
        let nnb = clone_span ps nb nb.node in
          Ast.LVAL_base nnb
    | Ast.LVAL_ext (base, ext) ->
        Ast.LVAL_ext ((clone_lval ps base), ext)
;;

let clone_atom (ps:pstate) (atom:Ast.atom) : Ast.atom =
  match atom with
      Ast.ATOM_literal _ -> atom
    | Ast.ATOM_lval lv -> Ast.ATOM_lval (clone_lval ps lv)
    | Ast.ATOM_pexp _ -> bug () "Parser.clone_atom on ATOM_pexp"
;;

let ctxt (n:string) (f:pstate -> 'a) (ps:pstate) : 'a =
  (ps.pstate_ctxt <- (n, lexpos ps) :: ps.pstate_ctxt;
   let res = f ps in
     ps.pstate_ctxt <- List.tl ps.pstate_ctxt;
     res)
;;

let rstr (r:bool) (f:pstate -> 'a) (ps:pstate) : 'a =
  let prev = ps.pstate_rstr in
    (ps.pstate_rstr <- r;
     let res = f ps in
       ps.pstate_rstr <- prev;
       res)
;;

let err (str:string) (ps:pstate) =
  (Parse_err (ps, (str)))
;;


let (slot_nil:Ast.slot) =
  { Ast.slot_mode = Ast.MODE_local;
    Ast.slot_ty = Some Ast.TY_nil }
;;

let (slot_auto:Ast.slot) =
  { Ast.slot_mode = Ast.MODE_local;
    Ast.slot_ty = None }
;;

let build_tmp
    (ps:pstate)
    (slot:Ast.slot)
    (apos:pos)
    (bpos:pos)
    : (temp_id * Ast.lval * Ast.stmt) =
  let r = ps.pstate_sess.Session.sess_temp_id_counter in
  let id = !r in
    r := Temp ((int_of_temp id)+1);
    iflog ps
      (fun _ -> log ps "building temporary %d" (int_of_temp id));
    let decl = Ast.DECL_slot (Ast.KEY_temp id, (span ps apos bpos slot)) in
    let declstmt = span ps apos bpos (Ast.STMT_decl decl) in
    let tmp = Ast.LVAL_base (span ps apos bpos (Ast.BASE_temp id)) in
      (id, tmp, declstmt)
;;

(* Simple helpers *)

(* FIXME (issue #71): please rename these, they make eyes bleed. *)

let arr (ls:'a list) : 'a array = Array.of_list ls ;;
let arl (ls:'a list) : 'a array = Array.of_list (List.rev ls) ;;
let arj (ar:('a array array)) = Array.concat (Array.to_list ar) ;;
let arj1st (pairs:(('a array) * 'b) array) : (('a array) * 'b array) =
  let (az, bz) = List.split (Array.to_list pairs) in
    (Array.concat az, Array.of_list bz)


(* Bottom-most parser actions. *)

let peek (ps:pstate) : token =
  iflog ps
    begin
      fun _ ->
        log ps "peeking at: %s     // %s"
          (string_of_tok ps.pstate_peek)
          (match ps.pstate_ctxt with
               (s, _) :: _ -> s
             | _ -> "<empty>")
    end;
  ps.pstate_peek
;;


let bump (ps:pstate) : unit =
  begin
    iflog ps (fun _ -> log ps "bumping past: %s"
                (string_of_tok ps.pstate_peek));
    ps.pstate_peek <- Lexer.token ps.pstate_lexbuf
  end
;;

let bump_bracequote (ps:pstate) : unit =
  begin
    assert (ps.pstate_peek = LBRACE);
    iflog ps (fun _ -> log ps "bumping past: %s"
                (string_of_tok ps.pstate_peek));
    let buf = Buffer.create 32 in
      ps.pstate_peek <- Lexer.bracequote buf 1 ps.pstate_lexbuf
  end
;;


let expect (ps:pstate) (t:token) : unit =
  let p = peek ps in
    if p == t
    then bump ps
    else
      let msg = ("Expected '" ^ (string_of_tok t) ^
                   "', found '" ^ (string_of_tok p ) ^ "'") in
        raise (Parse_err (ps, msg))
;;

let unexpected (ps:pstate) =
  err ("Unexpected token '" ^ (string_of_tok (peek ps)) ^ "'") ps
;;



(* Parser combinators. *)

let one_or_more
    (sep:token)
    (prule:pstate -> 'a)
    (ps:pstate)
    : 'a array =
  let accum = ref [prule ps] in
    while peek ps == sep
    do
      bump ps;
      accum := (prule ps) :: !accum
    done;
    arl !accum
;;

let bracketed_seq
    (mandatory:int)
    (bra:token)
    (ket:token)
    (sepOpt:token option)
    (prule:pstate -> 'a)
    (ps:pstate)
    : 'a array =
  expect ps bra;
  let accum = ref [] in
  let dosep _ =
    (match sepOpt with
         None -> ()
       | Some tok ->
           if (!accum = [])
           then ()
           else expect ps tok)
  in
    while mandatory > List.length (!accum) do
      dosep ();
      accum := (prule ps) :: (!accum)
    done;
    while (not (peek ps = ket))
    do
      dosep ();
      accum := (prule ps) :: !accum
    done;
    expect ps ket;
    arl !accum
;;


let bracketed_zero_or_more
    (bra:token)
    (ket:token)
    (sepOpt:token option)
    (prule:pstate -> 'a)
    (ps:pstate)
    : 'a array =
  bracketed_seq 0 bra ket sepOpt (ctxt "bracketed_seq" prule) ps
;;


let paren_comma_list
    (prule:pstate -> 'a)
    (ps:pstate)
    : 'a array =
  bracketed_zero_or_more LPAREN RPAREN (Some COMMA) prule ps
;;

let bracketed_one_or_more
    (bra:token)
    (ket:token)
    (sepOpt:token option)
    (prule:pstate -> 'a)
    (ps:pstate)
    : 'a array =
  bracketed_seq 1 bra ket sepOpt (ctxt "bracketed_seq" prule) ps
;;

let bracketed_two_or_more
    (bra:token)
    (ket:token)
    (sepOpt:token option)
    (prule:pstate -> 'a)
    (ps:pstate)
    : 'a array =
  bracketed_seq 2 bra ket sepOpt (ctxt "bracketed_seq" prule) ps
;;


let bracketed (bra:token) (ket:token) (prule:pstate -> 'a) (ps:pstate) : 'a =
  expect ps bra;
  let res = ctxt "bracketed" prule ps in
    expect ps ket;
    res
;;

(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
