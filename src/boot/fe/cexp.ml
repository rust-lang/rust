
open Common;;
open Token;;
open Parser;;

(* NB: cexps (crate-expressions / constant-expressions) are only used
 * transiently during compilation: they are the outermost expression-language
 * describing crate configuration and constants. They are completely evaluated
 * at compile-time, in a little micro-interpreter defined here, with the
 * results of evaluation being the sequence of directives controlling the rest
 * of the compiler.
 * 
 * Cexps, like pexps, do not escape the language front-end.
 * 
 * You can think of the AST as a statement-language called "item" sandwiched
 * between two expression-languages, "cexp" on the outside and "pexp" on the
 * inside. The front-end evaluates cexp on the outside in order to get one big
 * directive-list, evaluating those parts of pexp that are directly used by
 * cexp in passing, and desugaring those remaining parts of pexp that are
 * embedded within the items of the directives.
 * 
 * The rest of the compiler only deals with the directives, which are mostly
 * just a set of containers for items. Items are what most of AST describes
 * ("most" because the type-grammar spans both items and pexps).
 * 
 *)

type meta = (Ast.ident * Ast.pexp) array;;

type meta_pat = (Ast.ident * (Ast.pexp option)) array;;

type auth = (Ast.name * Ast.auth);;

type cexp =
    CEXP_alt of cexp_alt identified
  | CEXP_let of cexp_let identified
  | CEXP_src_mod of cexp_src identified
  | CEXP_dir_mod of cexp_dir identified
  | CEXP_use_mod of cexp_use identified
  | CEXP_nat_mod of cexp_nat identified
  | CEXP_meta of meta identified
  | CEXP_auth of auth identified

and cexp_alt =
    { alt_val: Ast.pexp;
      alt_arms: (Ast.pexp * cexp array) array;
      alt_else: cexp array }

and cexp_let =
    { let_ident: Ast.ident;
      let_value: Ast.pexp;
      let_body: cexp array; }

and cexp_src =
    { src_ident: Ast.ident;
      src_path: Ast.pexp option }

and cexp_dir =
    { dir_ident: Ast.ident;
      dir_path: Ast.pexp option;
      dir_body: cexp array }

and cexp_use =
    { use_ident: Ast.ident;
      use_meta: meta_pat; }

and cexp_nat =
    { nat_abi: string;
      nat_ident: Ast.ident;
      nat_path: Ast.pexp option;
      (* 
       * FIXME: possibly support embedding optional strings as
       * symbol-names, to handle mangling schemes that aren't
       * Token.IDENT values
       *)
      nat_items: Ast.mod_items;
    }
;;


(* Cexp grammar. *)

let parse_meta_input (ps:pstate) : (Ast.ident * Ast.pexp option) =
  let lab = (ctxt "meta input: label" Pexp.parse_ident ps) in
    match peek ps with
        EQ ->
          bump ps;
          let v =
            match peek ps with
                UNDERSCORE -> bump ps; None
              | _ -> Some (Pexp.parse_pexp ps)
          in
            (lab, v)
      | _ -> raise (unexpected ps)
;;

let parse_meta_pat (ps:pstate) : meta_pat =
  bracketed_zero_or_more LPAREN RPAREN
    (Some COMMA) parse_meta_input ps
;;

let parse_meta (ps:pstate) : meta =
  Array.map
    begin
      fun (id,v) ->
        match v with
            None ->
              raise (err ("wildcard found in meta pattern "
                          ^ "where value expected") ps)
          | Some v -> (id,v)
    end
    (parse_meta_pat ps)
;;

let parse_optional_meta_pat
    (ps:pstate)
    (ident:Ast.ident)
    : meta_pat =
  match peek ps with
      LPAREN -> parse_meta_pat ps
    | _ ->
        let apos = lexpos ps in
          [| ("name", Some (span ps apos apos (Ast.PEXP_str ident))) |]
;;

let rec parse_cexps (ps:pstate) (term:Token.token) : cexp array =
  let cexps = Queue.create () in
    while ((peek ps) <> term)
    do
      Queue.push (parse_cexp ps) cexps
    done;
    expect ps term;
    queue_to_arr cexps

and parse_cexp (ps:pstate) : cexp =

  let apos = lexpos ps in
    match peek ps with
        MOD ->
          begin
            bump ps;
            let name = ctxt "mod: name" Pexp.parse_ident ps in
            let path = ctxt "mod: path" parse_eq_pexp_opt ps
            in
              match peek ps with
                  SEMI ->
                    bump ps;
                    let bpos = lexpos ps in
                      CEXP_src_mod
                        (span ps apos bpos { src_ident = name;
                                             src_path = path })
                | LBRACE ->
                    let body =
                      bracketed_zero_or_more LBRACE RBRACE
                        None parse_cexp ps
                    in
                    let bpos = lexpos ps in
                      CEXP_dir_mod
                        (span ps apos bpos { dir_ident = name;
                                             dir_path = path;
                                             dir_body = body })
                | _ -> raise (unexpected ps)
        end

      | NATIVE ->
          begin
            bump ps;
            let abi =
                match peek ps with
                    MOD -> "cdecl"
                  | LIT_STR s -> bump ps; s
                  | _ -> raise (unexpected ps)
            in
            let _ = expect ps MOD in
            let name = ctxt "native mod: name" Pexp.parse_ident ps in
            let path = ctxt "native mod: path" parse_eq_pexp_opt ps in
            let items = Hashtbl.create 0 in
            let get_item ps =
              Array.map
                begin
                  fun (ident, item) ->
                    htab_put items ident item
                end
                (Item.parse_native_mod_item_from_signature ps)
            in
              ignore (bracketed_zero_or_more
                        LBRACE RBRACE None get_item ps);
              let bpos = lexpos ps in
                CEXP_nat_mod
                  (span ps apos bpos { nat_abi = abi;
                                       nat_ident = name;
                                       nat_path = path;
                                       nat_items = items })
          end

      | USE ->
          begin
            bump ps;
            let ident = ctxt "use mod: name" Pexp.parse_ident ps in
            let meta =
              ctxt "use mod: meta" parse_optional_meta_pat ps ident
            in
            let bpos = lexpos ps in
              expect ps SEMI;
              CEXP_use_mod
                (span ps apos bpos { use_ident = ident;
                                     use_meta = meta })
          end

      | LET ->
          begin
            bump ps;
            expect ps LPAREN;
            let id = Pexp.parse_ident ps in
              expect ps EQ;
              let v = Pexp.parse_pexp ps in
                expect ps RPAREN;
                expect ps LBRACE;
                let body = parse_cexps ps RBRACE in
                let bpos = lexpos ps in
                  CEXP_let
                    (span ps apos bpos
                       { let_ident = id;
                         let_value = v;
                         let_body = body })
          end

      | ALT ->
          begin
            bump ps;
            expect ps LPAREN;
            let v = Pexp.parse_pexp ps in
              expect ps RPAREN;
              expect ps LBRACE;
              let rec consume_arms arms =
                match peek ps with
                    CASE ->
                      begin
                        bump ps;
                        expect ps LPAREN;
                        let cond = Pexp.parse_pexp ps in
                          expect ps RPAREN;
                          expect ps LBRACE;
                          let consequent = parse_cexps ps RBRACE in
                            let arm = (cond, consequent) in
                            consume_arms (arm::arms)
                      end
                  | ELSE ->
                      begin
                        bump ps;
                        expect ps LBRACE;
                        let consequent = parse_cexps ps RBRACE in
                          expect ps RBRACE;
                          let bpos = lexpos ps in
                            span ps apos bpos
                              { alt_val = v;
                                alt_arms = Array.of_list (List.rev arms);
                                alt_else = consequent }
                      end

                  | _ -> raise (unexpected ps)
              in
                CEXP_alt (consume_arms [])
          end

      | META ->
          bump ps;
          let meta = parse_meta ps in
            expect ps SEMI;
            let bpos = lexpos ps in
              CEXP_meta (span ps apos bpos meta)

      | AUTH ->
          bump ps;
          let name = Pexp.parse_name ps in
            expect ps EQ;
            let au = Pexp.parse_auth ps in
              expect ps SEMI;
              let bpos = lexpos ps in
                CEXP_auth (span ps apos bpos (name, au))

      | _ -> raise (unexpected ps)


and  parse_eq_pexp_opt (ps:pstate) : Ast.pexp option =
  match peek ps with
      EQ ->
        begin
          bump ps;
          Some (Pexp.parse_pexp ps)
        end
    | _ -> None
;;


(*
 * Dynamic-typed micro-interpreter for the cexp language.
 * 
 * The product of evaluating a pexp is a pval.
 * 
 * The product of evlauating a cexp is a cdir array.
 *)

type pval =
    PVAL_str of string
  | PVAL_int of int64
  | PVAL_bool of bool
;;

type cdir =
    CDIR_meta of ((Ast.ident * string) array)
  | CDIR_syntax of Ast.name
  | CDIR_mod of (Ast.ident * Ast.mod_item)
  | CDIR_auth of auth

type env = { env_bindings: ((Ast.ident * pval) list) ref;
             env_prefix: filename list;
             env_items: (filename, Ast.mod_items) Hashtbl.t;
             env_files: (node_id,filename) Hashtbl.t;
             env_required: (node_id, (required_lib * nabi_conv)) Hashtbl.t;
             env_required_syms: (node_id, string) Hashtbl.t;
             env_ps: pstate; }

let unexpected_val (expected:string) (v:pval)  =
  let got =
    match v with
        PVAL_str s -> "str \"" ^ (String.escaped s) ^ "\""
      | PVAL_int i -> "int " ^ (Int64.to_string i)
      | PVAL_bool b -> if b then "bool true" else "bool false"
  in
    (* FIXME (issue #70): proper error reporting, please. *)
    bug () "expected %s, got %s" expected got
;;

let rewrap_items id items =
  let item = decl [||] (Ast.MOD_ITEM_mod items) in
    { id = id; node = item }
;;


let rec eval_cexps (env:env) (exps:cexp array) : cdir array =
  Parser.arj (Array.map (eval_cexp env) exps)

and eval_cexp (env:env) (exp:cexp) : cdir array =
  match exp with
      CEXP_alt { node = ca; id = _ } ->
        let v = eval_pexp env ca.alt_val in
        let rec try_arm i =
          if i >= Array.length ca.alt_arms
          then ca.alt_else
          else
            let (arm_head, arm_body) = ca.alt_arms.(i) in
            let v' = eval_pexp env arm_head in
              if v' = v
              then arm_body
              else try_arm (i+1)
        in
          eval_cexps env (try_arm 0)

    | CEXP_let { node = cl; id = _ } ->
        let ident = cl.let_ident in
        let v = eval_pexp env cl.let_value in
        let old_bindings = !(env.env_bindings) in
          env.env_bindings := (ident,v)::old_bindings;
          let res = eval_cexps env cl.let_body in
            env.env_bindings := old_bindings;
            res

    | CEXP_src_mod {node=s; id=id} ->
        let name = s.src_ident in
        let path =
          match s.src_path with
              None -> name ^ ".rs"
            | Some p -> eval_pexp_to_str env p
        in
        let full_path =
          List.fold_left Filename.concat ""
            (List.rev (path :: env.env_prefix))
        in
        let ps = env.env_ps in
        let p =
          make_parser
            ps.pstate_crate_cache
            ps.pstate_sess
            ps.pstate_get_mod
            ps.pstate_get_cenv_tok
            ps.pstate_infer_lib_name
            env.env_required
            env.env_required_syms
            full_path
        in
        let items = Item.parse_mod_items p EOF in
          htab_put env.env_files id full_path;
          [| CDIR_mod (name, rewrap_items id items) |]

    | CEXP_dir_mod {node=d; id=id} ->
        let items = Hashtbl.create 0 in
        let name = d.dir_ident in
        let path =
          match d.dir_path with
              None -> name
            | Some p -> eval_pexp_to_str env p
        in
        let env = { env with
                      env_prefix = path :: env.env_prefix } in
        let sub_directives = eval_cexps env d.dir_body in
        let add d =
          match d with
              CDIR_mod (name, item) ->
                htab_put items name item
            | _ -> raise (err "non-'mod' directive found in 'dir' directive"
                            env.env_ps)
        in
          Array.iter add sub_directives;
          [| CDIR_mod (name, rewrap_items id (Item.empty_view, items)) |]

    | CEXP_use_mod {node=u; id=id} ->
        let ps = env.env_ps in
        let name = u.use_ident in
        let (path, items) =
          let meta_pat =
            Array.map
              begin
                fun (k,vo) ->
                  match vo with
                      None -> (k, None)
                    | Some p -> (k, Some (eval_pexp_to_str env p))
              end
              u.use_meta
          in
            ps.pstate_get_mod meta_pat id ps.pstate_crate_cache
        in
          iflog ps
            begin
              fun _ ->
                log ps "extracted mod signature from %s (binding to %s)"
                  path name;
                log ps "%a" Ast.sprintf_mod_items items;
            end;
          let rlib = REQUIRED_LIB_rust { required_libname = path;
                                         required_prefix = 1 }
          in
          let item = decl [||] (Ast.MOD_ITEM_mod (Item.empty_view, items)) in
          let item = { id = id; node = item } in
          let span = Hashtbl.find ps.pstate_sess.Session.sess_spans id in
            Item.note_required_mod env.env_ps span CONV_rust rlib item;
            [| CDIR_mod (name, item) |]

    | CEXP_nat_mod {node=cn;id=id} ->
        let conv =
          let v = cn.nat_abi in
          match string_to_conv v with
              None -> unexpected_val "calling convention" (PVAL_str v)
            | Some c -> c
        in
        let name = cn.nat_ident in
        let filename =
          match cn.nat_path with
              None -> env.env_ps.pstate_infer_lib_name name
            | Some p -> eval_pexp_to_str env p
        in
        let item =
          decl [||] (Ast.MOD_ITEM_mod (Item.empty_view, cn.nat_items))
        in
        let item = { id = id; node = item } in
        let rlib = REQUIRED_LIB_c { required_libname = filename;
                                    required_prefix = 1 }
        in
        let ps = env.env_ps in
        let span = Hashtbl.find ps.pstate_sess.Session.sess_spans id in
          Item.note_required_mod env.env_ps span conv rlib item;
          [| CDIR_mod (name, item) |]

    | CEXP_meta m ->
        [| CDIR_meta
             begin
               Array.map
                 begin
                   fun (id, p) -> (id, eval_pexp_to_str env p)
                 end
                 m.node
             end |]

    | CEXP_auth a -> [| CDIR_auth a.node |]


and eval_pexp (env:env) (exp:Ast.pexp) : pval =
  match exp.node with
    | Ast.PEXP_binop (bop, a, b) ->
        begin
          let av = eval_pexp env a in
          let bv = eval_pexp env b in
            match (bop, av, bv) with
                (Ast.BINOP_add, PVAL_str az, PVAL_str bz) ->
                  PVAL_str (az ^ bz)
              | _ ->
                  let av = (need_int av) in
                  let bv = (need_int bv) in
                    PVAL_int
                      begin
                        match bop with
                            Ast.BINOP_add -> Int64.add av bv
                          | Ast.BINOP_sub -> Int64.sub av bv
                          | Ast.BINOP_mul -> Int64.mul av bv
                          | Ast.BINOP_div -> Int64.div av bv
                          | _ ->
                              bug ()
                                "unhandled arithmetic op in Cexp.eval_pexp"
                      end
        end

    | Ast.PEXP_unop (uop, a) ->
        begin
          match uop with
              Ast.UNOP_not ->
                PVAL_bool (not (eval_pexp_to_bool env a))
            | Ast.UNOP_neg ->
                PVAL_int (Int64.neg (eval_pexp_to_int env a))
            | _ -> bug () "Unexpected unop in Cexp.eval_pexp"
        end

    | Ast.PEXP_lval (Ast.PLVAL_base (Ast.BASE_ident ident)) ->
        begin
          match ltab_search !(env.env_bindings) ident with
              None -> raise (err (Printf.sprintf "no binding for '%s' found"
                                    ident) env.env_ps)
            | Some v -> v
        end

    | Ast.PEXP_lit (Ast.LIT_bool b) ->
        PVAL_bool b

    | Ast.PEXP_lit (Ast.LIT_int i) ->
        PVAL_int i

    | Ast.PEXP_str s ->
        PVAL_str s

    | _ -> bug () "unexpected Pexp in Cexp.eval_pexp"


and eval_pexp_to_str (env:env) (exp:Ast.pexp) : string =
  match eval_pexp env exp with
      PVAL_str s -> s
    | v -> unexpected_val "str" v

and need_int (cv:pval) : int64 =
  match cv with
      PVAL_int n -> n
    | v -> unexpected_val "int" v

and eval_pexp_to_int (env:env) (exp:Ast.pexp) : int64 =
  need_int (eval_pexp env exp)

and eval_pexp_to_bool (env:env) (exp:Ast.pexp) : bool =
  match eval_pexp env exp with
      PVAL_bool b -> b
    | v -> unexpected_val "bool" v

;;


let find_main_fn
    (ps:pstate)
    (crate_items:Ast.mod_items)
    : Ast.name =
  let fns = ref [] in
  let extend prefix_name ident =
    match prefix_name with
        None -> Ast.NAME_base (Ast.BASE_ident ident)
      | Some n -> Ast.NAME_ext (n, Ast.COMP_ident ident)
  in
  let rec dig prefix_name items =
    Hashtbl.iter (extract_fn prefix_name) items
  and extract_fn prefix_name ident item =
    if not (Array.length item.node.Ast.decl_params = 0) ||
      Hashtbl.mem ps.pstate_required item.id
    then ()
    else
      match item.node.Ast.decl_item with
          Ast.MOD_ITEM_mod (_, items) ->
            dig (Some (extend prefix_name ident)) items

       | Ast.MOD_ITEM_fn _ ->
            if ident = "main"
            then fns := (extend prefix_name ident) :: (!fns)
            else ()

        | _ -> ()
  in
    dig None crate_items;
    match !fns with
        [] -> raise (err "no 'main' function found" ps)
      | [x] -> x
      | _ -> raise (err "multiple 'main' functions found" ps)
;;


let with_err_handling sess thunk =
  try
    thunk ()
  with
      Parse_err (ps, str) ->
        Session.fail sess "%s: error: %s\n%!"
          (Session.string_of_pos (lexpos ps)) str;
        List.iter
          (fun (cx,pos) ->
             Session.fail sess "%s: (parse context): %s\n%!"
               (Session.string_of_pos pos) cx)
          ps.pstate_ctxt;
        let apos = lexpos ps in
          span ps apos apos Ast.empty_crate'
;;


let parse_crate_file
    (sess:Session.sess)
    (get_mod:get_mod_fn)
    (infer_lib_name:(Ast.ident -> filename))
    (crate_cache:(crate_id, Ast.mod_items) Hashtbl.t)
    : Ast.crate =
  let fname = Session.filename_of sess.Session.sess_in in
  let required = Hashtbl.create 4 in
  let required_syms = Hashtbl.create 4 in
  let files = Hashtbl.create 0 in
  let items = Hashtbl.create 4 in
  let target_bindings =
    let (os, arch, libc) =
      match sess.Session.sess_targ with
          Linux_x86_elf -> ("linux", "x86", "libc.so.6")
        | FreeBSD_x86_elf -> ("freebsd", "x86", "libc.so.7")
        | Win32_x86_pe -> ("win32", "x86", "msvcrt.dll")
        | MacOS_x86_macho -> ("macos", "x86", "libc.dylib")
    in
      [
        ("target_os", PVAL_str os);
        ("target_arch", PVAL_str arch);
        ("target_libc", PVAL_str libc)
      ]
  in
  let build_bindings =
    [
      ("build_compiler", PVAL_str Sys.executable_name);
      ("build_input", PVAL_str fname);
    ]
  in
  let bindings =
    ref (target_bindings
         @ build_bindings)
  in
  let get_cenv_tok ps ident =
      match ltab_search (!bindings) ident with
          None -> raise (err (Printf.sprintf "no binding for '%s' found"
                                ident) ps)
        | Some (PVAL_bool b) -> LIT_BOOL b
        | Some (PVAL_str s) -> LIT_STR s
        | Some (PVAL_int n) -> LIT_INT n
  in
  let ps =
    make_parser crate_cache sess get_mod get_cenv_tok
      infer_lib_name required required_syms fname
  in
  let env = { env_bindings = bindings;
              env_prefix = [Filename.dirname fname];
              env_items = Hashtbl.create 0;
              env_files = files;
              env_required = required;
              env_required_syms = required_syms;
              env_ps = ps; }
  in
  let auth = Hashtbl.create 0 in
    with_err_handling sess
      begin
        fun _ ->
          let apos = lexpos ps in
          let cexps = parse_cexps ps EOF in
          let cdirs = eval_cexps env cexps in
          let meta = Queue.create () in
          let _ =
            Array.iter
              begin
                fun d ->
                  match d with
                      CDIR_mod (name, item) ->
                        if Hashtbl.mem items name
                        then raise
                          (err ("duplicate mod declaration: " ^ name) ps)
                        else Hashtbl.add items name item
                    | CDIR_meta metas ->
                        Array.iter (fun m -> Queue.add m meta) metas
                    | CDIR_auth (n,e) ->
                        if Hashtbl.mem auth n
                        then raise (err "duplicate 'auth' clause" ps)
                        else Hashtbl.add auth n e
                    | _ ->
                        raise
                          (err "unhandled directive at top level" ps)
              end
              cdirs
          in
          let bpos = lexpos ps in
          let main =
            if ps.pstate_sess.Session.sess_library_mode
            then None
            else Some (find_main_fn ps items) in
          let crate = { Ast.crate_items = (Item.empty_view, items);
                        Ast.crate_meta = queue_to_arr meta;
                        Ast.crate_auth = auth;
                        Ast.crate_required = required;
                        Ast.crate_required_syms = required_syms;
                        Ast.crate_main = main;
                        Ast.crate_files = files }
          in
          let cratei = span ps apos bpos crate in
            htab_put files cratei.id fname;
            cratei
      end
;;

let parse_src_file
    (sess:Session.sess)
    (get_mod:get_mod_fn)
    (infer_lib_name:(Ast.ident -> filename))
    (crate_cache:(crate_id, Ast.mod_items) Hashtbl.t)
    : Ast.crate =
  let fname = Session.filename_of sess.Session.sess_in in
  let required = Hashtbl.create 0 in
  let required_syms = Hashtbl.create 0 in
  let get_cenv_tok ps ident =
    raise (err (Printf.sprintf "no binding for '%s' found"
                  ident) ps)
  in
  let ps =
    make_parser crate_cache sess get_mod get_cenv_tok
      infer_lib_name required required_syms fname
  in
    with_err_handling sess
      begin
        fun _ ->
          let apos = lexpos ps in
          let items = Item.parse_mod_items ps EOF in
          let bpos = lexpos ps in
          let files = Hashtbl.create 0 in
          let main =
            if ps.pstate_sess.Session.sess_library_mode
            then None
            else Some (find_main_fn ps (snd items))
          in
          let crate = { Ast.crate_items = items;
                        Ast.crate_required = required;
                        Ast.crate_required_syms = required_syms;
                        Ast.crate_auth = Hashtbl.create 0;
                        Ast.crate_meta = [||];
                        Ast.crate_main = main;
                        Ast.crate_files = files }
          in
          let cratei = span ps apos bpos crate in
            htab_put files cratei.id fname;
            cratei
      end
;;


(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
