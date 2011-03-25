open Common;;
open Ast;;

let ident_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_";;
let digit_chars = "1234567890";;

type scope =
    SCOPE_crate of crate
  | SCOPE_mod_item of (ident * mod_item)
  | SCOPE_block of block
  | SCOPE_anon
;;

type ctxt =
    {
      ctxt_scopes: scope Stack.t;
      ctxt_node_counter: int ref;
      ctxt_sess: Session.sess;
    }

let generate_ident _ : ident =
  let char n =
    if n = 0
    then '_'
    else ident_chars.[Random.int (String.length ident_chars)]
  in
  let i = 3 + (Random.int 10) in
  let s = String.create i in
    for j = 0 to (i-1)
    do
      s.[j] <- char j
    done;
    s
;;

let wrap (n:'a) (cx:ctxt) : 'a identified =
  incr cx.ctxt_node_counter;
  { node = n; id = Node (!(cx.ctxt_node_counter)) }
;;

let generate_in (scope:scope) (fn:(ctxt -> 'a)) (cx:ctxt) : 'a =
  Stack.push scope cx.ctxt_scopes;
  let x = fn cx in
    ignore (Stack.pop cx.ctxt_scopes);
    x
;;

let generate_some (fn:(ctxt -> 'a)) (cx:ctxt) : 'a array =
  let root_count = cx.ctxt_sess.Session.sess_fuzz_item_count in
  let depth = Stack.length cx.ctxt_scopes in
  if depth >= root_count
  then [| |]
  else
    Array.init (1 + (Random.int (root_count - depth)))
      (fun _ -> fn cx)
;;

let rec generate_ty (cx:ctxt) : ty =
  let subty _ =
    generate_in SCOPE_anon
      generate_ty cx
  in
    match Random.int (if Random.bool() then 10 else 17) with
      0 -> TY_nil
    | 1 -> TY_bool

    | 2 -> TY_mach TY_u8
    | 3 -> TY_mach TY_u32

    | 4 -> TY_mach TY_i8
    | 5 -> TY_mach TY_i32

    | 6 -> TY_int
    | 7 -> TY_uint
    | 8 -> TY_char
    | 9 -> TY_str

    | 10 -> TY_tup (generate_in SCOPE_anon
                      (generate_some
                         generate_ty) cx)
    | 11 -> TY_vec (subty())
    | 12 ->
        let generate_elt cx =
          (generate_ident cx, generate_ty cx)
        in
          TY_rec (generate_in SCOPE_anon
                    (generate_some generate_elt) cx)

    | 13 -> TY_chan (subty())
    | 14 -> TY_port (subty())

    | 15 -> TY_task

    | _ -> TY_box (subty())
;;


let rec generate_mod_item (mis:mod_items) (cx:ctxt) : unit =
  let ident = generate_ident () in
  let decl i = wrap { decl_item = i;
                      decl_params = [| |] } cx
  in
  let item =
    match Random.int 2 with
        0 ->
          let ty = generate_ty cx in
          let st = Ast.LAYER_value in
            decl (MOD_ITEM_type (st, ty))
      | _ ->
          let mis' = Hashtbl.create 0 in
          let view = { view_imports = Hashtbl.create 0;
                       view_exports = Hashtbl.create 0; }
          in
          let item =
            decl (MOD_ITEM_mod (view, mis'))
          in
          let scope =
            SCOPE_mod_item (ident, item)
          in
            ignore
              (generate_in scope
                 (generate_some (generate_mod_item mis'))
                 cx);
            item
  in
    Hashtbl.add mis ident item
;;

let fuzz (seed:int option) (sess:Session.sess) : unit =
  begin
    match seed with
        None -> Random.self_init ()
      | Some s -> Random.init s
  end;
  let filename =
    match sess.Session.sess_out with
        Some o -> o
      | None ->
          match seed with
              None -> "fuzz.rs"
            | Some seed -> "fuzz-" ^ (string_of_int seed) ^ ".rs"
  in
  let out = open_out_bin filename in
  let ff = Format.formatter_of_out_channel out in
  let cx = { ctxt_scopes = Stack.create ();
             ctxt_node_counter = ref 0;
             ctxt_sess = sess }
  in
  let mis = Hashtbl.create 0 in
    ignore (generate_some
              (generate_mod_item mis) cx);
    fmt_mod_items ff mis;
    Format.pp_print_flush ff ();
    close_out out;
    exit 0
;;


(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
