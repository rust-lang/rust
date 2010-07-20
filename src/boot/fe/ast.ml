(*
 * There are two kinds of rust files:
 *
 * .rc files, containing crates.
 * .rs files, containing source.
 *
 *)

open Common;;
open Fmt;;

type ident = string
;;

type slot_key =
    KEY_ident of ident
  | KEY_temp of temp_id
;;

(* "names" are statically computable references to particular items;
   they never involve dynamic indexing (nor even static tuple-indexing;
   you could add it but there are few contexts that need names that would
   benefit from it).

   Each component of a name may also be type-parametric; you must
   supply type parameters to reference through a type-parametric name
   component. So for example if foo is parametric in 2 types, you can
   write foo[int,int].bar but not foo.bar.
 *)

type effect =
    PURE
  | IO
  | STATE
  | UNSAFE
;;

type mutability =
    MUT_mutable
  | MUT_immutable
;;

type name_base =
    BASE_ident of ident
  | BASE_temp of temp_id
  | BASE_app of (ident * (ty array))

and name_component =
    COMP_ident of ident
  | COMP_app of (ident * (ty array))
  | COMP_idx of int

and name =
    NAME_base of name_base
  | NAME_ext of (name * name_component)

(*
 * Type expressions are transparent to type names, their equality is
 * structural.  (after normalization)
 *)
and ty =

    TY_any
  | TY_nil
  | TY_bool
  | TY_mach of ty_mach
  | TY_int
  | TY_uint
  | TY_char
  | TY_str

  | TY_tup of ty_tup
  | TY_vec of ty
  | TY_rec of ty_rec

  (*
   * Note that ty_idx is only valid inside a ty of a ty_iso group, not
   * in a general type term.
   *)
  | TY_tag of ty_tag
  | TY_iso of ty_iso
  | TY_idx of int

  | TY_fn of ty_fn
  | TY_chan of ty
  | TY_port of ty

  | TY_obj of ty_obj
  | TY_task

  | TY_native of opaque_id
  | TY_param of (ty_param_idx * effect)
  | TY_named of name
  | TY_type

  | TY_box of ty
  | TY_mutable of ty

  | TY_constrained of (ty * constrs)

(*
 * FIXME: this should be cleaned up to be a different
 * type definition. Only args can be by-ref, only locals
 * can be auto. The structure here is historical.
 *)

and mode =
  | MODE_local
  | MODE_alias

and slot = { slot_mode: mode;
             slot_ty: ty option; }

and ty_tup = ty array

(* In closed type terms a constraint may refer to components of the term by
 * anchoring off the "formal symbol" '*', which represents "the term this
 * constraint is attached to".
 * 
 * 
 * For example, if I have a tuple type tup(int,int), I may wish to enforce the
 * lt predicate on it; I can write this as a constrained type term like:
 * 
 * tup(int,int) : lt( *._0, *._1 )
 * 
 * In fact all tuple types are converted to this form for purpose of
 * type-compatibility testing; the argument tuple in a function
 * 
 * fn (int x, int y) : lt(x, y) -> int
 * 
 * desugars to
 * 
 * fn (tup(int, int) : lt( *._1, *._2 )) -> int
 * 
 *)

and carg_base =
    BASE_formal
  | BASE_named of name_base

and carg_path =
    CARG_base of carg_base
  | CARG_ext of (carg_path * name_component)

and carg =
    CARG_path of carg_path
  | CARG_lit of lit

and constr =
    {
      constr_name: name;
      constr_args: carg array;
    }

and constrs = constr array

and ty_rec = (ident * ty) array

(* ty_tag is a sum type.
 *
 * a tag type expression either normalizes to a TY_tag or a TY_iso,
 * which (like in ocaml) is an indexed projection from an iso-recursive
 * group of TY_tags.
 *)

and ty_tag = (name, ty_tup) Hashtbl.t

and ty_iso =
    {
      iso_index: int;
      iso_group: ty_tag array
    }

and ty_sig =
    {
      sig_input_slots: slot array;
      sig_input_constrs: constrs;
      sig_output_slot: slot;
    }

and ty_fn_aux =
    {
      fn_is_iter: bool;
      fn_effect: effect;
    }

and ty_fn = (ty_sig * ty_fn_aux)

and ty_obj_header = (slot array * constrs)

and ty_obj = (effect * ((ident,ty_fn) Hashtbl.t))

and check_calls = (lval * (atom array)) array

and rec_input = (ident * mutability * atom)

and tup_input = (mutability * atom)

and stmt' =

  (* lval-assigning stmts. *)
    STMT_spawn of (lval * domain * lval * (atom array))
  | STMT_init_rec of (lval * (rec_input array) * lval option)
  | STMT_init_tup of (lval * (tup_input array))
  | STMT_init_vec of (lval * mutability * atom array)
  | STMT_init_str of (lval * string)
  | STMT_init_port of lval
  | STMT_init_chan of (lval * (lval option))
  | STMT_init_box of (lval * mutability * atom)
  | STMT_copy of (lval * expr)
  | STMT_copy_binop of (lval * binop * atom)
  | STMT_call of (lval * lval * (atom array))
  | STMT_bind of (lval * lval * ((atom option) array))
  | STMT_recv of (lval * lval)
  | STMT_slice of (lval * lval * slice)

  (* control-flow stmts. *)
  | STMT_while of stmt_while
  | STMT_do_while of stmt_while
  | STMT_for of stmt_for
  | STMT_for_each of stmt_for_each
  | STMT_if of stmt_if
  | STMT_put of (atom option)
  | STMT_put_each of (lval * (atom array))
  | STMT_ret of (atom option)
  | STMT_be of (lval * (atom array))
  | STMT_alt_tag of stmt_alt_tag
  | STMT_alt_type of stmt_alt_type
  | STMT_alt_port of stmt_alt_port

  (* structural and misc stmts. *)
  | STMT_fail
  | STMT_yield
  | STMT_join of lval
  | STMT_send of (lval * lval)
  | STMT_log of atom
  | STMT_note of atom
  | STMT_prove of (constrs)
  | STMT_check of (constrs * check_calls)
  | STMT_check_expr of expr
  | STMT_check_if of (constrs * check_calls * block)
  | STMT_block of block
  | STMT_decl of stmt_decl

and stmt = stmt' identified

and stmt_alt_tag =
    {
      alt_tag_lval: lval;
      alt_tag_arms: arm array;
    }

and stmt_alt_type =
    {
      alt_type_lval: lval;
      alt_type_arms: (ident * slot * stmt) array;
      alt_type_else: stmt option;
    }

and block' = stmt array
and block = block' identified

and stmt_decl =
    DECL_mod_item of (ident * mod_item)
  | DECL_slot of (slot_key * (slot identified))

and stmt_alt_port =
    {
      (* else lval is a timeout value. *)
      alt_port_arms: (lval * lval) array;
      alt_port_else: (lval * stmt) option;
    }

and stmt_while =
    {
      while_lval: ((stmt array) * expr);
      while_body: block;
    }

and stmt_for_each =
    {
      for_each_slot: (slot identified * ident);
      for_each_call: (lval * atom array);
      for_each_head: block;
      for_each_body: block;
    }

and stmt_for =
    {
      for_slot: (slot identified * ident);
      for_seq: lval;
      for_body: block;
    }

and stmt_if =
    {
      if_test: expr;
      if_then: block;
      if_else: block option;
    }

and slice =
    { slice_start: atom option;
      slice_len: atom option; }

and domain =
    DOMAIN_local
  | DOMAIN_thread

(*
 * PAT_tag uses lval for the tag constructor so that we can reuse our lval
 * resolving machinery.  The lval is restricted during parsing to have only
 * named components.
 *)
and pat =
    PAT_lit of lit
  | PAT_tag of (lval * (pat array))
  | PAT_slot of ((slot identified) * ident)
  | PAT_wild

and arm' = pat * block
and arm = arm' identified

and atom =
    ATOM_literal of (lit identified)
  | ATOM_lval of lval

and expr =
    EXPR_binary of (binop * atom * atom)
  | EXPR_unary of (unop * atom)
  | EXPR_atom of atom

and lit =
  | LIT_nil
  | LIT_bool of bool
  | LIT_mach of (ty_mach * int64 * string)
  | LIT_int of (int64 * string)
  | LIT_uint of (int64 * string)
  | LIT_char of int


and lval_component =
    COMP_named of name_component
  | COMP_atom of atom
  | COMP_deref


(* identifying the name_base here is sufficient to identify the full lval *)
and lval =
    LVAL_base of name_base identified
  | LVAL_ext of (lval * lval_component)

and binop =
    BINOP_or
  | BINOP_and
  | BINOP_xor

  | BINOP_eq
  | BINOP_ne

  | BINOP_lt
  | BINOP_le
  | BINOP_ge
  | BINOP_gt

  | BINOP_lsl
  | BINOP_lsr
  | BINOP_asr

  | BINOP_add
  | BINOP_sub
  | BINOP_mul
  | BINOP_div
  | BINOP_mod
  | BINOP_send

and unop =
    UNOP_not
  | UNOP_bitnot
  | UNOP_neg
  | UNOP_cast of ty identified


and header_slots = ((slot identified) * ident) array

and header_tup = (slot identified) array

and fn =
    {
      fn_input_slots: header_slots;
      fn_input_constrs: constrs;
      fn_output_slot: slot identified;
      fn_aux: ty_fn_aux;
      fn_body: block;
    }

and obj =
    {
      obj_state: header_slots;
      obj_effect: effect;
      obj_constrs: constrs;
      obj_fns: (ident,fn identified) Hashtbl.t;
      obj_drop: block option;
    }

(*
 * An 'a decl is a sort-of-thing that represents a parametric (generative)
 * declaration. Every reference to one of these involves applying 0 or more
 * type arguments, as part of *name resolution*.
 *
 * Slots are *not* parametric declarations. A slot has a specific type
 * even if it's a type that's bound by a quantifier in its environment.
 *)

and ty_param = ident * (ty_param_idx * effect)

and mod_item' =
    MOD_ITEM_type of (effect * ty)
  | MOD_ITEM_tag of (header_tup * ty_tag * node_id)
  | MOD_ITEM_mod of (mod_view * mod_items)
  | MOD_ITEM_fn of fn
  | MOD_ITEM_obj of obj

and mod_item_decl =
    {
      decl_params: (ty_param identified) array;
      decl_item: mod_item';
    }

and mod_item = mod_item_decl identified
and mod_items = (ident, mod_item) Hashtbl.t

and export =
    EXPORT_all_decls
  | EXPORT_ident of ident

and mod_view =
    {
      view_imports: (ident, name) Hashtbl.t;
      view_exports: (export, unit) Hashtbl.t;
    }

and meta = (ident * string) array

and meta_pat = (ident * string option) array

and crate' =
    {
      crate_items: (mod_view * mod_items);
      crate_meta: meta;
      crate_auth: (name, effect) Hashtbl.t;
      crate_required: (node_id, (required_lib * nabi_conv)) Hashtbl.t;
      crate_required_syms: (node_id, string) Hashtbl.t;
      crate_files: (node_id,filename) Hashtbl.t;
      crate_main: name option;
    }
and crate = crate' identified
;;

let empty_crate' =
  { crate_items = ({ view_imports = Hashtbl.create 0;
                     view_exports = Hashtbl.create 0 },
                   Hashtbl.create 0);
    crate_meta = [||];
    crate_auth = Hashtbl.create 0;
    crate_required = Hashtbl.create 0;
    crate_required_syms = Hashtbl.create 0;
    crate_main = None;
    crate_files = Hashtbl.create 0 }
;;

(*
 * NB: names can only be type-parametric in their *last* path-entry.
 * All path-entries before that must be ident or idx (non-parametric).
 *)
let sane_name (n:name) : bool =
  let rec sane_prefix (n:name) : bool =
      match n with
          NAME_base (BASE_ident _)
        | NAME_base (BASE_temp _) -> true
        | NAME_ext (prefix, COMP_ident _)
        | NAME_ext (prefix, COMP_idx _) -> sane_prefix prefix
        | _ -> false
  in
    match n with
        NAME_base _ -> true
      | NAME_ext (prefix, _) -> sane_prefix prefix
;;


(***********************************************************************)

(* FIXME (issue #19): finish all parts with ?foo? as their output. *)

let fmt_ident (ff:Format.formatter) (i:ident) : unit =
  fmt ff  "%s" i

let fmt_temp (ff:Format.formatter) (t:temp_id) : unit =
  fmt ff  ".t%d" (int_of_temp t)

let fmt_slot_key ff (s:slot_key) : unit =
  match s with
      KEY_ident i -> fmt_ident ff i
    | KEY_temp t -> fmt_temp ff t

let rec fmt_app (ff:Format.formatter) (i:ident) (tys:ty array) : unit =
  fmt ff "%s" i;
  fmt_app_args ff tys

and fmt_app_args (ff:Format.formatter) (tys:ty array) : unit =
  fmt ff "[@[";
  for i = 0 to (Array.length tys) - 1;
  do
    if i != 0
    then fmt ff ",@ ";
    fmt_ty ff tys.(i);
  done;
  fmt ff "@]]"

and fmt_name_base (ff:Format.formatter) (nb:name_base) : unit =
  match nb with
      BASE_ident i -> fmt_ident ff i
    | BASE_temp t -> fmt_temp ff t
    | BASE_app (id, tys) -> fmt_app ff id tys

and fmt_name_component (ff:Format.formatter) (nc:name_component) : unit =
  match nc with
      COMP_ident i -> fmt_ident ff i
    | COMP_app (id, tys) -> fmt_app ff id tys
    | COMP_idx i -> fmt ff "_%d" i

and fmt_name (ff:Format.formatter) (n:name) : unit =
  match n with
      NAME_base nb -> fmt_name_base ff nb
    | NAME_ext (n, nc) ->
        fmt_name ff n;
        fmt ff ".";
        fmt_name_component ff nc

and fmt_mode (ff:Format.formatter) (m:mode) : unit =
  match m with
    | MODE_alias -> fmt ff "&"
    | MODE_local -> ()

and fmt_slot (ff:Format.formatter) (s:slot) : unit =
  match s.slot_ty with
      None -> fmt ff "auto"
    | Some t ->
        fmt_mode ff s.slot_mode;
        fmt_ty ff t

and fmt_tys
    (ff:Format.formatter)
    (tys:ty array)
    : unit =
  fmt_bracketed_arr_sep "(" ")" "," fmt_ty ff tys

and fmt_ident_tys
    (ff:Format.formatter)
    (entries:(ident * ty) array)
    : unit =
  fmt_bracketed_arr_sep "(" ")" ","
    (fun ff (ident, ty) ->
       fmt_ty ff ty;
       fmt ff " ";
       fmt_ident ff ident)
    ff
    entries

and fmt_slots
    (ff:Format.formatter)
    (slots:slot array)
    (idents:(ident array) option)
    : unit =
  fmt ff "(@[";
  for i = 0 to (Array.length slots) - 1
  do
    if i != 0
    then fmt ff ",@ ";
    fmt_slot ff slots.(i);
    begin
      match idents with
          None -> ()
        | Some ids -> (fmt ff " "; fmt_ident ff ids.(i))
    end;
  done;
  fmt ff "@])"

and fmt_effect
    (ff:Format.formatter)
    (effect:effect)
    : unit =
  match effect with
      PURE -> ()
    | IO -> fmt ff "io"
    | STATE -> fmt ff "state"
    | UNSAFE -> fmt ff "unsafe"

and fmt_ty_fn
    (ff:Format.formatter)
    (ident_and_params:(ident * ty_param array) option)
    (tf:ty_fn)
    : unit =
  let (tsig, ta) = tf in
    fmt_effect ff ta.fn_effect;
    if ta.fn_effect <> PURE then fmt ff " ";
    fmt ff "%s" (if ta.fn_is_iter then "iter" else "fn");
    begin
      match ident_and_params with
          Some (id, params) ->
            fmt ff " ";
            fmt_ident_and_params ff id params
        | None -> ()
    end;
    fmt_slots ff tsig.sig_input_slots None;
    fmt_decl_constrs ff tsig.sig_input_constrs;
    fmt ff " -> ";
    fmt_slot ff tsig.sig_output_slot;

and fmt_tag (ff:Format.formatter) (ttag:ty_tag) : unit =
  fmt ff "@[tag(@[";
  let first = ref true in
    Hashtbl.iter
      begin
        fun name ttup ->
          (if !first
           then first := false
           else fmt ff ",@ ");
          fmt_name ff name;
          fmt_tys ff ttup
      end
      ttag;
    fmt ff "@])@]"

and fmt_iso (ff:Format.formatter) (tiso:ty_iso) : unit =
  fmt ff "@[iso [@[";
  for i = 0 to (Array.length tiso.iso_group) - 1
  do
    if i != 0
    then fmt ff ",@ ";
    if i == tiso.iso_index
    then fmt ff "<%d>: " i
    else fmt ff "%d: " i;
    fmt_tag ff tiso.iso_group.(i);
  done;
  fmt ff "@]]@]"

and fmt_ty (ff:Format.formatter) (t:ty) : unit =
  match t with
    TY_any -> fmt ff "any"
  | TY_nil -> fmt ff "()"
  | TY_bool -> fmt ff "bool"
  | TY_mach m -> fmt_mach ff m
  | TY_int -> fmt ff "int"
  | TY_uint -> fmt ff "uint"
  | TY_char -> fmt ff "char"
  | TY_str -> fmt ff "str"

  | TY_tup tys -> (fmt ff "tup"; fmt_tys ff tys)
  | TY_vec t -> (fmt ff "vec["; fmt_ty ff t; fmt ff "]")
  | TY_chan t -> (fmt ff "chan["; fmt_ty ff t; fmt ff "]")
  | TY_port t -> (fmt ff "port["; fmt_ty ff t; fmt ff "]")

  | TY_rec entries ->
      fmt ff "@[rec";
      fmt_ident_tys ff entries;
      fmt ff "@]"

  | TY_param (i, e) -> (fmt_effect ff e;
                        if e <> PURE then fmt ff " ";
                        fmt ff "<p#%d>" i)
  | TY_native oid -> fmt ff "<native#%d>" (int_of_opaque oid)
  | TY_named n -> fmt_name ff n
  | TY_type -> fmt ff "type"

  | TY_box t ->
      fmt ff "@@";
      fmt_ty ff t

  | TY_mutable t ->
      fmt ff "mutable ";
      fmt_ty ff t

  | TY_fn tfn -> fmt_ty_fn ff None tfn
  | TY_task -> fmt ff "task"
  | TY_tag ttag -> fmt_tag ff ttag
  | TY_iso tiso -> fmt_iso ff tiso
  | TY_idx idx -> fmt ff "<idx#%d>" idx
  | TY_constrained _ -> fmt ff "?constrained?"

  | TY_obj (effect, fns) ->
      fmt_obox ff;
      fmt_effect ff effect;
      if effect <> PURE then fmt ff " ";
      fmt ff "obj ";
      fmt_obr ff;
      Hashtbl.iter
        begin
          fun id fn ->
            fmt ff "@\n";
            fmt_ty_fn ff (Some (id, [||])) fn;
            fmt ff ";"
        end
        fns;
      fmt_cbb ff


and fmt_constrs (ff:Format.formatter) (cc:constr array) : unit =
  Array.iter (fmt_constr ff) cc

and fmt_decl_constrs (ff:Format.formatter) (cc:constr array) : unit =
  if Array.length cc = 0
  then ()
  else
    begin
      fmt ff " : ";
      fmt_constrs ff cc
    end

and fmt_constr (ff:Format.formatter) (c:constr) : unit =
  fmt_name ff c.constr_name;
  fmt ff "(@[";
  for i = 0 to (Array.length c.constr_args) - 1
  do
    if i != 0
    then fmt ff ",@ ";
    fmt_carg ff c.constr_args.(i);
  done;
  fmt ff "@])"

and fmt_carg_path (ff:Format.formatter) (cp:carg_path) : unit =
  match cp with
      CARG_base BASE_formal -> fmt ff "*"
    | CARG_base (BASE_named nb) -> fmt_name_base ff nb
    | CARG_ext (base, nc) ->
        fmt_carg_path ff base;
        fmt ff ".";
        fmt_name_component ff nc

and fmt_carg (ff:Format.formatter) (ca:carg) : unit =
  match ca with
      CARG_path cp -> fmt_carg_path ff cp
    | CARG_lit lit -> fmt_lit ff lit

and fmt_stmts (ff:Format.formatter) (ss:stmt array) : unit =
  Array.iter (fmt_stmt ff) ss;

and fmt_block (ff:Format.formatter) (b:stmt array) : unit =
  fmt_obox ff;
  fmt_obr ff;
  fmt_stmts ff b;
  fmt_cbb ff;

and fmt_binop (ff:Format.formatter) (b:binop) : unit =
  fmt ff "%s"
    begin
      match b with
          BINOP_or -> "|"
        | BINOP_and -> "&"
        | BINOP_xor -> "^"

        | BINOP_eq -> "=="
        | BINOP_ne -> "!="

        | BINOP_lt -> "<"
        | BINOP_le -> "<="
        | BINOP_ge -> ">="
        | BINOP_gt -> ">"

        | BINOP_lsl -> "<<"
        | BINOP_lsr -> ">>"
        | BINOP_asr -> ">>>"

        | BINOP_add -> "+"
        | BINOP_sub -> "-"
        | BINOP_mul -> "*"
        | BINOP_div -> "/"
        | BINOP_mod -> "%"
        | BINOP_send -> "<|"
    end


and fmt_unop (ff:Format.formatter) (u:unop) (a:atom) : unit =
  begin
    match u with
        UNOP_not ->
          fmt ff "!";
          fmt_atom ff a

      | UNOP_bitnot ->
          fmt ff "~";
          fmt_atom ff a

      | UNOP_neg ->
          fmt ff "-";
          fmt_atom ff a

      | UNOP_cast t ->
          fmt_atom ff a;
          fmt ff " as ";
          fmt_ty ff t.node;
  end

and fmt_expr (ff:Format.formatter) (e:expr) : unit =
  match e with
    EXPR_binary (b,a1,a2) ->
      begin
        fmt_atom ff a1;
        fmt ff " ";
        fmt_binop ff b;
        fmt ff " ";
        fmt_atom ff a2
      end
  | EXPR_unary (u,a) ->
      begin
        fmt_unop ff u a;
      end
  | EXPR_atom a -> fmt_atom ff a

and fmt_mach (ff:Format.formatter) (m:ty_mach) : unit =
  match m with
    TY_u8 -> fmt ff "u8"
  | TY_u16 -> fmt ff "u16"
  | TY_u32 -> fmt ff "u32"
  | TY_u64 -> fmt ff "u64"
  | TY_i8 -> fmt ff "i8"
  | TY_i16 -> fmt ff "i16"
  | TY_i32 -> fmt ff "i32"
  | TY_i64 -> fmt ff "i64"
  | TY_f32 -> fmt ff "f32"
  | TY_f64 -> fmt ff "f64"

and fmt_lit (ff:Format.formatter) (l:lit) : unit =
  match l with
  | LIT_nil -> fmt ff "()"
  | LIT_bool true -> fmt ff "true"
  | LIT_bool false -> fmt ff "false"
  | LIT_mach (m, _, s) ->
      begin
        fmt_mach ff m;
        fmt ff "(%s)" s
      end
  | LIT_int (_,s) -> fmt ff "%s" s
  | LIT_uint (_,s) -> fmt ff "%s" s
  | LIT_char c -> fmt ff "'%s'" (Common.escaped_char c)

and fmt_domain (ff:Format.formatter) (d:domain) : unit =
  match d with
      DOMAIN_local -> ()
    | DOMAIN_thread -> fmt ff "thread "

and fmt_atom (ff:Format.formatter) (a:atom) : unit =
  match a with
      ATOM_literal lit -> fmt_lit ff lit.node
    | ATOM_lval lval -> fmt_lval ff lval

and fmt_atoms (ff:Format.formatter) (az:atom array) : unit =
  fmt ff "(";
  Array.iteri
    begin
      fun i a ->
        if i != 0
        then fmt ff ", ";
        fmt_atom ff a;
    end
    az;
  fmt ff ")"

and fmt_atom_opts (ff:Format.formatter) (az:(atom option) array) : unit =
  fmt ff "(";
  Array.iteri
    begin
      fun i a ->
        if i != 0
        then fmt ff ", ";
        match a with
            None -> fmt ff "_"
          | Some a -> fmt_atom ff a;
    end
    az;
  fmt ff ")"

and fmt_lval (ff:Format.formatter) (l:lval) : unit =
  match l with
      LVAL_base nbi -> fmt_name_base ff nbi.node
    | LVAL_ext (lv, lvc) ->
        begin
          match lvc with
              COMP_named nc ->
                fmt_lval ff lv;
                fmt ff ".";
                fmt_name_component ff nc
            | COMP_atom a ->
                fmt_lval ff lv;
                fmt ff ".";
                fmt_bracketed "(" ")" fmt_atom ff a;
            | COMP_deref ->
                fmt ff "*";
                fmt_lval ff lv
        end

and fmt_stmt (ff:Format.formatter) (s:stmt) : unit =
  fmt ff "@\n";
  fmt_stmt_body ff s

and fmt_stmt_body (ff:Format.formatter) (s:stmt) : unit =
  begin
    match s.node with
        STMT_log at ->
          begin
            fmt ff "log ";
            fmt_atom ff at;
            fmt ff ";"
          end

      | STMT_spawn (dst, domain, fn, args) ->
          fmt_lval ff dst;
          fmt ff " = spawn ";
          fmt_domain ff domain;
          fmt_lval ff fn;
          fmt_atoms ff args;
          fmt ff ";";

      | STMT_while sw ->
          let (stmts, e) = sw.while_lval in
            begin
              fmt_obox ff;
              fmt ff "while (";
              if Array.length stmts != 0
              then fmt_block ff stmts;
              fmt_expr ff e;
              fmt ff ") ";
              fmt_obr ff;
              fmt_stmts ff sw.while_body.node;
              fmt_cbb ff
            end

      | STMT_do_while sw ->
          let (stmts, e) = sw.while_lval in
            begin
              fmt_obox ff;
              fmt ff "do ";
              fmt_obr ff;
              fmt_stmts ff sw.while_body.node;
              fmt ff "while (";
              if Array.length stmts != 0
              then fmt_block ff stmts;
              fmt_expr ff e;
              fmt ff ");";
              fmt_cbb ff
            end

      | STMT_if sif ->
          fmt_obox ff;
          fmt ff "if (";
          fmt_expr ff sif.if_test;
          fmt ff ") ";
          fmt_obr ff;
          fmt_stmts ff sif.if_then.node;
          begin
            match sif.if_else with
                None -> ()
              | Some e ->
                  begin
                    fmt_cbb ff;
                    fmt_obox_3 ff;
                    fmt ff " else ";
                    fmt_obr ff;
                    fmt_stmts ff e.node
                  end
          end;
          fmt_cbb ff

      | STMT_ret (ao) ->
          fmt ff "ret";
          begin
            match ao with
                None -> ()
              | Some at ->
                  fmt ff " ";
                  fmt_atom ff at
          end;
          fmt ff ";"

      | STMT_be (fn, az) ->
          fmt ff "be ";
          fmt_lval ff fn;
          fmt_atoms ff az;
          fmt ff ";";

      | STMT_block b -> fmt_block ff b.node

      | STMT_copy (lv, ex) ->
          fmt_lval ff lv;
          fmt ff " = ";
          fmt_expr ff ex;
          fmt ff ";"

      | STMT_copy_binop (lv, binop, at) ->
          fmt_lval ff lv;
          fmt ff " ";
          fmt_binop ff binop;
          fmt ff "= ";
          fmt_atom ff at;
          fmt ff ";"

      | STMT_call (dst, fn, args) ->
          fmt_lval ff dst;
          fmt ff " = ";
          fmt_lval ff fn;
          fmt_atoms ff args;
          fmt ff ";";

      | STMT_bind (dst, fn, arg_opts) ->
          fmt_lval ff dst;
          fmt ff " = ";
          fmt_lval ff fn;
          fmt_atom_opts ff arg_opts;
          fmt ff ";";

      | STMT_decl (DECL_slot (skey, sloti)) ->
          if sloti.node.slot_ty != None then fmt ff "let ";
          fmt_slot ff sloti.node;
          fmt ff " ";
          fmt_slot_key ff skey;
          fmt ff ";"

      | STMT_decl (DECL_mod_item (ident, item)) ->
          fmt_mod_item ff ident item

      | STMT_init_rec (dst, entries, base) ->
          fmt_lval ff dst;
          fmt ff " = rec(";
          for i = 0 to (Array.length entries) - 1
          do
            if i != 0
            then fmt ff ", ";
            let (ident, mutability, atom) = entries.(i) in
              if mutability = MUT_mutable then fmt ff "mutable ";
              fmt_ident ff ident;
              fmt ff " = ";
              fmt_atom ff atom;
          done;
          begin
            match base with
                None -> ()
              | Some b ->
                  fmt ff " with ";
                  fmt_lval ff b
          end;
          fmt ff ");"

      | STMT_init_vec (dst, mutability, atoms) ->
          fmt_lval ff dst;
          fmt ff " = vec";
          if mutability = MUT_mutable then fmt ff "[mutable]";
          fmt ff "(";
          for i = 0 to (Array.length atoms) - 1
          do
            if i != 0
            then fmt ff ", ";
            fmt_atom ff atoms.(i);
          done;
          fmt ff ");"

      | STMT_init_tup (dst, entries) ->
          fmt_lval ff dst;
          fmt ff " = tup(";
          for i = 0 to (Array.length entries) - 1
          do
            if i != 0
            then fmt ff ", ";
            let (mutability, atom) = entries.(i) in
            if mutability = MUT_mutable then fmt ff "mutable ";
            fmt_atom ff atom;
          done;
          fmt ff ");";

      | STMT_init_str (dst, s) ->
          fmt_lval ff dst;
          fmt ff " = \"%s\"" (String.escaped s)

      | STMT_init_port dst ->
          fmt_lval ff dst;
          fmt ff " = port();"

      | STMT_init_chan (dst, port_opt) ->
          fmt_lval ff dst;
          fmt ff " = chan(";
          begin
            match port_opt with
                None -> ()
              | Some lv -> fmt_lval ff lv
          end;
          fmt ff ");"

      | STMT_check_expr expr ->
          fmt ff "check (";
          fmt_expr ff expr;
          fmt ff ");"

      | STMT_check_if (constrs, _, block) ->
          fmt_obox ff;
          fmt ff "check if (";
          fmt_constrs ff constrs;
          fmt ff ")";
          fmt_obr ff;
          fmt_stmts ff block.node;
          fmt_cbb ff

      | STMT_check (constrs, _) ->
          fmt ff "check ";
          fmt_constrs ff constrs;
          fmt ff ";"

      | STMT_prove constrs ->
          fmt ff "prove ";
          fmt_constrs ff constrs;
          fmt ff ";"

      | STMT_for sfor ->
          let (slot, ident) = sfor.for_slot in
          let lval = sfor.for_seq in
            begin
              fmt_obox ff;
              fmt ff "for (";
              fmt_slot ff slot.node;
              fmt ff " ";
              fmt_ident ff ident;
              fmt ff " in ";
              fmt_lval ff lval;
              fmt ff ") ";
              fmt_obr ff;
              fmt_stmts ff sfor.for_body.node;
              fmt_cbb ff
            end

      | STMT_for_each sf ->
          let (slot, ident) = sf.for_each_slot in
          let (f, az) = sf.for_each_call in
            begin
              fmt_obox ff;
              fmt ff "for each (";
              fmt_slot ff slot.node;
              fmt ff " ";
              fmt_ident ff ident;
              fmt ff " in ";
              fmt_lval ff f;
              fmt_atoms ff az;
              fmt ff ") ";
              fmt_obr ff;
              fmt_stmts ff sf.for_each_body.node;
              fmt_cbb ff
            end

      | STMT_put (atom) ->
          fmt ff "put";
          begin
            match atom with
                Some a -> (fmt ff " "; fmt_atom ff a)
              | None -> ()
          end;
          fmt ff ";"

      | STMT_put_each (f, az) ->
          fmt ff "put each ";
          fmt_lval ff f;
          fmt_atoms ff az;
          fmt ff ";"

      | STMT_fail -> fmt ff "fail;"
      | STMT_yield -> fmt ff "yield;"

      | STMT_send (chan, v) ->
          fmt_lval ff chan;
          fmt ff " <| ";
          fmt_lval ff v;
          fmt ff ";";

      | STMT_recv (d, port) ->
          fmt_lval ff d;
          fmt ff " <- ";
          fmt_lval ff port;
          fmt ff ";";

      | STMT_join t ->
          fmt ff "join ";
          fmt_lval ff t;
          fmt ff ";"

      | STMT_init_box (lv, mutability, at) ->
          fmt_lval ff lv;
          fmt ff " = @@";
          if mutability = MUT_mutable then fmt ff " mutable ";
          fmt_atom ff at;
          fmt ff ";"

      | STMT_alt_tag _ -> fmt ff "?stmt_alt_tag?"
      | STMT_alt_type _ -> fmt ff "?stmt_alt_type?"
      | STMT_alt_port _ -> fmt ff "?stmt_alt_port?"
      | STMT_note _ -> fmt ff "?stmt_note?"
      | STMT_slice _ -> fmt ff "?stmt_slice?"
  end

and fmt_decl_param (ff:Format.formatter) (param:ty_param) : unit =
  let (ident, (i, e)) = param in
  fmt_effect ff e;
  if e <> PURE then fmt ff " ";
  fmt_ident ff ident;
  fmt ff "=<p#%d>" i

and fmt_decl_params (ff:Format.formatter) (params:ty_param array) : unit =
  if Array.length params = 0
  then ()
  else
    begin
      fmt ff "[";
      for i = 0 to (Array.length params) - 1
      do
        if i <> 0
        then fmt ff ", ";
        fmt_decl_param ff params.(i)
      done;
      fmt ff "]"
    end;

and fmt_header_slots (ff:Format.formatter) (hslots:header_slots) : unit =
  fmt_slots ff
    (Array.map (fun (s,_) -> s.node) hslots)
    (Some (Array.map (fun (_, i) -> i) hslots))

and fmt_ident_and_params
    (ff:Format.formatter)
    (id:ident)
    (params:ty_param array)
    : unit =
  fmt_ident ff id;
  fmt_decl_params ff params

and fmt_effect_qual (ff:Format.formatter) (e:effect) : unit =
  fmt_effect ff e;
  if e <> PURE then fmt ff " ";

and fmt_fn
    (ff:Format.formatter)
    (id:ident)
    (params:ty_param array)
    (f:fn)
    : unit =
  fmt_obox ff;
  fmt_effect_qual ff f.fn_aux.fn_effect;
  fmt ff "%s "(if f.fn_aux.fn_is_iter then "iter" else "fn");
  fmt_ident_and_params ff id params;
  fmt_header_slots ff f.fn_input_slots;
  fmt_decl_constrs ff f.fn_input_constrs;
  fmt ff " -> ";
  fmt_slot ff f.fn_output_slot.node;
  fmt ff " ";
  fmt_obr ff;
  fmt_stmts ff f.fn_body.node;
  fmt_cbb ff


and fmt_obj
    (ff:Format.formatter)
    (id:ident)
    (params:ty_param array)
    (obj:obj)
    : unit =
  fmt_obox ff;
  fmt_effect_qual ff obj.obj_effect;
  fmt ff "obj ";
  fmt_ident_and_params ff id params;
  fmt_header_slots ff obj.obj_state;
  fmt_decl_constrs ff obj.obj_constrs;
  fmt ff " ";
  fmt_obr ff;
  Hashtbl.iter
    begin
      fun id fn ->
        fmt ff "@\n";
        fmt_fn ff id [||] fn.node
    end
    obj.obj_fns;
  begin
    match obj.obj_drop with
        None -> ()
      | Some d ->
          begin
            fmt ff "@\n";
            fmt_obox ff;
            fmt ff "drop ";
            fmt_obr ff;
            fmt_stmts ff d.node;
            fmt_cbb ff;
          end
  end;
  fmt_cbb ff


and fmt_mod_item (ff:Format.formatter) (id:ident) (item:mod_item) : unit =
  fmt ff "@\n";
  let params = item.node.decl_params in
  let params = Array.map (fun i -> i.node) params in
    begin
      match item.node.decl_item with
          MOD_ITEM_type (e, ty) ->
            fmt_effect_qual ff e;
            fmt ff "type ";
            fmt_ident_and_params ff id params;
            fmt ff " = ";
            fmt_ty ff ty;
            fmt ff ";";

        | MOD_ITEM_tag (hdr, ttag, _) ->
            fmt ff "fn ";
            fmt_ident_and_params ff id params;
            fmt_header_slots ff
              (Array.mapi (fun i s -> (s,(Printf.sprintf "_%d" i))) hdr);
            fmt ff " -> ";
            fmt_ty ff (TY_tag ttag);
            fmt ff ";";

        | MOD_ITEM_mod (view,items) ->
            fmt_obox ff;
            fmt ff "mod ";
            fmt_ident_and_params ff id params;
            fmt ff " ";
            fmt_obr ff;
            fmt_mod_view ff view;
            fmt_mod_items ff items;
            fmt_cbb ff

        | MOD_ITEM_fn f ->
            fmt_fn ff id params f

        | MOD_ITEM_obj obj ->
            fmt_obj ff id params obj
    end

and fmt_import (ff:Format.formatter) (ident:ident) (name:name) : unit =
  fmt ff "@\n";
  fmt ff "import ";
  fmt ff "%s = " ident;
  fmt_name ff name;

and fmt_export (ff:Format.formatter) (export:export) _ : unit =
  fmt ff "@\n";
  match export with
      EXPORT_all_decls -> fmt ff "export *;"
    | EXPORT_ident i -> fmt ff "export %s;" i

and fmt_mod_view (ff:Format.formatter) (mv:mod_view) : unit =
  Hashtbl.iter (fmt_import ff) mv.view_imports;
  Hashtbl.iter (fmt_export ff) mv.view_exports

and fmt_mod_items (ff:Format.formatter) (mi:mod_items) : unit =
  Hashtbl.iter (fmt_mod_item ff) mi

and fmt_crate (ff:Format.formatter) (c:crate) : unit =
  let (view,items) = c.node.crate_items in
    fmt_mod_view ff view;
    fmt_mod_items ff items
;;

let ty_children (ty:ty) : ty array =
  let children_of_ty_tag ty_tag = Array.concat (htab_vals ty_tag) in
  let children_of_ty_fn ty_fn =
    let (ty_sig, _) = ty_fn in
    let in_slots = ty_sig.sig_input_slots in
    let slots = Array.append in_slots [| ty_sig.sig_output_slot |] in
    arr_filter_some (Array.map (fun slot -> slot.slot_ty) slots)
  in
  match ty with
      TY_tup tys -> tys
    | TY_vec ty' | TY_chan ty' | TY_port ty' | TY_box ty' | TY_mutable ty'
          | TY_constrained (ty', _) ->
        [| ty' |]
    | TY_rec fields -> Array.map snd fields
    | TY_tag ty_tag -> children_of_ty_tag ty_tag
    | TY_iso ty_iso ->
        children_of_ty_tag (ty_iso.iso_group.(ty_iso.iso_index))
    | TY_fn ty_fn -> children_of_ty_fn ty_fn
    | TY_obj (_, methods) ->
        Array.concat (List.map children_of_ty_fn (htab_vals methods))
    | TY_any | TY_nil | TY_bool | TY_mach _ | TY_int | TY_uint | TY_char
          | TY_str | TY_idx _ | TY_task | TY_native _ | TY_param _
          | TY_named _ | TY_type ->
        [| |]
;;

let sprintf_binop = sprintf_fmt fmt_binop;;
let sprintf_expr = sprintf_fmt fmt_expr;;
let sprintf_name = sprintf_fmt fmt_name;;
let sprintf_name_component = sprintf_fmt fmt_name_component;;
let sprintf_lval = sprintf_fmt fmt_lval;;
let sprintf_atom = sprintf_fmt fmt_atom;;
let sprintf_slot = sprintf_fmt fmt_slot;;
let sprintf_slot_key = sprintf_fmt fmt_slot_key;;
let sprintf_ty = sprintf_fmt fmt_ty;;
let sprintf_effect = sprintf_fmt fmt_effect;;
let sprintf_tag = sprintf_fmt fmt_tag;;
let sprintf_carg = sprintf_fmt fmt_carg;;
let sprintf_constr = sprintf_fmt fmt_constr;;
let sprintf_mod_item =
  sprintf_fmt (fun ff (id,item) -> fmt_mod_item ff id item);;
let sprintf_mod_items = sprintf_fmt fmt_mod_items;;
let sprintf_decl_param = sprintf_fmt fmt_decl_param;;
let sprintf_decl_params = sprintf_fmt fmt_decl_params;;
let sprintf_app_args = sprintf_fmt fmt_app_args;;

(* You probably want this one; stmt has a leading \n *)
let sprintf_stmt = sprintf_fmt fmt_stmt_body;;

(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
