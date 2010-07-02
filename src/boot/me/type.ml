open Common;;
open Semant;;

type tyspec =
    TYSPEC_equiv of tyvar
  | TYSPEC_all
  | TYSPEC_resolved of (Ast.ty_param array) * Ast.ty
  | TYSPEC_box of tyvar                       (* @ of some t *)
  | TYSPEC_mutable of tyvar                   (* something mutable *)
  | TYSPEC_callable of (tyvar * tyvar array)  (* out, ins *)
  | TYSPEC_collection of tyvar                (* vec or str *)
  | TYSPEC_comparable                         (* comparable with = and != *)
  | TYSPEC_plusable                           (* nums, vecs, and strings *)
  | TYSPEC_dictionary of dict
  | TYSPEC_integral                           (* int-like *)
  | TYSPEC_numeric                            (* int-like or float-like *)
  | TYSPEC_ordered                            (* comparable with < etc. *)
  | TYSPEC_record of dict
  | TYSPEC_tuple of tyvar array               (* heterogeneous tuple *)
  | TYSPEC_vector of tyvar
  | TYSPEC_app of (tyvar * Ast.ty array)

and dict = (Ast.ident, tyvar) Hashtbl.t

and tyvar = tyspec ref;;

(* Signatures for binary operators. *)
type binopsig =
    BINOPSIG_bool_bool_bool     (* bool * bool -> bool *)
  | BINOPSIG_comp_comp_bool     (* comparable a * comparable a -> bool *)
  | BINOPSIG_ord_ord_bool       (* ordered a * ordered a -> bool *)
  | BINOPSIG_integ_integ_integ  (* integral a * integral a -> integral a *)
  | BINOPSIG_num_num_num        (* numeric a * numeric a -> numeric a *)
  | BINOPSIG_plus_plus_plus     (* plusable a * plusable a -> plusable a *)
;;


let rec tyspec_to_str (ts:tyspec) : string =

  let fmt = Format.fprintf in
  let fmt_ident (ff:Format.formatter) (i:Ast.ident) : unit =
    fmt ff  "%s" i
  in
  let fmt_obox ff = Format.pp_open_box ff 4 in
  let fmt_cbox ff = Format.pp_close_box ff () in
  let fmt_obr ff = fmt ff "<" in
  let fmt_cbr ff = fmt ff ">" in
  let fmt_obb ff = (fmt_obox ff; fmt_obr ff) in
  let fmt_cbb ff = (fmt_cbox ff; fmt_cbr ff) in

  let rec fmt_fields (flav:string) (ff:Format.formatter) (flds:dict) : unit =
    fmt_obb ff;
    fmt ff "%s :" flav;
    let fmt_entry ident tv =
      fmt ff "@\n";
      fmt_ident ff ident;
      fmt ff " : ";
      fmt_tyspec ff (!tv);
    in
      Hashtbl.iter fmt_entry flds;
      fmt_cbb ff

  and fmt_app ff tv args =
    begin
      assert (Array.length args <> 0);
      fmt_obb ff;
      fmt ff "app(";
      fmt_tyspec ff (!tv);
      fmt ff ")";
      Ast.fmt_app_args ff args;
      fmt_cbb ff;
    end

  and fmt_tvs ff tvs =
    fmt_obox ff;
    let fmt_tv i tv =
      if i <> 0
      then fmt ff ", ";
      fmt_tyspec ff (!tv)
    in
      Array.iteri fmt_tv tvs;
      fmt_cbox ff;

  and fmt_tyspec ff ts =
    match ts with
        TYSPEC_all -> fmt ff "<?>"
      | TYSPEC_comparable -> fmt ff "<comparable>"
      | TYSPEC_plusable -> fmt ff "<plusable>"
      | TYSPEC_integral -> fmt ff "<integral>"
      | TYSPEC_numeric -> fmt ff "<numeric>"
      | TYSPEC_ordered -> fmt ff "<ordered>"
      | TYSPEC_resolved (params, ty) ->
          if Array.length params <> 0
          then
            begin
              fmt ff "abs";
              Ast.fmt_decl_params ff params;
              fmt ff "(";
              Ast.fmt_ty ff ty;
              fmt ff ")"
            end
          else
            Ast.fmt_ty ff ty

      | TYSPEC_equiv tv ->
          fmt_tyspec ff (!tv)

      | TYSPEC_box tv ->
          fmt_obr ff;
          fmt ff "box ";
          fmt_tyspec ff (!tv);
          fmt_cbr ff;

      | TYSPEC_mutable tv ->
          fmt_obr ff;
          fmt ff "mut ";
          fmt_tyspec ff (!tv);
          fmt_cbr ff

      | TYSPEC_callable (out, ins) ->
          fmt_obb ff;
          fmt ff "callable fn(";
          fmt_tvs ff ins;
          fmt ff ") -> ";
          fmt_tyspec ff (!out);
          fmt_cbb ff;

      | TYSPEC_collection tv ->
          fmt_obb ff;
          fmt ff "collection : ";
          fmt_tyspec ff (!tv);
          fmt_cbb ff;

      | TYSPEC_tuple tvs ->
          fmt_obr ff;
          fmt ff "tuple (";
          fmt_tvs ff tvs;
          fmt ff ")";
          fmt_cbr ff;

      | TYSPEC_vector tv ->
          fmt_obb ff;
          fmt ff "vector ";
          fmt_tyspec ff (!tv);
          fmt_cbb ff;

      | TYSPEC_dictionary dct ->
          fmt_fields "dictionary" ff dct

      | TYSPEC_record dct ->
          fmt_fields "record" ff dct

      | TYSPEC_app (tv, args) ->
          fmt_app ff tv args

  in
  let buf = Buffer.create 16 in
  let bf = Format.formatter_of_buffer buf in
    begin
      fmt_tyspec bf ts;
      Format.pp_print_flush bf ();
      Buffer.contents buf
    end
;;

let iflog cx thunk =
  if cx.ctxt_sess.Session.sess_log_type
  then thunk ()
  else ()
;;

let rec resolve_tyvar (tv:tyvar) : tyvar =
  match !tv with
      TYSPEC_equiv subtv -> resolve_tyvar subtv
    | _ -> tv
;;

type unify_ctxt =
    { mut_ok: bool;
      box_ok: bool }
;;

let arg_pass_ctx =
  { box_ok = false;
    mut_ok = true }
;;

let rval_ctx =
  { box_ok = true;
    mut_ok = true }
;;

let lval_ctx =
  { box_ok = false;
    mut_ok = true }
;;

let init_ctx =
  { box_ok = true;
    mut_ok = true }
;;

let strict_ctx =
  { box_ok = false;
    mut_ok = false }
;;


let process_crate (cx:ctxt) (crate:Ast.crate) : unit =

  let depth = ref 0 in

  let log cx = Session.log "type"
    cx.ctxt_sess.Session.sess_log_type
    cx.ctxt_sess.Session.sess_log_out
  in

  let retval_tvs = Stack.create () in
  let push_retval_tv tv =
    Stack.push tv retval_tvs
  in
  let pop_retval_tv _ =
    ignore (Stack.pop retval_tvs)
  in
  let retval_tv _ =
    Stack.top retval_tvs
  in

  let pat_tvs = Stack.create () in
  let push_pat_tv tv =
    Stack.push tv pat_tvs
  in
  let pop_pat_tv _ =
    ignore (Stack.pop pat_tvs)
  in
  let pat_tv _ =
    Stack.top pat_tvs
  in

  let (bindings:(node_id, tyvar) Hashtbl.t) = Hashtbl.create 10 in
  let (item_params:(node_id, tyvar array) Hashtbl.t) = Hashtbl.create 10 in
  let (lval_tyvars:(node_id, tyvar) Hashtbl.t) = Hashtbl.create 0 in

  let path = Stack.create () in

  let visitor (cx:ctxt) (inner:Walk.visitor) : Walk.visitor =

    let rec unify_slot
        (ucx:unify_ctxt)
        (slot:Ast.slot)
        (id_opt:node_id option)
        (tv:tyvar) : unit =
      match id_opt with
          Some id ->
            unify_tyvars ucx (Hashtbl.find bindings id) tv
        | None ->
            match slot.Ast.slot_ty with
                None -> bug () "untyped unidentified slot"
              | Some ty -> unify_ty ucx ty tv

    and check_sane_tyvar tv =
      match !tv with
          TYSPEC_resolved (_, (Ast.TY_named _)) ->
            bug () "named-type in type checker"
        | _ -> ()

    and unify_tyvars (ucx:unify_ctxt) (av:tyvar) (bv:tyvar) : unit =
      let indent = String.make (4 * (!depth)) ' ' in
        iflog cx
          (fun _ ->
             log cx "%s> unifying types:" indent;
             if ucx.box_ok || ucx.mut_ok
             then
               log cx "%s> (w/ %s%s%s)"
                 indent
                 (if ucx.box_ok then "ext-ok" else "")
                 (if ucx.box_ok && ucx.mut_ok then " " else "")
                 (if ucx.mut_ok then "mut-ok" else "");
             log cx "%s> input tyvar A:     %s" indent (tyspec_to_str !av);
             log cx "%s> input tyvar B:     %s" indent (tyspec_to_str !bv));
        check_sane_tyvar av;
        check_sane_tyvar bv;

        incr depth;
        unify_tyvars' ucx av bv;
        decr depth;

        iflog cx
          (fun _ ->
             log cx "%s< unified types:" indent;
             log cx "%s< output tyvar A:     %s" indent (tyspec_to_str !av);
             log cx "%s< output tyvar B:     %s" indent (tyspec_to_str !bv));
        check_sane_tyvar av;
        check_sane_tyvar bv;

    and unify_tyvars' (ucx:unify_ctxt) (av:tyvar) (bv:tyvar) : unit =
      let (a, b) = ((resolve_tyvar av), (resolve_tyvar bv)) in
      let fail () =
        err None "mismatched types: %s vs. %s" (tyspec_to_str !av)
          (tyspec_to_str !bv);
      in

      let merge_dicts a b =
        let c = Hashtbl.create ((Hashtbl.length a) + (Hashtbl.length b)) in
        let merge ident tv_a =
          if Hashtbl.mem c ident
          then unify_tyvars ucx (Hashtbl.find c ident) tv_a
          else Hashtbl.add c ident tv_a
        in
          Hashtbl.iter (Hashtbl.add c) b;
          Hashtbl.iter merge a;
          c
      in

      let unify_dict_with_record_fields
          (dct:dict)
          (fields:Ast.ty_rec)
          : unit =
        let find_ty (query:Ast.ident) : Ast.ty =
          match atab_search fields query with
              None -> fail()
            | Some t -> t
        in

        let check_entry ident tv =
          unify_ty ucx (find_ty ident) tv
        in
          Hashtbl.iter check_entry dct
      in

      let unify_dict_with_obj_fns
          (dct:dict)
          (fns:(Ast.ident,Ast.ty_fn) Hashtbl.t) : unit =
        let check_entry (query:Ast.ident) tv : unit =
          match htab_search fns query with
              None -> fail ()
            | Some fn -> unify_ty ucx (Ast.TY_fn fn) tv
        in
          Hashtbl.iter check_entry dct
      in

      let rec unify_resolved_types
          (ty_a:Ast.ty)
          (ty_b:Ast.ty)
          : Ast.ty =
        match ty_a, ty_b with
            a, b when a = b -> a
          | Ast.TY_box a, b | b, Ast.TY_box a when ucx.box_ok ->
              Ast.TY_box (unify_resolved_types a b)
          | Ast.TY_mutable a, b | b, Ast.TY_mutable a when ucx.mut_ok ->
              Ast.TY_mutable (unify_resolved_types a b)
          | Ast.TY_constrained (a, constrs), b
          | b, Ast.TY_constrained (a, constrs) ->
              Ast.TY_constrained ((unify_resolved_types a b), constrs)
          | _ -> fail()
      in

      let rec is_comparable_or_ordered (comparable:bool) (ty:Ast.ty) : bool =
        match ty with
            Ast.TY_mach _ | Ast.TY_int | Ast.TY_uint
          | Ast.TY_char | Ast.TY_str -> true
          | Ast.TY_any | Ast.TY_nil | Ast.TY_bool | Ast.TY_chan _
          | Ast.TY_port _ | Ast.TY_task | Ast.TY_tup _ | Ast.TY_vec _
          | Ast.TY_rec _ | Ast.TY_tag _ | Ast.TY_iso _ | Ast.TY_idx _ ->
              comparable
          | Ast.TY_fn _ | Ast.TY_obj _
          | Ast.TY_param _ | Ast.TY_native _ | Ast.TY_type -> false
          | Ast.TY_named _ -> bug () "unexpected named type"
          | Ast.TY_constrained (ty, _) ->
              is_comparable_or_ordered comparable ty
          | Ast.TY_mutable ty ->
              ucx.mut_ok && is_comparable_or_ordered comparable ty
          | Ast.TY_box ty ->
              ucx.box_ok && is_comparable_or_ordered comparable ty
      in

      let rec floating (ty:Ast.ty) : bool =
        match ty with
            Ast.TY_mach TY_f32 | Ast.TY_mach TY_f64 -> true
          | Ast.TY_mutable ty when ucx.mut_ok -> floating ty
          | Ast.TY_box ty when ucx.box_ok -> floating ty
          | _ -> false
      in

      let rec integral (ty:Ast.ty) : bool =
        match ty with
            Ast.TY_int | Ast.TY_uint | Ast.TY_mach TY_u8 | Ast.TY_mach TY_u16
          | Ast.TY_mach TY_u32 | Ast.TY_mach TY_u64 | Ast.TY_mach TY_i8
          | Ast.TY_mach TY_i16 | Ast.TY_mach TY_i32
          | Ast.TY_mach TY_i64 ->
              true
          | Ast.TY_mutable ty when ucx.mut_ok -> integral ty
          | Ast.TY_box ty when ucx.box_ok -> integral ty
          | _ -> false
      in

      let numeric (ty:Ast.ty) : bool = (integral ty) || (floating ty) in

      let rec plusable (ty:Ast.ty) : bool =
        match ty with
            Ast.TY_str -> true
          | Ast.TY_vec _ -> true
          | Ast.TY_mutable ty when ucx.mut_ok -> plusable ty
          | Ast.TY_box ty when ucx.box_ok -> plusable ty
          | _ -> numeric ty
      in

      let result =
        match (!a, !b) with
            (TYSPEC_equiv _, _) | (_, TYSPEC_equiv _) ->
              bug () "equiv found even though tyvar was resolved"

          | (TYSPEC_all, other) | (other, TYSPEC_all) -> other

          (* box *)

          | (TYSPEC_box a', TYSPEC_box b') ->
              unify_tyvars ucx a' b'; !a

          | (TYSPEC_box tv,
             TYSPEC_resolved (params, Ast.TY_box ty))
          | (TYSPEC_resolved (params, Ast.TY_box ty),
             TYSPEC_box tv) ->
               unify_ty_parametric ucx ty params tv; !a

          | (_, TYSPEC_resolved (params, Ast.TY_box ty))
              when ucx.box_ok ->
              unify_ty_parametric ucx ty params a; !b

          | (TYSPEC_resolved (params, Ast.TY_box ty), _)
              when ucx.box_ok ->
              unify_ty_parametric ucx ty params b; !a

          | (TYSPEC_box a', _) when ucx.box_ok
              -> unify_tyvars ucx a' b; !a
          | (_, TYSPEC_box b') when ucx.box_ok
              -> unify_tyvars ucx a b'; !b

          | (_, TYSPEC_box _)
          | (TYSPEC_box _, _) -> fail()

          (* mutable *)

          | (TYSPEC_mutable a', TYSPEC_mutable b') ->
              unify_tyvars ucx a' b'; !a

          | (TYSPEC_mutable tv,
             TYSPEC_resolved (params, Ast.TY_mutable ty))
          | (TYSPEC_resolved (params, Ast.TY_mutable ty),
             TYSPEC_mutable tv) ->
               unify_ty_parametric ucx ty params tv; !a

          | (_, TYSPEC_resolved (params, Ast.TY_mutable ty))
              when ucx.mut_ok ->
              unify_ty_parametric ucx ty params a; !b

          | (TYSPEC_resolved (params, Ast.TY_mutable ty), _)
              when ucx.mut_ok ->
              unify_ty_parametric ucx ty params b; !a

          | (TYSPEC_mutable a', _) when ucx.mut_ok
              -> unify_tyvars ucx a' b; !a
          | (_, TYSPEC_mutable b') when ucx.mut_ok
              -> unify_tyvars ucx a b'; !b

          | (_, TYSPEC_mutable _)
          | (TYSPEC_mutable _, _) -> fail()

          (* resolved *)

          | (TYSPEC_resolved (params_a, ty_a),
             TYSPEC_resolved (params_b, ty_b)) ->
              if params_a <> params_b then fail()
              else TYSPEC_resolved
                (params_a, (unify_resolved_types ty_a ty_b))

          | (TYSPEC_resolved (params, ty),
             TYSPEC_callable (out_tv, in_tvs))
          | (TYSPEC_callable (out_tv, in_tvs),
             TYSPEC_resolved (params, ty)) ->
              let unify_in_slot i in_slot =
                unify_slot arg_pass_ctx in_slot None in_tvs.(i)
              in
                let rec unify ty =
                  match ty with
                      Ast.TY_fn ({
                                   Ast.sig_input_slots = in_slots;
                                   Ast.sig_output_slot = out_slot
                                 }, _) ->
                        if Array.length in_slots != Array.length in_tvs
                        then
                          fail ()
                        else
                          unify_slot arg_pass_ctx out_slot None out_tv;
                          Array.iteri unify_in_slot in_slots;
                          ty
                    | Ast.TY_box ty when ucx.box_ok
                        -> Ast.TY_box (unify ty)
                    | Ast.TY_mutable ty when ucx.mut_ok
                        -> Ast.TY_mutable (unify ty)
                    | _ -> fail ()
                in
                TYSPEC_resolved (params, unify ty)

          | (TYSPEC_resolved (params, ty), TYSPEC_collection tv)
          | (TYSPEC_collection tv, TYSPEC_resolved (params, ty)) ->
              let rec unify ty =
                match ty with
                    Ast.TY_vec ty' -> unify_ty ucx ty' tv; ty
                  | Ast.TY_str ->
                      unify_ty ucx (Ast.TY_mach TY_u8) tv; ty
                  | Ast.TY_box ty
                      when ucx.box_ok -> Ast.TY_box (unify ty)
                  | Ast.TY_mutable ty
                      when ucx.mut_ok -> Ast.TY_mutable (unify ty)
                  | _ -> fail ()
              in
              TYSPEC_resolved (params, unify ty)

          | (TYSPEC_resolved (params, ty), TYSPEC_comparable)
          | (TYSPEC_comparable, TYSPEC_resolved (params, ty)) ->
              if not (is_comparable_or_ordered true ty) then fail ()
              else TYSPEC_resolved (params, ty)

          | (TYSPEC_resolved (params, ty), TYSPEC_plusable)
          | (TYSPEC_plusable, TYSPEC_resolved (params, ty)) ->
              if not (plusable ty) then fail ()
              else TYSPEC_resolved (params, ty)

          | (TYSPEC_resolved (params, ty), TYSPEC_dictionary dct)
          | (TYSPEC_dictionary dct, TYSPEC_resolved (params, ty)) ->
              let rec unify ty =
                match ty with
                    Ast.TY_rec fields ->
                      unify_dict_with_record_fields dct fields;
                      ty
                  | Ast.TY_obj (_, fns) ->
                      unify_dict_with_obj_fns dct fns;
                      ty
                  | Ast.TY_box ty
                      when ucx.box_ok -> Ast.TY_box (unify ty)
                  | Ast.TY_mutable ty
                      when ucx.mut_ok -> Ast.TY_mutable (unify ty)
                  | _ -> fail ()
              in
              TYSPEC_resolved (params, unify ty)

          | (TYSPEC_resolved (params, ty), TYSPEC_integral)
          | (TYSPEC_integral, TYSPEC_resolved (params, ty)) ->
              if not (integral ty)
              then fail ()
              else TYSPEC_resolved (params, ty)

          | (TYSPEC_resolved (params, ty), TYSPEC_numeric)
          | (TYSPEC_numeric, TYSPEC_resolved (params, ty)) ->
              if not (numeric ty) then fail ()
              else TYSPEC_resolved (params, ty)

          | (TYSPEC_resolved (params, ty), TYSPEC_ordered)
          | (TYSPEC_ordered, TYSPEC_resolved (params, ty)) ->
              if not (is_comparable_or_ordered false ty) then fail ()
              else TYSPEC_resolved (params, ty)

          | (TYSPEC_resolved (params, ty), TYSPEC_app (tv, args))
          | (TYSPEC_app (tv, args), TYSPEC_resolved (params, ty)) ->
              let ty = rebuild_ty_under_params ty params args false in
                unify_ty ucx ty tv;
                TYSPEC_resolved ([| |], ty)

          | (TYSPEC_resolved (params, ty), TYSPEC_record dct)
          | (TYSPEC_record dct, TYSPEC_resolved (params, ty)) ->
              let rec unify ty =
                match ty with
                    Ast.TY_rec fields ->
                      unify_dict_with_record_fields dct fields;
                      ty
                  | Ast.TY_box ty
                      when ucx.box_ok -> Ast.TY_box (unify ty)
                  | Ast.TY_mutable ty
                      when ucx.mut_ok -> Ast.TY_mutable (unify ty)
                  | _ -> fail ()
              in
              TYSPEC_resolved (params, unify ty)

          | (TYSPEC_resolved (params, ty), TYSPEC_tuple tvs)
          | (TYSPEC_tuple tvs, TYSPEC_resolved (params, ty)) ->
              let rec unify ty =
                match ty with
                    Ast.TY_tup (elem_tys:Ast.ty array) ->
                      if (Array.length elem_tys) < (Array.length tvs)
                      then fail ()
                      else
                        let check_elem i tv =
                          unify_ty ucx (elem_tys.(i)) tv
                        in
                          Array.iteri check_elem tvs;
                          ty
                  | Ast.TY_box ty
                      when ucx.box_ok -> Ast.TY_box (unify ty)
                  | Ast.TY_mutable ty
                      when ucx.box_ok -> Ast.TY_mutable (unify ty)
                  | _ -> fail ()
              in
              TYSPEC_resolved (params, unify ty)

          | (TYSPEC_resolved (params, ty), TYSPEC_vector tv)
          | (TYSPEC_vector tv, TYSPEC_resolved (params, ty)) ->
              let rec unify ty =
                match ty with
                    Ast.TY_vec ty' -> unify_ty ucx ty' tv; ty
                  | Ast.TY_box ty when ucx.box_ok ->
                      Ast.TY_box (unify ty)
                  | Ast.TY_mutable ty when ucx.mut_ok ->
                      Ast.TY_mutable (unify ty)
                  | _ -> fail ()
              in
              TYSPEC_resolved (params, unify ty)

          (* callable *)

          | (TYSPEC_callable (a_out_tv, a_in_tvs),
             TYSPEC_callable (b_out_tv, b_in_tvs)) ->
              unify_tyvars arg_pass_ctx a_out_tv b_out_tv;
              let check_in_tv i a_in_tv =
                unify_tyvars arg_pass_ctx
                  a_in_tv b_in_tvs.(i)
              in
                Array.iteri check_in_tv a_in_tvs;
                TYSPEC_callable (a_out_tv, a_in_tvs)

          | (TYSPEC_callable _, TYSPEC_collection _)
          | (TYSPEC_callable _, TYSPEC_comparable)
          | (TYSPEC_callable _, TYSPEC_plusable)
          | (TYSPEC_callable _, TYSPEC_dictionary _)
          | (TYSPEC_callable _, TYSPEC_integral)
          | (TYSPEC_callable _, TYSPEC_numeric)
          | (TYSPEC_callable _, TYSPEC_ordered)
          | (TYSPEC_callable _, TYSPEC_app _)
          | (TYSPEC_callable _, TYSPEC_record _)
          | (TYSPEC_callable _, TYSPEC_tuple _)
          | (TYSPEC_callable _, TYSPEC_vector _)
          | (TYSPEC_collection _, TYSPEC_callable _)
          | (TYSPEC_comparable, TYSPEC_callable _)
          | (TYSPEC_plusable, TYSPEC_callable _)
          | (TYSPEC_dictionary _, TYSPEC_callable _)
          | (TYSPEC_integral, TYSPEC_callable _)
          | (TYSPEC_numeric, TYSPEC_callable _)
          | (TYSPEC_ordered, TYSPEC_callable _)
          | (TYSPEC_app _, TYSPEC_callable _)
          | (TYSPEC_record _, TYSPEC_callable _)
          | (TYSPEC_tuple _, TYSPEC_callable _)
          | (TYSPEC_vector _, TYSPEC_callable _) -> fail ()

          (* collection *)

          | (TYSPEC_collection av, TYSPEC_collection bv) ->
              unify_tyvars ucx av bv;
              TYSPEC_collection av

          | (TYSPEC_collection av, TYSPEC_comparable)
          | (TYSPEC_comparable, TYSPEC_collection av) ->
              TYSPEC_collection av

          | (TYSPEC_collection v, TYSPEC_plusable)
          | (TYSPEC_plusable, TYSPEC_collection v) -> TYSPEC_collection v

          | (TYSPEC_collection _, TYSPEC_dictionary _)
          | (TYSPEC_collection _, TYSPEC_integral)
          | (TYSPEC_collection _, TYSPEC_numeric)
          | (TYSPEC_collection _, TYSPEC_ordered)
          | (TYSPEC_collection _, TYSPEC_app _)
          | (TYSPEC_collection _, TYSPEC_record _)
          | (TYSPEC_collection _, TYSPEC_tuple _)
          | (TYSPEC_dictionary _, TYSPEC_collection _)
          | (TYSPEC_integral, TYSPEC_collection _)
          | (TYSPEC_numeric, TYSPEC_collection _)
          | (TYSPEC_ordered, TYSPEC_collection _)
          | (TYSPEC_app _, TYSPEC_collection _)
          | (TYSPEC_record _, TYSPEC_collection _)
          | (TYSPEC_tuple _, TYSPEC_collection _) -> fail ()

          | (TYSPEC_collection av, TYSPEC_vector bv)
          | (TYSPEC_vector bv, TYSPEC_collection av) ->
              unify_tyvars ucx av bv;
              TYSPEC_vector av

          (* comparable *)

          | (TYSPEC_comparable, TYSPEC_comparable) -> TYSPEC_comparable

          | (TYSPEC_comparable, TYSPEC_plusable)
          | (TYSPEC_plusable, TYSPEC_comparable) -> TYSPEC_plusable

          | (TYSPEC_comparable, TYSPEC_dictionary dict)
          | (TYSPEC_dictionary dict, TYSPEC_comparable) ->
              TYSPEC_dictionary dict

          | (TYSPEC_comparable, TYSPEC_integral)
          | (TYSPEC_integral, TYSPEC_comparable) -> TYSPEC_integral

          | (TYSPEC_comparable, TYSPEC_numeric)
          | (TYSPEC_numeric, TYSPEC_comparable) -> TYSPEC_numeric

          | (TYSPEC_comparable, TYSPEC_ordered)
          | (TYSPEC_ordered, TYSPEC_comparable) -> TYSPEC_ordered

          | (TYSPEC_comparable, TYSPEC_app _)
          | (TYSPEC_app _, TYSPEC_comparable) -> fail ()

          | (TYSPEC_comparable, TYSPEC_record r)
          | (TYSPEC_record r, TYSPEC_comparable) -> TYSPEC_record r

          | (TYSPEC_comparable, TYSPEC_tuple t)
          | (TYSPEC_tuple t, TYSPEC_comparable) -> TYSPEC_tuple t

          | (TYSPEC_comparable, TYSPEC_vector v)
          | (TYSPEC_vector v, TYSPEC_comparable) -> TYSPEC_vector v

          (* plusable *)

          | (TYSPEC_plusable, TYSPEC_plusable) -> TYSPEC_plusable

          | (TYSPEC_plusable, TYSPEC_dictionary _)
          | (TYSPEC_dictionary _, TYSPEC_plusable) -> fail ()

          | (TYSPEC_plusable, TYSPEC_integral)
          | (TYSPEC_integral, TYSPEC_plusable) -> TYSPEC_integral

          | (TYSPEC_plusable, TYSPEC_numeric)
          | (TYSPEC_numeric, TYSPEC_plusable) -> TYSPEC_numeric

          | (TYSPEC_plusable, TYSPEC_ordered)
          | (TYSPEC_ordered, TYSPEC_plusable) -> TYSPEC_plusable

          | (TYSPEC_plusable, TYSPEC_record _)
          | (TYSPEC_record _, TYSPEC_plusable) -> fail ()

          | (TYSPEC_plusable, TYSPEC_tuple _)
          | (TYSPEC_tuple _, TYSPEC_plusable) -> fail ()

          | (TYSPEC_plusable, TYSPEC_vector v)
          | (TYSPEC_vector v, TYSPEC_plusable) -> TYSPEC_vector v

          | (TYSPEC_plusable, TYSPEC_app _)
          | (TYSPEC_app _, TYSPEC_plusable) -> fail ()

          (* dictionary *)

          | (TYSPEC_dictionary da, TYSPEC_dictionary db) ->
              TYSPEC_dictionary (merge_dicts da db)

          | (TYSPEC_dictionary _, TYSPEC_integral)
          | (TYSPEC_dictionary _, TYSPEC_numeric)
          | (TYSPEC_dictionary _, TYSPEC_ordered)
          | (TYSPEC_dictionary _, TYSPEC_app _)
          | (TYSPEC_integral, TYSPEC_dictionary _)
          | (TYSPEC_numeric, TYSPEC_dictionary _)
          | (TYSPEC_ordered, TYSPEC_dictionary _)
          | (TYSPEC_app _, TYSPEC_dictionary _) -> fail ()

          | (TYSPEC_dictionary d, TYSPEC_record r)
          | (TYSPEC_record r, TYSPEC_dictionary d) ->
              TYSPEC_record (merge_dicts d r)

          | (TYSPEC_dictionary _, TYSPEC_tuple _)
          | (TYSPEC_dictionary _, TYSPEC_vector _)
          | (TYSPEC_tuple _, TYSPEC_dictionary _)
          | (TYSPEC_vector _, TYSPEC_dictionary _) -> fail ()

          (* integral *)

          | (TYSPEC_integral, TYSPEC_integral)
          | (TYSPEC_integral, TYSPEC_numeric)
          | (TYSPEC_integral, TYSPEC_ordered)
          | (TYSPEC_numeric, TYSPEC_integral)
          | (TYSPEC_ordered, TYSPEC_integral) -> TYSPEC_integral

          | (TYSPEC_integral, TYSPEC_app _)
          | (TYSPEC_integral, TYSPEC_record _)
          | (TYSPEC_integral, TYSPEC_tuple _)
          | (TYSPEC_integral, TYSPEC_vector _)
          | (TYSPEC_app _, TYSPEC_integral)
          | (TYSPEC_record _, TYSPEC_integral)
          | (TYSPEC_tuple _, TYSPEC_integral)
          | (TYSPEC_vector _, TYSPEC_integral) -> fail ()

          (* numeric *)

          | (TYSPEC_numeric, TYSPEC_numeric) -> TYSPEC_numeric

          | (TYSPEC_numeric, TYSPEC_ordered)
          | (TYSPEC_ordered, TYSPEC_numeric) -> TYSPEC_ordered

          | (TYSPEC_numeric, TYSPEC_app _)
          | (TYSPEC_numeric, TYSPEC_record _)
          | (TYSPEC_numeric, TYSPEC_tuple _)
          | (TYSPEC_numeric, TYSPEC_vector _)
          | (TYSPEC_app _, TYSPEC_numeric)
          | (TYSPEC_record _, TYSPEC_numeric)
          | (TYSPEC_tuple _, TYSPEC_numeric)
          | (TYSPEC_vector _, TYSPEC_numeric) -> fail ()

          (* ordered *)

          | (TYSPEC_ordered, TYSPEC_ordered) -> TYSPEC_ordered

          | (TYSPEC_ordered, TYSPEC_app _)
          | (TYSPEC_ordered, TYSPEC_record _)
          | (TYSPEC_ordered, TYSPEC_tuple _)
          | (TYSPEC_ordered, TYSPEC_vector _)
          | (TYSPEC_app _, TYSPEC_ordered)
          | (TYSPEC_record _, TYSPEC_ordered)
          | (TYSPEC_tuple _, TYSPEC_ordered)
          | (TYSPEC_vector _, TYSPEC_ordered) -> fail ()

          (* app *)

          | (TYSPEC_app (tv_a, args_a),
             TYSPEC_app (tv_b, args_b)) ->
              if args_a <> args_b
              then fail()
              else
                begin
                  unify_tyvars ucx tv_a tv_b;
                  TYSPEC_app (tv_a, args_a)
                end

          | (TYSPEC_app _, TYSPEC_record _)
          | (TYSPEC_app _, TYSPEC_tuple _)
          | (TYSPEC_app _, TYSPEC_vector _)
          | (TYSPEC_record _, TYSPEC_app _)
          | (TYSPEC_tuple _, TYSPEC_app _)
          | (TYSPEC_vector _, TYSPEC_app _) -> fail ()

          (* record *)

          | (TYSPEC_record da, TYSPEC_record db) ->
              TYSPEC_record (merge_dicts da db)

          | (TYSPEC_record _, TYSPEC_tuple _)
          | (TYSPEC_record _, TYSPEC_vector _)
          | (TYSPEC_tuple _, TYSPEC_record _)
          | (TYSPEC_vector _, TYSPEC_record _) -> fail ()

          (* tuple *)

          | (TYSPEC_tuple tvs_a, TYSPEC_tuple tvs_b) ->
              let len_a = Array.length tvs_a in
              let len_b = Array.length tvs_b in
              let max_len = max len_a len_b in
              let init_tuple_elem i =
                if i >= len_a
                then tvs_b.(i)
                else if i >= len_b
                then tvs_a.(i)
                else begin
                  unify_tyvars strict_ctx tvs_a.(i) tvs_b.(i);
                  tvs_a.(i)
                end
              in
                TYSPEC_tuple (Array.init max_len init_tuple_elem)

          | (TYSPEC_tuple _, TYSPEC_vector _)
          | (TYSPEC_vector _, TYSPEC_tuple _) -> fail ()

          (* vector *)

          | (TYSPEC_vector av, TYSPEC_vector bv) ->
              unify_tyvars strict_ctx av bv;
              TYSPEC_vector av
      in
      let c = ref result in
        a := TYSPEC_equiv c;
        b := TYSPEC_equiv c

    and unify_ty_parametric
        (ucx:unify_ctxt)
        (ty:Ast.ty)
        (tps:Ast.ty_param array)
        (tv:tyvar)
        : unit =
      unify_tyvars ucx (ref (TYSPEC_resolved (tps, ty))) tv

    and unify_ty (ucx:unify_ctxt) (ty:Ast.ty) (tv:tyvar) : unit =
      unify_ty_parametric ucx ty [||] tv

    in

    let rec unify_lit (ucx:unify_ctxt) (lit:Ast.lit) (tv:tyvar) : unit =
      let ty =
        match lit with
            Ast.LIT_nil -> Ast.TY_nil
          | Ast.LIT_bool _ -> Ast.TY_bool
          | Ast.LIT_mach (mty, _, _) -> Ast.TY_mach mty
          | Ast.LIT_int (_, _) -> Ast.TY_int
          | Ast.LIT_uint (_, _) -> Ast.TY_uint
          | Ast.LIT_char _ -> Ast.TY_char
      in
        unify_ty ucx ty tv

    and unify_atom (ucx:unify_ctxt) (atom:Ast.atom) (tv:tyvar) : unit =
      match atom with
          Ast.ATOM_literal { node = literal; id = _ } ->
            unify_lit ucx literal tv
        | Ast.ATOM_lval lval ->
            unify_lval ucx lval tv

    and unify_expr (ucx:unify_ctxt) (expr:Ast.expr) (tv:tyvar) : unit =
      match expr with
          Ast.EXPR_binary (binop, lhs, rhs) ->
            let binop_sig = match binop with
                Ast.BINOP_eq
              | Ast.BINOP_ne -> BINOPSIG_comp_comp_bool

              | Ast.BINOP_lt
              | Ast.BINOP_le
              | Ast.BINOP_ge
              | Ast.BINOP_gt -> BINOPSIG_ord_ord_bool

              | Ast.BINOP_or
              | Ast.BINOP_and
              | Ast.BINOP_xor
              | Ast.BINOP_lsl
              | Ast.BINOP_lsr
              | Ast.BINOP_asr -> BINOPSIG_integ_integ_integ

              | Ast.BINOP_add -> BINOPSIG_plus_plus_plus

              | Ast.BINOP_sub
              | Ast.BINOP_mul
              | Ast.BINOP_div
              | Ast.BINOP_mod -> BINOPSIG_num_num_num

              | Ast.BINOP_send -> bug () "BINOP_send found in expr"
            in
              begin
                match binop_sig with
                    BINOPSIG_bool_bool_bool ->
                      unify_atom rval_ctx lhs
                        (ref (TYSPEC_resolved ([||], Ast.TY_bool)));
                      unify_atom rval_ctx rhs
                        (ref (TYSPEC_resolved ([||], Ast.TY_bool)));
                      unify_ty rval_ctx Ast.TY_bool tv
                  | BINOPSIG_comp_comp_bool ->
                      let tv_a = ref TYSPEC_comparable in
                        unify_atom rval_ctx lhs tv_a;
                        unify_atom rval_ctx rhs tv_a;
                        unify_ty rval_ctx Ast.TY_bool tv
                  | BINOPSIG_ord_ord_bool ->
                      let tv_a = ref TYSPEC_ordered in
                        unify_atom rval_ctx lhs tv_a;
                        unify_atom rval_ctx rhs tv_a;
                        unify_ty rval_ctx Ast.TY_bool tv
                  | BINOPSIG_integ_integ_integ ->
                      let tv_a = ref TYSPEC_integral in
                        unify_atom rval_ctx lhs tv_a;
                        unify_atom rval_ctx rhs tv_a;
                        unify_tyvars rval_ctx tv tv_a
                  | BINOPSIG_num_num_num ->
                      let tv_a = ref TYSPEC_numeric in
                        unify_atom rval_ctx lhs tv_a;
                        unify_atom rval_ctx rhs tv_a;
                        unify_tyvars rval_ctx tv tv_a
                  | BINOPSIG_plus_plus_plus ->
                      let tv_a = ref TYSPEC_plusable in
                        unify_atom rval_ctx lhs tv_a;
                        unify_atom rval_ctx rhs tv_a;
                        unify_tyvars rval_ctx tv tv_a
              end
        | Ast.EXPR_unary (unop, atom) ->
            begin
              match unop with
                  Ast.UNOP_not ->
                    unify_atom rval_ctx atom
                      (ref (TYSPEC_resolved ([||], Ast.TY_bool)));
                    unify_ty rval_ctx Ast.TY_bool tv
                | Ast.UNOP_bitnot ->
                    let tv_a = ref TYSPEC_integral in
                      unify_atom rval_ctx atom tv_a;
                      unify_tyvars rval_ctx tv tv_a
                | Ast.UNOP_neg ->
                    let tv_a = ref TYSPEC_numeric in
                      unify_atom rval_ctx atom tv_a;
                      unify_tyvars rval_ctx tv tv_a
                | Ast.UNOP_cast t ->
                    (* FIXME (issue #84): check cast-validity in
                     * post-typecheck pass.  Only some casts make sense.
                     *)
                    let tv_a = ref TYSPEC_all in
                    let t = Hashtbl.find cx.ctxt_all_cast_types t.id in
                      unify_atom rval_ctx atom tv_a;
                      unify_ty rval_ctx t tv
            end
        | Ast.EXPR_atom atom -> unify_atom ucx atom tv

    and unify_lval' (ucx:unify_ctxt) (lval:Ast.lval) (tv:tyvar) : unit =
      let note_args args =
        iflog cx (fun _ -> log cx "noting lval '%a' type arguments: %a"
                    Ast.sprintf_lval lval Ast.sprintf_app_args args);
        Hashtbl.add
          cx.ctxt_call_lval_params
          (lval_base_id lval)
          args;
      in
        match lval with
            Ast.LVAL_base nbi ->
              let referent = lval_to_referent cx nbi.id in
                begin
                  match Hashtbl.find cx.ctxt_all_defns referent with
                      DEFN_slot slot ->
                        iflog cx
                          begin
                            fun _ ->
                              let tv = Hashtbl.find bindings referent in
                                log cx "lval-base slot tyspec for %a = %s"
                                  Ast.sprintf_lval lval (tyspec_to_str (!tv));
                          end;
                        begin
                          match htab_search
                            cx.ctxt_auto_deref_lval nbi.id
                          with
                              None ->
                                htab_put cx.ctxt_auto_deref_lval
                                  nbi.id ucx.box_ok
                            | Some b ->
                                (* A given source-occurrence of a name-base
                                 * should never change its auto-deref
                                 * nature.
                                 *)
                                assert (b = ucx.box_ok);
                        end;
                        unify_slot ucx slot (Some referent) tv

                    | _ ->
                        let spec = (!(Hashtbl.find bindings referent)) in
                        let _ =
                          iflog cx
                            begin
                              fun _ ->
                                log cx "lval-base item tyspec for %a = %s"
                                  Ast.sprintf_lval lval (tyspec_to_str spec);
                                log cx "unifying with supplied spec %s"
                                  (tyspec_to_str !tv)
                            end
                        in
                        let tv =
                          match nbi.node with
                              Ast.BASE_ident _ -> tv
                            | Ast.BASE_app (_, args) ->
                                note_args args;
                                ref (TYSPEC_app (tv, args))
                            | _ -> err None "bad lval / tyspec combination"
                        in
                          unify_tyvars ucx (ref spec) tv
                end
          | Ast.LVAL_ext (base, comp) ->
              let base_ts = match comp with
                  Ast.COMP_named (Ast.COMP_ident id) ->
                    let names = Hashtbl.create 1 in
                      Hashtbl.add names id tv;
                      TYSPEC_dictionary names

                | Ast.COMP_named (Ast.COMP_app (id, args)) ->
                    note_args args;
                    let tv = ref (TYSPEC_app (tv, args)) in
                    let names = Hashtbl.create 1 in
                      Hashtbl.add names id tv;
                      TYSPEC_dictionary names

                | Ast.COMP_named (Ast.COMP_idx i) ->
                    let init j = if i + 1 == j then tv else ref TYSPEC_all in
                      TYSPEC_tuple (Array.init (i + 1) init)

                | Ast.COMP_atom atom ->
                    unify_atom rval_ctx atom
                      (ref (TYSPEC_resolved ([||], Ast.TY_int)));
                    TYSPEC_collection tv

                | Ast.COMP_deref ->
                    TYSPEC_box tv
              in
              let base_tv = ref base_ts in
                unify_lval' { ucx with box_ok = true } base base_tv;
                match !(resolve_tyvar base_tv) with
                    TYSPEC_resolved (_, ty) ->
                      unify_ty ucx (project_type ty comp) tv
                  | _ ->
                      ()

    and unify_lval (ucx:unify_ctxt) (lval:Ast.lval) (tv:tyvar) : unit =
      let id = lval_base_id lval in
        (* Fetch lval with type components resolved. *)
        let lval = Hashtbl.find cx.ctxt_all_lvals id in
        iflog cx (fun _ -> log cx
                    "fetched resolved version of lval #%d = %a"
                    (int_of_node id) Ast.sprintf_lval lval);
          Hashtbl.add lval_tyvars id tv;
          unify_lval' ucx lval tv

    in
    let gen_atom_tvs atoms =
      let gen_atom_tv atom =
        let tv = ref TYSPEC_all in
          unify_atom strict_ctx atom tv;
          tv
      in
        Array.map gen_atom_tv atoms
    in
    let visit_stmt_pre_full (stmt:Ast.stmt) : unit =

      let check_callable out_tv callee args =
        let in_tvs = gen_atom_tvs args in
        let callee_tv = ref (TYSPEC_callable (out_tv, in_tvs)) in
          unify_lval rval_ctx callee callee_tv;
      in

      let set_auto_deref lv b =
        Hashtbl.replace cx.ctxt_auto_deref_lval (lval_base_id lv) b;
      in

      let ty t = ref (TYSPEC_resolved ([||], t)) in
      let any _ = ref TYSPEC_all in

      match stmt.node with
          Ast.STMT_spawn (dst, _, callee, args) ->
            let out_tv = ty Ast.TY_nil in
              unify_lval lval_ctx dst (ty Ast.TY_task);
              check_callable out_tv callee args

        | Ast.STMT_init_rec (dst, fields, Some base) ->
            let dct = Hashtbl.create 10 in
            let tvrec = ref (TYSPEC_record dct) in
            let add_field (ident, atom) =
              let tv = any() in
                unify_atom arg_pass_ctx atom tv;
                Hashtbl.add dct ident tv
            in
              Array.iter add_field fields;
              let tvbase = any() in
                unify_lval rval_ctx base tvbase;
                unify_tyvars rval_ctx tvrec tvbase;
                unify_lval init_ctx dst tvrec

        | Ast.STMT_init_rec (dst, fields, None) ->
            let dct = Hashtbl.create 10 in
            let add_field (ident, atom) =
              let tv = any() in
                unify_atom arg_pass_ctx atom tv;
                Hashtbl.add dct ident tv
            in
              Array.iter add_field fields;
              unify_lval init_ctx dst (ref (TYSPEC_record dct))

        | Ast.STMT_init_tup (dst, members) ->
            let member_to_tv atom =
              let tv = any() in
                unify_atom arg_pass_ctx atom tv;
                tv
            in
            let member_tvs = Array.map member_to_tv members in
              unify_lval init_ctx dst (ref (TYSPEC_tuple member_tvs))

        | Ast.STMT_init_vec (dst, atoms) ->
            let tv = any() in
            let unify_with_tv atom = unify_atom arg_pass_ctx atom tv in
              Array.iter unify_with_tv atoms;
              unify_lval init_ctx dst (ref (TYSPEC_vector tv))

        | Ast.STMT_init_str (dst, _) ->
            unify_lval init_ctx dst (ty Ast.TY_str)

        | Ast.STMT_copy (dst, expr) ->
            let tv = any() in
              unify_expr arg_pass_ctx expr tv;
              unify_lval lval_ctx dst tv

        | Ast.STMT_copy_binop (dst, binop, at) ->
            let tv = any() in
              unify_expr arg_pass_ctx
                (Ast.EXPR_binary (binop, Ast.ATOM_lval dst, at)) tv;
              (* Force-override the 'auto-deref' judgment that was cached 
               * in cx.ctxt_auto_deref_lval by preceding unify_expr call.
               *)
              set_auto_deref dst false;
              unify_lval lval_ctx dst tv;

        | Ast.STMT_call (out, callee, args) ->
            let out_tv = any() in
              unify_lval arg_pass_ctx out out_tv;
              check_callable out_tv callee args

        | Ast.STMT_log atom ->
            begin
              match atom with
                  Ast.ATOM_lval lv -> set_auto_deref lv true
                | _ -> ()
            end

        | Ast.STMT_check_expr expr ->
            unify_expr rval_ctx expr (ty Ast.TY_bool)

        | Ast.STMT_check (_, check_calls) ->
            let out_tv = ty Ast.TY_bool in
              Array.iter
                (fun (callee, args) ->
                   check_callable out_tv callee args)
                check_calls

        | Ast.STMT_while { Ast.while_lval = (_, expr) }
        | Ast.STMT_do_while { Ast.while_lval = (_, expr) } ->
            unify_expr rval_ctx expr (ty Ast.TY_bool)

        | Ast.STMT_if { Ast.if_test = if_test } ->
            unify_expr rval_ctx if_test (ty Ast.TY_bool);

        | Ast.STMT_ret atom_opt
        | Ast.STMT_put atom_opt ->
            begin
              match atom_opt with
                  None -> unify_ty arg_pass_ctx Ast.TY_nil (retval_tv())
                | Some atom -> unify_atom arg_pass_ctx atom (retval_tv())
            end

        | Ast.STMT_be (callee, args) ->
            check_callable (retval_tv()) callee args

        | Ast.STMT_bind (bound, callee, arg_opts) ->
            (* FIXME (issue #81): handle binding type parameters
             * eventually.
             *)
            let out_tv = any() in
            let residue = ref [] in
            let gen_atom_opt_tvs atoms =
              let gen_atom_tv atom_opt =
                let tv = any() in
                  begin
                    match atom_opt with
                        None -> residue := tv :: (!residue);
                      | Some atom -> unify_atom arg_pass_ctx atom tv
                  end;
                  tv
              in
                Array.map gen_atom_tv atoms
            in

            let in_tvs = gen_atom_opt_tvs arg_opts in
            let arg_residue_tvs = Array.of_list (List.rev (!residue)) in
            let callee_tv = ref (TYSPEC_callable (out_tv, in_tvs)) in
            let bound_tv = ref (TYSPEC_callable (out_tv, arg_residue_tvs)) in
              unify_lval rval_ctx callee callee_tv;
              unify_lval lval_ctx bound bound_tv

        | Ast.STMT_for_each fe ->
            let out_tv = any() in
            let (si, _) = fe.Ast.for_each_slot in
            let (callee, args) = fe.Ast.for_each_call in
              unify_slot lval_ctx si.node (Some si.id) out_tv;
              check_callable out_tv callee args

        | Ast.STMT_for fo ->
            let mem_tv = ref TYSPEC_all in
            let seq_tv = ref (TYSPEC_collection mem_tv) in
            let (si, _) = fo.Ast.for_slot in
            let (_, seq) = fo.Ast.for_seq in
              unify_lval rval_ctx seq seq_tv;
              unify_slot lval_ctx si.node (Some si.id) mem_tv

        | Ast.STMT_alt_tag
            { Ast.alt_tag_lval = lval; Ast.alt_tag_arms = arms } ->
            let lval_tv = any() in
              unify_lval arg_pass_ctx lval lval_tv;
              Array.iter (fun _ -> push_pat_tv lval_tv) arms

        | Ast.STMT_join lval ->
            unify_lval rval_ctx lval (ty Ast.TY_task);

        | Ast.STMT_init_box (dst, v) ->
            let tv = ref (TYSPEC_mutable (ref (TYSPEC_box (any())))) in
              unify_lval strict_ctx dst tv;
              unify_atom rval_ctx v tv;

        (* FIXME (issue #52): Finish these. *)
        (* Fake-typecheck a few comm-related statements for now, just enough
         * to supply the auto-deref contexts; we will need new tyspecs for
         * port and channel constraints.
         *)

        | Ast.STMT_recv (dst, port) ->
            set_auto_deref dst rval_ctx.box_ok;
            set_auto_deref port rval_ctx.box_ok;

        | Ast.STMT_send (chan, v) ->
            set_auto_deref chan rval_ctx.box_ok;
            set_auto_deref v rval_ctx.box_ok;

        | Ast.STMT_init_chan (dst, port_opt) ->
            begin
              match port_opt with
                  None -> ()
                | Some port -> set_auto_deref port rval_ctx.box_ok
            end;
            set_auto_deref dst init_ctx.box_ok

        | Ast.STMT_init_port dst ->
            set_auto_deref dst init_ctx.box_ok


        (* Nothing to typecheck on these. *)
        | Ast.STMT_block _
        | Ast.STMT_decl _
        | Ast.STMT_yield
        | Ast.STMT_fail -> ()

        (* Unimplemented. *)
        | Ast.STMT_check_if _
        | Ast.STMT_prove _
        | Ast.STMT_note _
        | Ast.STMT_alt_port _
        | Ast.STMT_alt_type _
        | Ast.STMT_put_each _
        | Ast.STMT_slice _ -> err None "Unimplemented typecheck for stmt"
    in

    let visit_stmt_pre (stmt:Ast.stmt) : unit =
      try
        log cx "";
        log cx "typechecking stmt: %a" Ast.sprintf_stmt stmt;
        log cx "";
        visit_stmt_pre_full stmt;
        (* 
         * Reset any item-parameters that were resolved to types
         * during inference for this statement.
         *)
        Hashtbl.iter
          (fun _ params -> Array.iter (fun tv -> tv := TYSPEC_all) params)
          item_params;
        log cx "finished typechecking stmt: %a" Ast.sprintf_stmt stmt;
      with
          Semant_err (None, msg) ->
            raise (Semant_err ((Some stmt.id), msg))
    in

    let enter_fn fn retspec =
      let out = fn.Ast.fn_output_slot in
        push_retval_tv (ref retspec);
        unify_slot arg_pass_ctx out.node (Some out.id) (retval_tv())
    in

    let visit_obj_fn_pre obj ident fn =
      enter_fn fn.node TYSPEC_all;
      inner.Walk.visit_obj_fn_pre obj ident fn
    in

    let visit_obj_fn_post obj ident fn =
      inner.Walk.visit_obj_fn_post obj ident fn;
      pop_retval_tv ();
    in

    let visit_mod_item_pre n p mod_item =
      begin
        try
          match mod_item.node.Ast.decl_item with
              Ast.MOD_ITEM_fn fn ->
                enter_fn fn TYSPEC_all

            | _ -> ()
        with Semant_err (None, msg) ->
          raise (Semant_err ((Some mod_item.id), msg))
      end;
      inner.Walk.visit_mod_item_pre n p mod_item
    in

    let path_name (_:unit) : string =
      string_of_name (Walk.path_to_name path)
    in

    let visit_mod_item_post n p mod_item =
      inner.Walk.visit_mod_item_post n p mod_item;
      match mod_item.node.Ast.decl_item with

        | Ast.MOD_ITEM_fn _ ->
            pop_retval_tv ();
            if (Some (path_name())) = cx.ctxt_main_name
            then
              begin
                match Hashtbl.find cx.ctxt_all_item_types mod_item.id with
                    Ast.TY_fn (tsig, _) ->
                      begin
                        let vec_str =
                          local_slot (Ast.TY_vec Ast.TY_str)
                        in
                          match tsig.Ast.sig_input_slots with
                              [| |] -> ()
                            | [| vs |] when vs = vec_str -> ()
                            | _ -> err (Some mod_item.id)
                                "main fn has bad type signature"
                      end
                  | _ ->
                      err (Some mod_item.id) "main item is not a function"
              end
        | _ -> ()
    in

    (*
     * Tag patterns give us the type of every sub-pattern in the tag tuple, so
     * we can "expect" those types by pushing them on a stack.  Checking a
     * pattern therefore involves seeing that it matches the "expected" type,
     * and in turn setting any expectations for the inner descent.
     *)
    let visit_pat_pre (pat:Ast.pat) : unit =
      let expected = pat_tv() in
        match pat with
            Ast.PAT_lit lit -> unify_lit strict_ctx lit expected

          | Ast.PAT_tag (lval, _) ->
              let expect ty =
                let tv = ref TYSPEC_all in
                  unify_ty strict_ctx ty tv;
                  push_pat_tv tv;
              in

              let lval_nm = lval_to_name lval in

              (* The lval here is our tag constructor, which we've already
               * resolved (in Resolve) to have a the actual tag constructor
               * function item as its referent.  It should hence unify
               * exactly to that function type, rebuilt under any latent type
               * parameters applied in the lval. *)
              let lval_tv = ref TYSPEC_all in
                unify_lval strict_ctx lval lval_tv;
                let tag_ctor_ty =
                  match !(resolve_tyvar lval_tv) with
                      TYSPEC_resolved (_, ty) -> ty
                    | _ ->
                        bug () "tag constructor is not a fully resolved type."
                in

                let tag_ty = fn_output_ty tag_ctor_ty in
                let tag_ty_tup = tag_or_iso_ty_tup_by_name tag_ty lval_nm in

                let tag_tv = ref TYSPEC_all in
                  unify_ty strict_ctx tag_ty tag_tv;
                  unify_tyvars strict_ctx expected tag_tv;
                  List.iter expect
                    (List.rev (Array.to_list tag_ty_tup));

          | Ast.PAT_slot (sloti, _) ->
              unify_slot lval_ctx sloti.node (Some sloti.id) expected

          | Ast.PAT_wild -> ()
    in

    let visit_pat_post (_:Ast.pat) : unit =
      pop_pat_tv()
    in

      {
        inner with
          Walk.visit_mod_item_pre = visit_mod_item_pre;
          Walk.visit_mod_item_post = visit_mod_item_post;
          Walk.visit_obj_fn_pre = visit_obj_fn_pre;
          Walk.visit_obj_fn_post = visit_obj_fn_post;
          Walk.visit_stmt_pre = visit_stmt_pre;
          Walk.visit_pat_pre = visit_pat_pre;
          Walk.visit_pat_post = visit_pat_post;
      }

  in
    try
      let auto_queue = Queue.create () in

      let init_slot_tyvar id defn =
        match defn with
            DEFN_slot { Ast.slot_mode = _; Ast.slot_ty = None } ->
              Queue.add id auto_queue;
              Hashtbl.add bindings id (ref (TYSPEC_mutable (ref TYSPEC_all)))
          | DEFN_slot { Ast.slot_mode = _; Ast.slot_ty = Some ty } ->
              let _ = iflog cx (fun _ -> log cx "initial slot #%d type: %a"
                                  (int_of_node id) Ast.sprintf_ty ty)
              in
                Hashtbl.add bindings id (ref (TYSPEC_resolved ([||], ty)))
          | _ -> ()
      in

      let init_item_tyvar id ty =
        let _ = iflog cx (fun _ -> log cx "initial item #%d type: %a"
                            (int_of_node id) Ast.sprintf_ty ty)
        in
        let params =
          match Hashtbl.find cx.ctxt_all_defns id with
              DEFN_item i -> Array.map (fun p -> p.node) i.Ast.decl_params
            | DEFN_obj_fn _ -> [| |]
            | DEFN_obj_drop _ -> [| |]
            | DEFN_loop_body _ -> [| |]
            | _ -> err (Some id) "expected item defn for item tyvar"
        in
        let spec = TYSPEC_resolved (params, ty) in
          Hashtbl.add bindings id (ref spec)
      in

      let init_mod_dict id defn =
        let rec tv_of_item id item =
          match item.Ast.decl_item with
              Ast.MOD_ITEM_mod (_, items) ->
                if Hashtbl.mem bindings id
                then Hashtbl.find bindings id
                else
                  let dict = htab_map items
                    (fun i item -> (i, tv_of_item item.id item.node))
                  in
                  let spec = TYSPEC_dictionary dict in
                  let tv = ref spec in
                    Hashtbl.add bindings id tv;
                    tv
            | _ ->
                Hashtbl.find bindings id
        in
          match defn with
              DEFN_item ({ Ast.decl_item = Ast.MOD_ITEM_mod _ } as item) ->
                ignore (tv_of_item id item)
            | _ -> ()
      in
        Hashtbl.iter init_slot_tyvar cx.ctxt_all_defns;
        Hashtbl.iter init_item_tyvar cx.ctxt_all_item_types;
        Hashtbl.iter init_mod_dict cx.ctxt_all_defns;
        Walk.walk_crate
          (Walk.path_managing_visitor path
             (Walk.mod_item_logging_visitor
                (log cx "typechecking pass: %s")
                path
                (visitor cx Walk.empty_visitor)))
          crate;

        let update_auto_tyvar id ty =
          let defn = Hashtbl.find cx.ctxt_all_defns id in
            match defn with
                DEFN_slot slot_defn ->
                  begin
                    match slot_defn.Ast.slot_ty with
                        Some _ -> ()
                      | None ->
                          log cx "setting auto slot #%d = %a to type %a"
                            (int_of_node id)
                            Ast.sprintf_slot_key
                            (Hashtbl.find cx.ctxt_slot_keys id)
                            Ast.sprintf_ty ty;
                          Hashtbl.replace cx.ctxt_all_defns id
                            (DEFN_slot { slot_defn with
                                           Ast.slot_ty = Some ty })
                  end
              | _ -> bug () "check_auto_tyvar: no slot defn"
        in

        let rec get_resolved_ty tv id =
          let ts = !(resolve_tyvar tv) in
            match ts with
                TYSPEC_resolved ([||], ty) -> ty
              | TYSPEC_box tv ->
                  Ast.TY_box (get_resolved_ty tv id)

              | TYSPEC_mutable tv ->
                  Ast.TY_mutable (get_resolved_ty tv id)

              | TYSPEC_vector tv ->
                  Ast.TY_vec (get_resolved_ty tv id)

              | TYSPEC_tuple tvs ->
                  Ast.TY_tup
                    (Array.map
                       (fun tv -> get_resolved_ty tv id) tvs)

              | _ -> err (Some id)
                  "unresolved type %s (%d)"
                    (tyspec_to_str ts)
                    (int_of_node id)
        in

        let check_auto_tyvar id =
          let tv = Hashtbl.find bindings id in
          let ty = get_resolved_ty tv id in
            update_auto_tyvar id ty
        in

        let record_lval_ty id tv =
          let ty = get_resolved_ty tv id in
          let _ =
            iflog cx
              (fun _ ->
                 log cx "recording resolved lval #%d type %a"
                   (int_of_node id)
                   Ast.sprintf_ty ty)
          in
            Hashtbl.add cx.ctxt_all_lval_types id ty
        in

          Queue.iter check_auto_tyvar auto_queue;
          Hashtbl.iter record_lval_ty lval_tyvars;
    with Semant_err (ido, str) -> report_err cx ido str
;;

(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)

