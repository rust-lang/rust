open Common;;
open Semant;;

type tyspec =
    TYSPEC_equiv of (simpl * tyvar)
  | TYSPEC_all
  | TYSPEC_resolved of (Ast.ty_param array) * Ast.ty
  | TYSPEC_callable of (tyvar * tyvar array)   (* out, ins *)
  | TYSPEC_collection of tyvar                 (* vec or str *)
  | TYSPEC_comparable                          (* comparable with = and != *)
  | TYSPEC_plusable                            (* nums, vecs, and strings *)
  | TYSPEC_dictionary of dict
  | TYSPEC_integral                            (* int-like *)
  | TYSPEC_loggable
  | TYSPEC_numeric                             (* int-like or float-like *)
  | TYSPEC_ordered                             (* comparable with < etc. *)
  | TYSPEC_record of dict
  | TYSPEC_tuple of tyvar array                (* heterogeneous tuple *)
  | TYSPEC_vector of tyvar
  | TYSPEC_app of (tyvar * Ast.ty array)

and simpl = SIMPL_none
            | SIMPL_exterior
            | SIMPL_mutable

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
      | TYSPEC_loggable -> fmt ff "<loggable>"
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

      | TYSPEC_equiv (SIMPL_none, tv) ->
          fmt_tyspec ff (!tv)

      | TYSPEC_equiv (SIMPL_exterior, tv) ->
          fmt ff "@";
          fmt_tyspec ff (!tv)

      | TYSPEC_equiv (SIMPL_mutable, tv) ->
          fmt ff "mutable ";
          fmt_tyspec ff (!tv)

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
          fmt ff "(";
          fmt_tvs ff tvs;
          fmt ff ")";

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
      TYSPEC_equiv (_, subtv) -> resolve_tyvar subtv
    | _ -> tv
;;

let process_crate (cx:ctxt) (crate:Ast.crate) : unit =
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
        (simplify:bool)
        (slot:Ast.slot)
        (id_opt:node_id option)
        (tv:tyvar) : unit =
      match id_opt with
          Some id -> unify_tyvars simplify (Hashtbl.find bindings id) tv
        | None ->
            match slot.Ast.slot_ty with
                None -> bug () "untyped unidentified slot"
              | Some ty -> unify_ty simplify ty tv

    and check_sane_tyvar tv =
      match !tv with
          TYSPEC_resolved (_, (Ast.TY_named _)) ->
            bug () "named-type in type checker"
        | _ -> ()

    and unify_tyvars (simplify:bool) (av:tyvar) (bv:tyvar) : unit =
      let sstr = if simplify then "w/ simplification" else "" in
        iflog cx (fun _ ->
                    log cx "unifying types%s:" sstr;
                    log cx "input tyvar A: %s" (tyspec_to_str !av);
                    log cx "input tyvar B: %s" (tyspec_to_str !bv));
        check_sane_tyvar av;
        check_sane_tyvar bv;

        unify_tyvars' simplify av bv;

        iflog cx (fun _ ->
                    log cx "unified types%s:" sstr;
                    log cx "output tyvar A: %s" (tyspec_to_str !av);
                    log cx "output tyvar B: %s" (tyspec_to_str !bv));
        check_sane_tyvar av;
        check_sane_tyvar bv;


    (* In some instances we will strip off a layer of mutability or
     * exterior-ness, as trans is willing to transplant and/or overlook
     * mutability / exterior differences wrt. many operators.
     * 
     * Note: there is a secondary mutability-checking pass in effect.ml to
     * ensure you're not actually mutating the insides of an immutable. That's
     * not the typechecker's job.
     *)
    and unify_tyvars' (simplify:bool) (av:tyvar) (bv:tyvar) : unit =
      let (a, b) = ((resolve_tyvar av), (resolve_tyvar bv)) in
      let wrap tv =
        match !tv with
            TYSPEC_resolved (params, Ast.TY_mutable ty) ->
              tv := TYSPEC_equiv (SIMPL_mutable,
                                  (ref (TYSPEC_resolved (params, ty))));
              true
          | TYSPEC_resolved (params, Ast.TY_exterior ty) ->
              tv := TYSPEC_equiv (SIMPL_exterior,
                                  (ref (TYSPEC_resolved (params, ty))));
              true
          | _ -> false
      in
        if simplify
        then
          if (wrap a) || (wrap b)
          then unify_tyvars' simplify a b
          else unify_tyvars'' a b
        else
          unify_tyvars'' av bv

    and unify_tyvars'' (av:tyvar) (bv:tyvar) : unit =
      let (a, b) = ((resolve_tyvar av), (resolve_tyvar bv)) in
      let fail () =
        err None "mismatched types: %s vs. %s" (tyspec_to_str !av)
          (tyspec_to_str !bv);
      in

      let merge_dicts a b =
        let c = Hashtbl.create ((Hashtbl.length a) + (Hashtbl.length b)) in
        let merge ident tv_a =
          if Hashtbl.mem c ident
          then unify_tyvars false (Hashtbl.find c ident) tv_a
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
          unify_ty false (find_ty ident) tv
        in
          Hashtbl.iter check_entry dct
      in

      let unify_dict_with_obj_fns
          (dct:dict)
          (fns:(Ast.ident,Ast.ty_fn) Hashtbl.t) : unit =
        let check_entry (query:Ast.ident) tv : unit =
          match htab_search fns query with
              None -> fail ()
            | Some fn -> unify_ty false (Ast.TY_fn fn) tv
        in
          Hashtbl.iter check_entry dct
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
          | Ast.TY_exterior ty
          | Ast.TY_mutable ty
          | Ast.TY_constrained (ty, _) ->
              is_comparable_or_ordered comparable ty
      in

      let floating (ty:Ast.ty) : bool =
        match ty with
            Ast.TY_mach TY_f32 | Ast.TY_mach TY_f64 -> true
          | _ -> false
      in

      let integral (ty:Ast.ty) : bool =
        match ty with
            Ast.TY_int | Ast.TY_uint | Ast.TY_mach TY_u8 | Ast.TY_mach TY_u16
          | Ast.TY_mach TY_u32 | Ast.TY_mach TY_u64 | Ast.TY_mach TY_i8
          | Ast.TY_mach TY_i16 | Ast.TY_mach TY_i32
          | Ast.TY_mach TY_i64 ->
              true
          | _ -> false
      in

      let numeric (ty:Ast.ty) : bool = (integral ty) || (floating ty) in

      let plusable (ty:Ast.ty) : bool =
        match ty with
            Ast.TY_str -> true
          | Ast.TY_vec _ -> true
          | _ -> numeric ty
      in

      let loggable (ty:Ast.ty) : bool =
        match ty with
            Ast.TY_str | Ast.TY_bool | Ast.TY_int | Ast.TY_uint
          | Ast.TY_char
          | Ast.TY_mach TY_u8 | Ast.TY_mach TY_u16 | Ast.TY_mach TY_u32
          | Ast.TY_mach TY_i8 | Ast.TY_mach TY_i16 | Ast.TY_mach TY_i32
              -> true
          | _ -> false
      in

      let result =
        match (!a, !b) with
            (TYSPEC_equiv _, _) | (_, TYSPEC_equiv _) ->
              bug () "equiv found even though tyvar was resolved"

          | (TYSPEC_all, other) | (other, TYSPEC_all) -> other

          (* resolved *)

          | (TYSPEC_resolved (params_a, ty_a),
             TYSPEC_resolved (params_b, ty_b)) ->
              if params_a <> params_b || ty_a <> ty_b
              then fail()
              else TYSPEC_resolved (params_a, ty_a)

          | (TYSPEC_resolved (params, ty),
             TYSPEC_callable (out_tv, in_tvs))
          | (TYSPEC_callable (out_tv, in_tvs),
             TYSPEC_resolved (params, ty)) ->
              let unify_in_slot i in_slot =
                unify_slot true in_slot None in_tvs.(i)
              in
                begin
                  match ty with
                      Ast.TY_fn ({
                                   Ast.sig_input_slots = in_slots;
                                   Ast.sig_output_slot = out_slot
                                 }, _) ->
                        if Array.length in_slots != Array.length in_tvs
                        then fail ();
                        unify_slot true out_slot None out_tv;
                        Array.iteri unify_in_slot in_slots
                    | _ -> fail ()
                end;
                TYSPEC_resolved (params, ty)

          | (TYSPEC_resolved (params, ty), TYSPEC_collection tv)
          | (TYSPEC_collection tv, TYSPEC_resolved (params, ty)) ->
              begin
                match ty with
                    Ast.TY_vec ty -> unify_ty false ty tv
                  | Ast.TY_str -> unify_ty false (Ast.TY_mach TY_u8) tv
                  | _ -> fail ()
              end;
              TYSPEC_resolved (params, ty)

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
              begin
                match ty with
                    Ast.TY_rec fields ->
                      unify_dict_with_record_fields dct fields
                  | Ast.TY_obj (_, fns) ->
                      unify_dict_with_obj_fns dct fns
                  | _ -> fail ()
              end;
              TYSPEC_resolved (params, ty)

          | (TYSPEC_resolved (params, ty), TYSPEC_integral)
          | (TYSPEC_integral, TYSPEC_resolved (params, ty)) ->
              if not (integral ty)
              then fail ()
              else TYSPEC_resolved (params, ty)

          | (TYSPEC_resolved (params, ty), TYSPEC_loggable)
          | (TYSPEC_loggable, TYSPEC_resolved (params, ty)) ->
              if not (loggable ty)
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
                unify_ty false ty tv;
                TYSPEC_resolved ([| |], ty)

          | (TYSPEC_resolved (params, ty), TYSPEC_record dct)
          | (TYSPEC_record dct, TYSPEC_resolved (params, ty)) ->
              begin
                match ty with
                    Ast.TY_rec fields ->
                      unify_dict_with_record_fields dct fields
                  | _ -> fail ()
              end;
              TYSPEC_resolved (params, ty)

          | (TYSPEC_resolved (params, ty), TYSPEC_tuple tvs)
          | (TYSPEC_tuple tvs, TYSPEC_resolved (params, ty)) ->
              begin
                match ty with
                    Ast.TY_tup (elem_tys:Ast.ty array) ->
                      if (Array.length elem_tys) <> (Array.length tvs)
                      then fail ()
                      else
                        let check_elem i tv =
                          unify_ty false (elem_tys.(i)) tv
                        in
                          Array.iteri check_elem tvs
                  | _ -> fail ()
              end;
              TYSPEC_resolved (params, ty)

          | (TYSPEC_resolved (params, ty), TYSPEC_vector tv)
          | (TYSPEC_vector tv, TYSPEC_resolved (params, ty)) ->
              begin
                match ty with
                    Ast.TY_vec ty ->
                      unify_ty false ty tv;
                      TYSPEC_resolved (params, Ast.TY_vec ty)
                  | _ -> fail ()
              end

          (* callable *)

          | (TYSPEC_callable (a_out_tv, a_in_tvs),
             TYSPEC_callable (b_out_tv, b_in_tvs)) ->
              unify_tyvars true a_out_tv b_out_tv;
              let check_in_tv i a_in_tv =
                unify_tyvars true a_in_tv b_in_tvs.(i)
              in
                Array.iteri check_in_tv a_in_tvs;
                unify_tyvars true a_out_tv b_out_tv;
                TYSPEC_callable (a_out_tv, a_in_tvs)

          | (TYSPEC_callable _, TYSPEC_collection _)
          | (TYSPEC_callable _, TYSPEC_comparable)
          | (TYSPEC_callable _, TYSPEC_plusable)
          | (TYSPEC_callable _, TYSPEC_dictionary _)
          | (TYSPEC_callable _, TYSPEC_integral)
          | (TYSPEC_callable _, TYSPEC_loggable)
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
          | (TYSPEC_loggable, TYSPEC_callable _)
          | (TYSPEC_numeric, TYSPEC_callable _)
          | (TYSPEC_ordered, TYSPEC_callable _)
          | (TYSPEC_app _, TYSPEC_callable _)
          | (TYSPEC_record _, TYSPEC_callable _)
          | (TYSPEC_tuple _, TYSPEC_callable _)
          | (TYSPEC_vector _, TYSPEC_callable _) -> fail ()

          (* collection *)

          | (TYSPEC_collection av, TYSPEC_collection bv) ->
              unify_tyvars false av bv;
              TYSPEC_collection av

          | (TYSPEC_collection av, TYSPEC_comparable)
          | (TYSPEC_comparable, TYSPEC_collection av) ->
              TYSPEC_collection av

          | (TYSPEC_collection v, TYSPEC_plusable)
          | (TYSPEC_plusable, TYSPEC_collection v) -> TYSPEC_collection v

          | (TYSPEC_collection _, TYSPEC_dictionary _)
          | (TYSPEC_collection _, TYSPEC_integral)
          | (TYSPEC_collection _, TYSPEC_loggable)
          | (TYSPEC_collection _, TYSPEC_numeric)
          | (TYSPEC_collection _, TYSPEC_ordered)
          | (TYSPEC_collection _, TYSPEC_app _)
          | (TYSPEC_collection _, TYSPEC_record _)
          | (TYSPEC_collection _, TYSPEC_tuple _)
          | (TYSPEC_dictionary _, TYSPEC_collection _)
          | (TYSPEC_integral, TYSPEC_collection _)
          | (TYSPEC_loggable, TYSPEC_collection _)
          | (TYSPEC_numeric, TYSPEC_collection _)
          | (TYSPEC_ordered, TYSPEC_collection _)
          | (TYSPEC_app _, TYSPEC_collection _)
          | (TYSPEC_record _, TYSPEC_collection _)
          | (TYSPEC_tuple _, TYSPEC_collection _) -> fail ()

          | (TYSPEC_collection av, TYSPEC_vector bv)
          | (TYSPEC_vector bv, TYSPEC_collection av) ->
              unify_tyvars false av bv;
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

          | (TYSPEC_comparable, TYSPEC_loggable)
          | (TYSPEC_loggable, TYSPEC_comparable) -> TYSPEC_loggable

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

          | (TYSPEC_plusable, TYSPEC_loggable)
          | (TYSPEC_loggable, TYSPEC_plusable) -> TYSPEC_plusable

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
          | (TYSPEC_dictionary _, TYSPEC_loggable)
          | (TYSPEC_dictionary _, TYSPEC_numeric)
          | (TYSPEC_dictionary _, TYSPEC_ordered)
          | (TYSPEC_dictionary _, TYSPEC_app _)
          | (TYSPEC_integral, TYSPEC_dictionary _)
          | (TYSPEC_loggable, TYSPEC_dictionary _)
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
          | (TYSPEC_integral, TYSPEC_loggable)
          | (TYSPEC_integral, TYSPEC_numeric)
          | (TYSPEC_integral, TYSPEC_ordered)
          | (TYSPEC_loggable, TYSPEC_integral)
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

          (* loggable *)

          | (TYSPEC_loggable, TYSPEC_loggable) -> TYSPEC_loggable

          | (TYSPEC_loggable, TYSPEC_numeric)
          | (TYSPEC_numeric, TYSPEC_loggable) -> TYSPEC_numeric

          | (TYSPEC_loggable, TYSPEC_ordered)
          | (TYSPEC_ordered, TYSPEC_loggable) -> TYSPEC_ordered

          | (TYSPEC_loggable, TYSPEC_app _)
          | (TYSPEC_loggable, TYSPEC_record _)
          | (TYSPEC_loggable, TYSPEC_tuple _)
          | (TYSPEC_loggable, TYSPEC_vector _)
          | (TYSPEC_app _, TYSPEC_loggable)
          | (TYSPEC_record _, TYSPEC_loggable)
          | (TYSPEC_tuple _, TYSPEC_loggable)
          | (TYSPEC_vector _, TYSPEC_loggable) -> fail ()

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
                  unify_tyvars false tv_a tv_b;
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
                  unify_tyvars false tvs_a.(i) tvs_b.(i);
                  tvs_a.(i)
                end
              in
                TYSPEC_tuple (Array.init max_len init_tuple_elem)

          | (TYSPEC_tuple _, TYSPEC_vector _)
          | (TYSPEC_vector _, TYSPEC_tuple _) -> fail ()

          (* vector *)

          | (TYSPEC_vector av, TYSPEC_vector bv) ->
              unify_tyvars false av bv;
              TYSPEC_vector av
      in
      let c = ref result in
        a := TYSPEC_equiv (SIMPL_none, c);
        b := TYSPEC_equiv (SIMPL_none, c)

    and unify_ty_parametric
        (simplify:bool)
        (ty:Ast.ty)
        (tps:Ast.ty_param array)
        (tv:tyvar)
        : unit =
      unify_tyvars simplify (ref (TYSPEC_resolved (tps, ty))) tv

    and unify_ty (simplify:bool) (ty:Ast.ty) (tv:tyvar) : unit =
      unify_ty_parametric simplify ty [||] tv

    in

    let rec unify_lit (simplify:bool) (lit:Ast.lit) (tv:tyvar) : unit =
      let ty =
        match lit with
            Ast.LIT_nil -> Ast.TY_nil
          | Ast.LIT_bool _ -> Ast.TY_bool
          | Ast.LIT_mach (mty, _, _) -> Ast.TY_mach mty
          | Ast.LIT_int (_, _) -> Ast.TY_int
          | Ast.LIT_uint (_, _) -> Ast.TY_uint
          | Ast.LIT_char _ -> Ast.TY_char
      in
        unify_ty simplify ty tv

    and unify_atom (simplify:bool) (atom:Ast.atom) (tv:tyvar) : unit =
      match atom with
          Ast.ATOM_literal { node = literal; id = _ } ->
            unify_lit simplify literal tv
        | Ast.ATOM_lval lval ->
            unify_lval simplify lval tv

    and unify_expr (expr:Ast.expr) (tv:tyvar) : unit =
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
                      unify_atom true lhs
                        (ref (TYSPEC_resolved ([||], Ast.TY_bool)));
                      unify_atom true rhs
                        (ref (TYSPEC_resolved ([||], Ast.TY_bool)));
                      unify_ty true Ast.TY_bool tv
                  | BINOPSIG_comp_comp_bool ->
                      let tv_a = ref TYSPEC_comparable in
                        unify_atom true lhs tv_a;
                        unify_atom true rhs tv_a;
                        unify_ty true Ast.TY_bool tv
                  | BINOPSIG_ord_ord_bool ->
                      let tv_a = ref TYSPEC_ordered in
                        unify_atom true lhs tv_a;
                        unify_atom true rhs tv_a;
                        unify_ty true Ast.TY_bool tv
                  | BINOPSIG_integ_integ_integ ->
                      let tv_a = ref TYSPEC_integral in
                        unify_atom true lhs tv_a;
                        unify_atom true rhs tv_a;
                        unify_tyvars true tv tv_a
                  | BINOPSIG_num_num_num ->
                      let tv_a = ref TYSPEC_numeric in
                        unify_atom true lhs tv_a;
                        unify_atom true rhs tv_a;
                        unify_tyvars true tv tv_a
                  | BINOPSIG_plus_plus_plus ->
                      let tv_a = ref TYSPEC_plusable in
                        unify_atom true lhs tv_a;
                        unify_atom true rhs tv_a;
                        unify_tyvars true tv tv_a
              end
        | Ast.EXPR_unary (unop, atom) ->
            begin
              match unop with
                  Ast.UNOP_not ->
                    unify_atom true atom
                      (ref (TYSPEC_resolved ([||], Ast.TY_bool)));
                    unify_ty true Ast.TY_bool tv
                | Ast.UNOP_bitnot ->
                    let tv_a = ref TYSPEC_integral in
                      unify_atom true atom tv_a;
                      unify_tyvars true tv tv_a
                | Ast.UNOP_neg ->
                    let tv_a = ref TYSPEC_numeric in
                      unify_atom true atom tv_a;
                      unify_tyvars true tv tv_a
                | Ast.UNOP_cast t ->
                    (* FIXME (issue #84): check cast-validity in
                     * post-typecheck pass.  Only some casts make sense.
                     *)
                    let tv_a = ref TYSPEC_all in
                    let t = Hashtbl.find cx.ctxt_all_cast_types t.id in
                      unify_atom true atom tv_a;
                      unify_ty true t tv
            end
        | Ast.EXPR_atom atom -> unify_atom true atom tv

    and unify_lval' (simplify:bool) (lval:Ast.lval) (tv:tyvar) : unit =
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
                        unify_slot simplify slot (Some referent) tv

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
                          unify_tyvars simplify (ref spec) tv
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
                    unify_atom simplify atom
                      (ref (TYSPEC_resolved ([||], Ast.TY_int)));
                    TYSPEC_collection tv
              in
              let base_tv = ref base_ts in
                unify_lval' simplify base base_tv;
                match !(resolve_tyvar base_tv) with
                    TYSPEC_resolved (_, ty) ->
                      unify_ty simplify (project_type ty comp) tv
                  | _ ->
                      ()

    and unify_lval (simplify:bool) (lval:Ast.lval) (tv:tyvar) : unit =
      let id = lval_base_id lval in
        (* Fetch lval with type components resolved. *)
        let lval = Hashtbl.find cx.ctxt_all_lvals id in
        iflog cx (fun _ -> log cx
                    "fetched resolved version of lval #%d = %a"
                    (int_of_node id) Ast.sprintf_lval lval);
          Hashtbl.add lval_tyvars id tv;
          unify_lval' simplify lval tv

    in
    let gen_atom_tvs atoms =
      let gen_atom_tv atom =
        let tv = ref TYSPEC_all in
          unify_atom true atom tv;
          tv
      in
        Array.map gen_atom_tv atoms
    in
    let visit_stmt_pre_full (stmt:Ast.stmt) : unit =

      let check_callable out_tv callee args =
        let in_tvs = gen_atom_tvs args in
        let callee_tv = ref (TYSPEC_callable (out_tv, in_tvs)) in
          unify_lval true callee callee_tv;
      in
      match stmt.node with
          Ast.STMT_spawn (out, _, callee, args) ->
            let out_tv = ref (TYSPEC_resolved ([||], Ast.TY_nil)) in
              unify_lval true out (ref (TYSPEC_resolved ([||], Ast.TY_task)));
              check_callable out_tv callee args

        | Ast.STMT_init_rec (lval, fields, Some base) ->
            let dct = Hashtbl.create 10 in
            let tvrec = ref (TYSPEC_record dct) in
            let add_field (ident, atom) =
              let tv = ref TYSPEC_all in
                unify_atom true atom tv;
                Hashtbl.add dct ident tv
            in
              Array.iter add_field fields;
              let tvbase = ref TYSPEC_all in
                unify_lval true base tvbase;
                unify_tyvars true tvrec tvbase;
                unify_lval true lval tvrec

        | Ast.STMT_init_rec (lval, fields, None) ->
            let dct = Hashtbl.create 10 in
            let add_field (ident, atom) =
              let tv = ref TYSPEC_all in
                unify_atom true atom tv;
                Hashtbl.add dct ident tv
            in
              Array.iter add_field fields;
              unify_lval true lval (ref (TYSPEC_record dct))

        | Ast.STMT_init_tup (lval, members) ->
            let member_to_tv atom =
              let tv = ref TYSPEC_all in
                unify_atom true atom tv;
                tv
            in
            let member_tvs = Array.map member_to_tv members in
              unify_lval true lval (ref (TYSPEC_tuple member_tvs))

        | Ast.STMT_init_vec (lval, atoms) ->
            let tv = ref TYSPEC_all in
            let unify_with_tv atom = unify_atom true atom tv in
              Array.iter unify_with_tv atoms;
              unify_lval true lval (ref (TYSPEC_vector tv))

        | Ast.STMT_init_str (lval, _) ->
            unify_lval true lval (ref (TYSPEC_resolved ([||], Ast.TY_str)))

        | Ast.STMT_copy (lval, expr) ->
            let tv = ref TYSPEC_all in
              unify_expr expr tv;
              unify_lval true lval tv

        | Ast.STMT_copy_binop (lval, binop, at) ->
            let tv = ref TYSPEC_all in
              unify_expr (Ast.EXPR_binary (binop, Ast.ATOM_lval lval, at)) tv;
              unify_lval true lval tv;

        | Ast.STMT_call (out, callee, args) ->
            let out_tv = ref TYSPEC_all in
              unify_lval true out out_tv;
              check_callable out_tv callee args

        | Ast.STMT_log atom -> unify_atom true atom (ref TYSPEC_loggable)

        | Ast.STMT_check_expr expr ->
            unify_expr expr (ref (TYSPEC_resolved ([||], Ast.TY_bool)))

        | Ast.STMT_check (_, check_calls) ->
            let out_tv = ref (TYSPEC_resolved ([||], Ast.TY_bool)) in
              Array.iter
                (fun (callee, args) ->
                   check_callable out_tv callee args)
                check_calls

        | Ast.STMT_while { Ast.while_lval = (_, expr); Ast.while_body = _ } ->
            unify_expr expr (ref (TYSPEC_resolved ([||], Ast.TY_bool)))

        | Ast.STMT_if { Ast.if_test = if_test } ->
            unify_expr if_test (ref (TYSPEC_resolved ([||], Ast.TY_bool)));

        | Ast.STMT_decl _ -> ()

        | Ast.STMT_ret atom_opt
        | Ast.STMT_put atom_opt ->
            begin
              match atom_opt with
                  None -> unify_ty true Ast.TY_nil (retval_tv())
                | Some atom -> unify_atom true atom (retval_tv())
            end

        | Ast.STMT_be (callee, args) ->
            check_callable (retval_tv()) callee args

        | Ast.STMT_bind (bound, callee, arg_opts) ->
            (* FIXME (issue #81): handle binding type parameters
             * eventually.
             *)
            let out_tv = ref TYSPEC_all in
            let residue = ref [] in
            let gen_atom_opt_tvs atoms =
              let gen_atom_tv atom_opt =
                let tv = ref TYSPEC_all in
                  begin
                    match atom_opt with
                        None -> residue := tv :: (!residue);
                      | Some atom -> unify_atom true atom tv
                  end;
                  tv
              in
                Array.map gen_atom_tv atoms
            in

            let in_tvs = gen_atom_opt_tvs arg_opts in
            let arg_residue_tvs = Array.of_list (List.rev (!residue)) in
            let callee_tv = ref (TYSPEC_callable (out_tv, in_tvs)) in
            let bound_tv = ref (TYSPEC_callable (out_tv, arg_residue_tvs)) in
              unify_lval true callee callee_tv;
              unify_lval true bound bound_tv

        | Ast.STMT_for_each fe ->
            let out_tv = ref TYSPEC_all in
            let (si, _) = fe.Ast.for_each_slot in
            let (callee, args) = fe.Ast.for_each_call in
              unify_slot true si.node (Some si.id) out_tv;
              check_callable out_tv callee args

        | Ast.STMT_for fo ->
            let mem_tv = ref TYSPEC_all in
            let seq_tv = ref (TYSPEC_collection mem_tv) in
            let (si, _) = fo.Ast.for_slot in
            let (_, seq) = fo.Ast.for_seq in
              unify_lval true seq seq_tv;
              unify_slot true si.node (Some si.id) mem_tv

        | Ast.STMT_alt_tag
            { Ast.alt_tag_lval = lval; Ast.alt_tag_arms = arms } ->
            let lval_tv = ref TYSPEC_all in
              unify_lval true lval lval_tv;
              Array.iter (fun _ -> push_pat_tv lval_tv) arms

        (* FIXME (issue #52): plenty more to handle here. *)
        | _ ->
            log cx "warning: not typechecking stmt %s\n"
              (Ast.sprintf_stmt () stmt)
    in

    let visit_stmt_pre (stmt:Ast.stmt) : unit =
      try
        visit_stmt_pre_full stmt;
        (* 
         * Reset any item-parameters that were resolved to types
         * during inference for this statement.
         *)
        Hashtbl.iter
          (fun _ params -> Array.iter (fun tv -> tv := TYSPEC_all) params)
          item_params;
      with
          Semant_err (None, msg) ->
            raise (Semant_err ((Some stmt.id), msg))
    in

    let enter_fn fn retspec =
      let out = fn.Ast.fn_output_slot in
        push_retval_tv (ref retspec);
        unify_slot true out.node (Some out.id) (retval_tv())
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
                          interior_slot (Ast.TY_vec Ast.TY_str)
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
            Ast.PAT_lit lit -> unify_lit true lit expected

          | Ast.PAT_tag (lval, _) ->
              let expect ty =
                let tv = ref TYSPEC_all in
                  unify_ty true ty tv;
                  push_pat_tv tv;
              in

              let lval_nm = lval_to_name lval in

              (* The lval here is our tag constructor, which we've already
               * resolved (in Resolve) to have a the actual tag constructor
               * function item as its referent.  It should hence unify
               * exactly to that function type, rebuilt under any latent type
               * parameters applied in the lval. *)
              let lval_tv = ref TYSPEC_all in
                unify_lval true lval lval_tv;
                let tag_ctor_ty =
                  match !(resolve_tyvar lval_tv) with
                      TYSPEC_resolved (_, ty) -> ty
                    | _ ->
                        bug () "tag constructor is not a fully resolved type."
                in

                let tag_ty = fn_output_ty tag_ctor_ty in
                let tag_ty_tup = tag_or_iso_ty_tup_by_name tag_ty lval_nm in

                let tag_tv = ref TYSPEC_all in
                  unify_ty true tag_ty tag_tv;
                  unify_tyvars true expected tag_tv;
                  List.iter expect
                    (List.rev (Array.to_list tag_ty_tup));

          | Ast.PAT_slot (sloti, _) ->
              unify_slot true sloti.node (Some sloti.id) expected

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
              Hashtbl.add bindings id (ref TYSPEC_all)
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
                          Hashtbl.replace cx.ctxt_all_defns id
                            (DEFN_slot { slot_defn with
                                           Ast.slot_ty = Some ty })
                  end
              | _ -> bug () "check_auto_tyvar: no slot defn"
        in

        let rec get_resolved_ty tv id =
          match !tv with
              TYSPEC_resolved ([||], ty) -> ty
            | TYSPEC_vector tv ->
                Ast.TY_vec (get_resolved_ty tv id)
            | TYSPEC_equiv (SIMPL_none, tv) ->
                get_resolved_ty tv id
            | TYSPEC_equiv (SIMPL_mutable, tv) ->
                Ast.TY_mutable (get_resolved_ty tv id)
            | TYSPEC_equiv (SIMPL_exterior, tv) ->
                Ast.TY_exterior (get_resolved_ty tv id)
            | _ -> err (Some id)
                "unresolved type %s (%d)"
                  (tyspec_to_str !tv)
                  (int_of_node id)
        in

        let check_auto_tyvar id =
          let tv = Hashtbl.find bindings id in
          let ty = get_resolved_ty tv id in
            update_auto_tyvar id ty
        in

        let record_lval_ty id tv =
          let ty = get_resolved_ty tv id in
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

