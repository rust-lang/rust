(* rust/src/boot/me/type.ml *)

(* An ltype is the type of a segment of an lvalue. It is used only by
 * [check_lval] and friends and are the closest Rust ever comes to polymorphic
 * types. All ltypes must be resolved into monotypes by the time an outer
 * lvalue (i.e. an lvalue whose parent is not also an lvalue) is finished
 * typechecking. *)

type ltype =
    LTYPE_mono of Ast.ty
  | LTYPE_poly of Ast.ty_param array * Ast.ty   (* "big lambda" *)
  | LTYPE_module of Ast.mod_items               (* type of a module *)

(* A "type pattern" used for inference. *)
type ty_pat =
    TYPAT_wild                (* matches any type *)
  | TYPAT_ty of Ast.ty        (* matches only the given type *)
  | TYPAT_fn of Ast.ty array  (* matches a function with some arg types *)

type fn_ctx = {
  fnctx_return_type: Ast.ty;
  fnctx_is_iter: bool;
  mutable fnctx_just_saw_ret: bool;
  fnctx_blocks: Common.node_id Stack.t;
  fnctx_slot_decls: (Ast.slot_key,Common.node_id) Hashtbl.t;
}

exception Type_error of string * string

let log cx =
  Session.log
    "type"
    (Semant.should_log cx cx.Semant.ctxt_sess.Session.sess_log_type)
    cx.Semant.ctxt_sess.Session.sess_log_out

let iflog cx thunk =
  if (Semant.should_log cx cx.Semant.ctxt_sess.Session.sess_log_type)
  then thunk ()
  else ()
;;

let head_only ty =
  match ty with

      Ast.TY_tup _ -> "tup(...)"
    | Ast.TY_rec _ -> "rec(...)"

    | Ast.TY_fn _ -> "fn (...) -> ..."

    | Ast.TY_vec _ -> "vec[...]"
    | Ast.TY_chan _ -> "chan[...]"
    | Ast.TY_port _ -> "port[...]"

    | Ast.TY_obj _ -> "obj { ... }"
    | Ast.TY_box _ -> "@(...)"
    | Ast.TY_mutable _ -> "(mutable ...)"
    | Ast.TY_constrained _ -> "(... : <constrained>)"

    | _ -> Printf.sprintf "%a" Ast.sprintf_ty ty
;;


let rec rec_diff
    (cx:Semant.ctxt)
    (a:Ast.ty_rec) (b:Ast.ty_rec)
    (bbuf:Buffer.t)
    : unit =

  Buffer.add_string bbuf "rec(";

  let rec append_first_diff buf a b i =
    let alen = Array.length a in
    let blen = Array.length b in
      if i >= alen
      then
        Buffer.add_string buf "...)"
      else
        if i >= blen
        then
          Printf.bprintf buf
            "... <%d elements>)" (blen - i)
        else
          let (alab, aty) = a.(i) in
          let (blab, bty) = b.(i) in
            if alab <> blab
            then
              Printf.bprintf buf "... <ty> %s ...)" alab
            else
              if aty <> bty
              then
                let (a,_) = summarize_difference cx aty bty in
                  Printf.bprintf buf "... %s %s ...)" a alab
              else
                append_first_diff buf a b (i+1)
  in
    append_first_diff bbuf b a 0;
    Buffer.add_string bbuf ")";


and tup_diff
    (cx:Semant.ctxt)
    (a:Ast.ty_tup) (b:Ast.ty_tup)
    (bbuf:Buffer.t)
    : unit =

  Buffer.add_string bbuf "tup(";

  let rec append_first_diff buf a b i =
    let alen = Array.length a in
    let blen = Array.length b in
      if i >= alen
      then
        Buffer.add_string buf "...)"
      else
        if i >= blen
        then
          Printf.bprintf buf
            "... <%d elements>)" (blen - i)
        else
          let (aty) = a.(i) in
          let (bty) = b.(i) in
            if aty <> bty
            then
              let (a,_) = summarize_difference cx aty bty in
                Printf.bprintf buf "... %s ...)" a
            else
              append_first_diff buf a b (i+1)
  in
    append_first_diff bbuf b a 0;
    Buffer.add_string bbuf ")";


and summarize_difference cx (expected:Ast.ty) (actual:Ast.ty)
    : (string * string) =
  if expected = actual
  then ("_", "_")
  else
    begin
      let ebuf = Buffer.create 10 in
      let abuf = Buffer.create 10 in

      let p s =
        Buffer.add_string abuf s
      in

      let sub e a =
        let (_, a) = summarize_difference cx e a in
          Printf.bprintf abuf "%s" a
      in

      Buffer.add_string ebuf (Semant.pretty_ty_str cx head_only expected);

        begin
          match expected, actual with
              (Ast.TY_tup etys,  Ast.TY_tup atys) ->
                tup_diff cx etys atys abuf

            | (Ast.TY_rec eelts, Ast.TY_rec aelts) ->
                rec_diff cx eelts aelts abuf

            | (Ast.TY_vec e, Ast.TY_vec a) ->
                p "vec["; sub e a; p "]";

            | (Ast.TY_chan e, Ast.TY_port a) ->
                p "chan["; sub e a; p "]";

            | (Ast.TY_port e, Ast.TY_port a) ->
                p "port["; sub e a; p "]";

            | (Ast.TY_box e, Ast.TY_box a) ->
                p "@"; sub e a;

            | (Ast.TY_mutable e, Ast.TY_mutable a) ->
                p "mutable "; sub e a;

            | (_, a) ->
                Buffer.add_string abuf (Semant.pretty_ty_str cx head_only a)
        end;
        (Buffer.contents ebuf, Buffer.contents abuf)
    end
;;

let type_error_full expected actual =
  raise (Type_error (expected, actual))
;;

let type_error cx expected actual =
  type_error_full expected (Semant.pretty_ty_str cx head_only actual)
;;

(* We explicitly curry [cx] like this to avoid threading it through all the
 * inner functions. *)
let check_block (cx:Semant.ctxt) : (fn_ctx -> Ast.block -> unit) =
  let pretty_ty_str = Semant.pretty_ty_str cx (Ast.sprintf_ty ()) in

  (* Returns the part of the type that matters for typechecking. *)
  let rec fundamental_ty (ty:Ast.ty) : Ast.ty =
    match ty with
        Ast.TY_constrained (ty', _) | Ast.TY_mutable ty' -> fundamental_ty ty'
      | _ -> ty
  in

  let sprintf_ltype _ (lty:ltype) : string =
    match lty with
        LTYPE_mono ty | LTYPE_poly (_, ty) -> pretty_ty_str ty
      | LTYPE_module items -> Ast.sprintf_mod_items () items
  in

  let get_slot_ty (slot:Ast.slot) : Ast.ty =
    match slot.Ast.slot_ty with
        Some ty -> ty
      | None -> Common.bug () "get_slot_ty: no type in slot"
  in

  (* [unbox ty] strips off all boxes in [ty] and returns a tuple consisting of
   * the number of boxes that were stripped off. *)
  let unbox (ty:Ast.ty) : (Ast.ty * int) =
    let rec unbox ty acc =
      match ty with
          Ast.TY_box ty' -> unbox ty' (acc + 1)
        | Ast.TY_mutable ty' | Ast.TY_constrained (ty', _) -> unbox ty' acc
        | _ -> (ty, acc)
    in
    unbox ty 0
  in

  let maybe_mutable (mutability:Ast.mutability) (ty:Ast.ty) : Ast.ty =
    let res =
      if mutability = Ast.MUT_mutable then Ast.TY_mutable ty else ty
    in
      iflog cx
        (fun _ ->
           log cx "maybe_mutable: %a -> %a"
             Ast.sprintf_ty ty Ast.sprintf_ty res);
      res
  in

  (*
   * Type assertions
   *)

  let type_error = type_error cx in

  let is_integer (ty:Ast.ty) =
    match ty with
        Ast.TY_int | Ast.TY_uint
      | Ast.TY_mach Common.TY_u8 | Ast.TY_mach Common.TY_u16
      | Ast.TY_mach Common.TY_u32 | Ast.TY_mach Common.TY_u64
      | Ast.TY_mach Common.TY_i8 | Ast.TY_mach Common.TY_i16
      | Ast.TY_mach Common.TY_i32 | Ast.TY_mach Common.TY_i64 -> true
      | _ -> false
  in

  let demand (expected:Ast.ty) (actual:Ast.ty) : unit =
    let expected, actual = fundamental_ty expected, fundamental_ty actual in
    if expected <> actual then
      let (e,a) = summarize_difference cx expected actual in
        type_error_full e a
  in
  let demand_integer (actual:Ast.ty) : unit =
    if not (is_integer (fundamental_ty actual)) then
      type_error "integer" actual
  in
  let demand_bool_or_char_or_integer_or_native (actual:Ast.ty) : unit =
    match fundamental_ty actual with
        Ast.TY_bool | Ast.TY_char | Ast.TY_native _ -> ()
      | ty when is_integer ty -> ()
      | _ -> type_error "bool, char, integer or native" actual
  in
  let demand_number (actual:Ast.ty) : unit =
    match fundamental_ty actual with
        Ast.TY_int | Ast.TY_uint | Ast.TY_mach _ -> ()
      | ty -> type_error "number" ty
  in
  let demand_number_or_str_or_vector (actual:Ast.ty) : unit =
    match fundamental_ty actual with
        Ast.TY_int | Ast.TY_uint | Ast.TY_mach _ | Ast.TY_str
      | Ast.TY_vec _ ->
          ()
      | ty -> type_error "number, string, or vector" ty
  in
  let demand_vec (actual:Ast.ty) : Ast.ty =
    match fundamental_ty actual with
        Ast.TY_vec ty -> ty
      | ty -> type_error "vector" ty
  in
  let demand_vec_with_mutability
      (mut:Ast.mutability)
      (actual:Ast.ty)
      : Ast.ty =
    match mut, fundamental_ty actual with
        Ast.MUT_mutable, Ast.TY_vec ((Ast.TY_mutable _) as ty) -> ty
      | Ast.MUT_mutable, ty -> type_error "mutable vector" ty
      | Ast.MUT_immutable, ((Ast.TY_vec (Ast.TY_mutable _)) as ty) ->
          type_error "immutable vector" ty
      | Ast.MUT_immutable, Ast.TY_vec ty -> ty
      | Ast.MUT_immutable, ty -> type_error "immutable vector" ty
  in
  let demand_vec_or_str (actual:Ast.ty) : Ast.ty =
    match fundamental_ty actual with
        Ast.TY_vec ty -> ty
      | Ast.TY_str -> Ast.TY_mach Common.TY_u8
      | ty -> type_error "vector or string" ty
  in
  let demand_rec (actual:Ast.ty) : Ast.ty_rec =
    match fundamental_ty actual with
        Ast.TY_rec ty_rec -> ty_rec
      | ty -> type_error "record" ty
  in
  let demand_fn
      ?param_handler:(param_handler=
        fun a idx effect -> demand a (Ast.TY_param (idx, effect)))
      (arg_tys:Ast.ty option array)
      (actual:Ast.ty)
      : Ast.ty =
    let expected = lazy begin
      Format.fprintf Format.str_formatter "fn(";
      let print_arg_ty i arg_ty_opt =
        if i > 0 then Format.fprintf Format.str_formatter ", ";
        match arg_ty_opt with
            None -> Format.fprintf Format.str_formatter "<?>"
          | Some arg_ty -> Ast.fmt_ty Format.str_formatter arg_ty
      in
      Array.iteri print_arg_ty arg_tys;
      Format.fprintf Format.str_formatter ")";
      Format.flush_str_formatter()
    end in
    match fundamental_ty actual with
        Ast.TY_fn (ty_sig, _) as ty ->
          let in_slots = ty_sig.Ast.sig_input_slots in
          if Array.length arg_tys != Array.length in_slots then
            type_error (Lazy.force expected) ty;
          let in_slot_tys = Array.map get_slot_ty in_slots in
          let maybe_demand a_opt b =
            match a_opt, b with
                None, _ -> ()
              | Some a, Ast.TY_param (idx, effect) ->
                  param_handler a idx effect
              | Some a, _ -> demand b a
          in
          Common.arr_iter2 maybe_demand arg_tys in_slot_tys;
          get_slot_ty (ty_sig.Ast.sig_output_slot)
      | ty -> type_error "function" ty
  in
  let demand_chan (actual:Ast.ty) : Ast.ty =
    match fundamental_ty actual with
        Ast.TY_chan ty -> ty
      | ty -> type_error "channel" ty
  in
  let demand_port (actual:Ast.ty) : Ast.ty =
    match fundamental_ty actual with
        Ast.TY_port ty -> ty
      | ty -> type_error "port" ty
  in
  let demand_all (tys:Ast.ty array) : Ast.ty =
    if Array.length tys == 0 then
      Common.bug () "demand_all called with an empty array of types";
    let pivot = fundamental_ty tys.(0) in
    for i = 1 to Array.length tys - 1 do
      demand pivot tys.(i)
    done;
    pivot
  in

  (* Performs beta-reduction (that is, type-param substitution). *)
  let beta_reduce
      (lval_id:Common.node_id)
      (lty:ltype)
      (args:Ast.ty array)
      : ltype =
    if Hashtbl.mem cx.Semant.ctxt_call_lval_params lval_id then
      assert (args = Hashtbl.find cx.Semant.ctxt_call_lval_params lval_id)
    else
      Hashtbl.add cx.Semant.ctxt_call_lval_params lval_id args;

    match lty with
        LTYPE_poly (params, ty) ->
          LTYPE_mono (Semant.rebuild_ty_under_params
                        cx None ty params args true)
      | _ ->
        Common.err None "expected polymorphic type but found %a"
          sprintf_ltype lty
  in

  (*
   * Lvalue and slot checking.
   *
   * We use a slightly different type language here which includes polytypes;
   * see [ltype] above.
   *)

  let check_literal (lit:Ast.lit) : Ast.ty =
    match lit with
        Ast.LIT_nil -> Ast.TY_nil
      | Ast.LIT_bool _ -> Ast.TY_bool
      | Ast.LIT_mach_int (mty, _) -> Ast.TY_mach mty
      | Ast.LIT_int _ -> Ast.TY_int
      | Ast.LIT_uint _ -> Ast.TY_uint
      | Ast.LIT_char _ -> Ast.TY_char
  in

  (* Here the actual inference happens. *)
  let internal_check_slot
      (infer:ty_pat)
      (defn_id:Common.node_id)
      : Ast.ty =
    let slot =
      match Hashtbl.find cx.Semant.ctxt_all_defns defn_id with
          Semant.DEFN_slot slot -> slot
        | _ ->
          Common.bug
            ()
            "internal_check_slot: supplied defn wasn't a slot at all"
    in
    match infer, slot.Ast.slot_ty with
        TYPAT_ty expected, Some actual ->
          demand expected actual;
          actual
      | TYPAT_ty inferred, None ->
          iflog cx
            (fun _ ->
               log cx "setting auto slot #%d = %a to type %a"
                 (Common.int_of_node defn_id)
                 Ast.sprintf_slot_key
                 (Hashtbl.find cx.Semant.ctxt_slot_keys defn_id)
                 Ast.sprintf_ty inferred);
          let new_slot = { slot with Ast.slot_ty = Some inferred } in
          Hashtbl.replace cx.Semant.ctxt_all_defns defn_id
            (Semant.DEFN_slot new_slot);
          inferred
      | TYPAT_fn arg_tys, Some actual ->
          ignore (demand_fn (Array.map (fun ty -> Some ty) arg_tys) actual);
          actual
      | TYPAT_wild, Some actual -> actual
      | TYPAT_fn _, None | TYPAT_wild, None ->
          Common.err None "can't infer any type for this slot"
  in

  let internal_check_mod_item_decl
      (mid:Ast.mod_item_decl)
      (mid_id:Common.node_id)
      : ltype =
    match mid.Ast.decl_item with
        Ast.MOD_ITEM_mod (_, items) -> LTYPE_module items
      | Ast.MOD_ITEM_fn _ | Ast.MOD_ITEM_obj _ | Ast.MOD_ITEM_tag _ ->
          let ty = Hashtbl.find cx.Semant.ctxt_all_item_types mid_id in
          let params = mid.Ast.decl_params in
          if Array.length params == 0 then
            LTYPE_mono ty
          else
            LTYPE_poly ((Array.map (fun p -> p.Common.node) params), ty)
      | Ast.MOD_ITEM_const _ ->
          let ty = Hashtbl.find cx.Semant.ctxt_all_item_types mid_id in
            LTYPE_mono ty
      | Ast.MOD_ITEM_type _ ->
          Common.err None "Type-item used in non-type context"
  in

  let rec internal_check_base_lval
      (infer:ty_pat)
      (nbi:Ast.name_base Common.identified)
      : ltype =
    let lval_id = nbi.Common.id in
    let defn_id = Semant.lval_base_id_to_defn_base_id cx lval_id in
    let lty =
      match Hashtbl.find cx.Semant.ctxt_all_defns defn_id with
          Semant.DEFN_slot _ ->
            LTYPE_mono (internal_check_slot infer defn_id)
        | Semant.DEFN_item mid -> internal_check_mod_item_decl mid defn_id
        | _ -> Common.bug () "internal_check_base_lval: unexpected defn type"
    in
    match nbi.Common.node with
        Ast.BASE_ident _ | Ast.BASE_temp _ -> lty
      | Ast.BASE_app (_, args) -> beta_reduce lval_id lty args

  and internal_check_ext_lval
      (base:Ast.lval)
      (comp:Ast.lval_component)
      : ltype =
    let base_ity =
      match internal_check_lval TYPAT_wild base with
          LTYPE_poly (_, ty) ->
            Common.err None "can't index the polymorphic type '%s'"
              (pretty_ty_str ty)
        | LTYPE_mono ty -> `Type (fundamental_ty ty)
        | LTYPE_module items -> `Module items
    in

    let string_of_itype () =
      match base_ity with
          `Type ty -> pretty_ty_str ty
        | `Module items -> Ast.sprintf_mod_items () items
    in

    let _ =
      iflog cx
        (fun _ ->
           log cx "base lval %a, base type %s"
             Ast.sprintf_lval base (string_of_itype ()))
    in

    let rec typecheck base_ity =
      match base_ity, comp with
          `Type (Ast.TY_rec ty_rec), Ast.COMP_named (Ast.COMP_ident id) ->
            let find _ (k, v) = if id = k then Some v else None in
            let comp_ty =
              match Common.arr_search ty_rec find with
                  Some elem_ty -> elem_ty
                | None ->
                    Common.err
                      None
                      "field '%s' is not one of the fields of '%s'"
                      id
                      (string_of_itype ())
            in
            LTYPE_mono comp_ty

        | `Type (Ast.TY_rec _), _ ->
            Common.err None "the record type '%s' must be indexed by name"
              (string_of_itype ())

        | `Type (Ast.TY_obj ty_obj), Ast.COMP_named (Ast.COMP_ident id) ->
            let comp_ty =
              try
                Ast.TY_fn (Hashtbl.find (snd ty_obj) id)
              with Not_found ->
                Common.err
                  None
                  "method '%s' is not one of the methods of '%s'"
                  id
                  (string_of_itype ())
            in
            LTYPE_mono comp_ty

        | `Type (Ast.TY_obj _), _ ->
            Common.err
              None
              "the object type '%s' must be indexed by name"
              (string_of_itype ())

        | `Type (Ast.TY_tup ty_tup), Ast.COMP_named (Ast.COMP_idx idx)
              when idx < Array.length ty_tup ->
            LTYPE_mono (ty_tup.(idx))

        | `Type (Ast.TY_tup _), Ast.COMP_named (Ast.COMP_idx idx) ->
            Common.err
              None
              "member '_%d' is not one of the members of '%s'"
              idx
              (string_of_itype ())

        | `Type (Ast.TY_tup _), _ ->
            Common.err
              None
              "the tuple type '%s' must be indexed by tuple index"
              (string_of_itype ())

        | `Type (Ast.TY_vec ty_vec), Ast.COMP_atom atom ->
            demand_integer (check_atom atom);
            LTYPE_mono ty_vec

        | `Type (Ast.TY_vec _), _ ->
            Common.err None
              "the vector type '%s' must be indexed by an integral type"
              (string_of_itype ())

        | `Type Ast.TY_str, Ast.COMP_atom atom ->
            demand_integer (check_atom atom);
            LTYPE_mono (Ast.TY_mach Common.TY_u8)

        | `Type Ast.TY_str, _ ->
            Common.err None "strings must be indexed by an integral type"

        | `Type (Ast.TY_box ty_box), Ast.COMP_deref -> LTYPE_mono ty_box

        | `Type (Ast.TY_box ty_box), _ ->
            typecheck (`Type ty_box)  (* automatically dereference! *)

        | `Type ty, Ast.COMP_named (Ast.COMP_ident _) ->
            Common.err None "the type '%s' can't be indexed by name"
              (pretty_ty_str ty)

        | `Type ty, Ast.COMP_named (Ast.COMP_app _) ->
            Common.err
              None
              "the type '%s' has no type parameters, so it can't be applied \
              to types"
              (pretty_ty_str ty)

        | `Module items, Ast.COMP_named ((Ast.COMP_ident id) as name_comp)
        | `Module items, Ast.COMP_named ((Ast.COMP_app (id, _))
                                           as name_comp) ->
            let mod_item =
              try
                Hashtbl.find items id
              with Not_found ->
                Common.err
                  None
                  "the name '%s' can't be found in the module"
                  id
            in
            let lty =
              internal_check_mod_item_decl
                mod_item.Common.node
                mod_item.Common.id
            in
            begin
              match name_comp with
                  Ast.COMP_ident _ -> lty
                | Ast.COMP_app (_, args) ->
                    beta_reduce (Semant.lval_base_id base) lty args
                | _ ->
                  Common.bug ()
                    "internal_check_ext_lval: unexpected name_comp"
            end

        | _, Ast.COMP_named (Ast.COMP_idx _) ->
            Common.err
              None
              "%s isn't a tuple, so it can't be indexed by tuple index"
              (string_of_itype ())

        | _, Ast.COMP_atom atom ->
            Common.err
              None
              "%s can't by indexed by the type '%s'"
              (string_of_itype ())
              (pretty_ty_str (check_atom atom))

        | _, Ast.COMP_deref ->
            Common.err
              None
              "%s isn't a box and can't be dereferenced"
              (string_of_itype ())
    in
    typecheck base_ity

  and internal_check_lval (infer:ty_pat) (lval:Ast.lval) : ltype =
    match lval with
        Ast.LVAL_base nbi -> internal_check_base_lval infer nbi
      | Ast.LVAL_ext (base, comp) -> internal_check_ext_lval base comp

  (* Checks the outermost lvalue and returns the resulting monotype and the
   * number of layers of indirection needed to access it (i.e. the number of
   * boxes that were automatically dereferenced, which will always be zero if
   * the supplied [autoderef] parameter is false). This function is the bridge
   * between the polymorphically typed lvalue world and the monomorphically
   * typed value world. *)
  and internal_check_outer_lval
      ~mut:(mut:Ast.mutability)
      ~deref:(deref:bool)
      (infer:ty_pat)
      (lval:Ast.lval)
      : (Ast.ty * int) =
    let yield_ty ty =
      let (ty, n_boxes) = if deref then unbox ty else (ty, 0) in
        (maybe_mutable mut ty, n_boxes)
    in
    match infer, internal_check_lval infer lval with
        TYPAT_wild, LTYPE_mono ty -> yield_ty ty
      | TYPAT_ty expected, LTYPE_mono actual ->
          demand expected actual;
          yield_ty actual
      | TYPAT_fn arg_tys, LTYPE_mono actual ->
          ignore (demand_fn (Array.map (fun ty -> Some ty) arg_tys) actual);
          yield_ty actual
      | TYPAT_fn arg_tys, (LTYPE_poly (ty_params, ty) as lty) ->
          (* Perform automatic instantiation of polymorphic types. *)
          let ty = fundamental_ty ty in
          let substs = Array.make (Array.length ty_params) None in
          let param_handler substituted_ty idx _ =
            match substs.(idx) with
            | None -> substs.(idx) <- Some substituted_ty
            | Some substituted_ty' -> demand substituted_ty substituted_ty'
          in
          let arg_ty_opts = Array.map (fun ty -> Some ty) arg_tys in
          ignore (demand_fn ~param_handler:param_handler arg_ty_opts ty);
          let get_subst i subst_opt =
            match subst_opt with
                Some subst -> subst
              | None ->
                  Common.err
                    None
                    "not enough context to instantiate parameter %d of the \
                    function with type '%a'; please supply type parameters \
                    explicitly"
                    (i + 1)
                    sprintf_ltype lty
          in
          let substs = Array.mapi get_subst substs in
          begin
            match beta_reduce (Semant.lval_base_id lval) lty substs with
                LTYPE_mono ty -> yield_ty ty
              | _ ->
                  Common.bug ()
                    "internal_check_outer_lval: beta reduction didn't yield \
                      a monotype"
          end
      | TYPAT_wild, (LTYPE_poly _ as lty) ->
          Common.err
            None
            "not enough context to automatically instantiate the polymorphic \
              type '%a'; supply type parameters explicitly"
            sprintf_ltype lty
      | TYPAT_ty expected, (LTYPE_poly _ as lty) ->
          Common.err
            None
            "not enough context to automatically instantiate '%a' to '%s'; \
            please supply type parameters explicitly"
            sprintf_ltype lty
            (pretty_ty_str expected)
      | _, LTYPE_module _ ->
          Common.err None "can't refer to a module as a first-class value"

  and generic_check_lval
      ~mut:(mut:Ast.mutability)
      ~deref:(deref:bool)
      (infer:ty_pat)
      (lval:Ast.lval)
      : Ast.ty =
    (* The lval we got is an impostor (it may contain unresolved TY_nameds).
     * Get the real one. *)
    let lval_id = Semant.lval_base_id lval in
    let lval = Hashtbl.find cx.Semant.ctxt_all_lvals lval_id in
    let _ =
      iflog cx
        (fun _ ->
           log cx "generic_check_lval %a mut=%s deref=%s infer=%s"
             Ast.sprintf_lval lval
             (if mut = Ast.MUT_mutable then "mutable" else "immutable")
             (if deref then "true" else "false")
             (match infer with
                  TYPAT_wild -> "_"
                | TYPAT_ty t -> Fmt.fmt_to_str Ast.fmt_ty t
                | TYPAT_fn _ -> "<fn>" (* FIXME *)))
    in
    let (lval_ty, n_boxes) =
      internal_check_outer_lval ~mut:mut ~deref:deref infer lval
    in
    let _ =
      iflog cx
        (fun _ ->
           log cx "checked lval %a with type %a"
             Ast.sprintf_lval lval
             Ast.sprintf_ty lval_ty)
    in

    if Hashtbl.mem cx.Semant.ctxt_all_lval_types lval_id then
      assert ((Hashtbl.find cx.Semant.ctxt_all_lval_types lval_id) = lval_ty)
    else
      Hashtbl.replace cx.Semant.ctxt_all_lval_types lval_id lval_ty;

    if Hashtbl.mem cx.Semant.ctxt_auto_deref_lval lval_id then begin
      let prev_autoderef =
        Hashtbl.find cx.Semant.ctxt_auto_deref_lval lval_id
      in
      if n_boxes == 0 && prev_autoderef then
        Common.bug () "generic_check_lval: lval was previously marked as \
          deref but isn't any longer";
      if n_boxes > 0 && not prev_autoderef then
        Common.bug () "generic_check_lval: lval wasn't marked as autoderef \
          before, but now it is"
    end;
    if n_boxes > 1 then
      (* TODO: allow more than one level of automatic dereference *)
      Common.unimpl (Some (Semant.lval_base_id lval))
        "sorry, only one level of automatic dereference is \
        implemented; please add explicit dereference operators";
    Hashtbl.replace cx.Semant.ctxt_auto_deref_lval lval_id (n_boxes > 0);

      (* Before demoting the lval to a value, strip off mutability. *)
      fundamental_ty lval_ty

  (* Note that this function should be avoided when possible, because it
   * cannot perform type inference. In general you should prefer
   * [infer_lval]. *)
  and check_lval
      ?mut:(mut=Ast.MUT_immutable)
      ?deref:(deref=false)
      (lval:Ast.lval)
      : Ast.ty =
    generic_check_lval ~mut:mut ~deref:deref TYPAT_wild lval

  and check_atom ?deref:(deref=false) (atom:Ast.atom) : Ast.ty =
    match atom with
        Ast.ATOM_lval lval -> check_lval ~deref:deref lval
      | Ast.ATOM_literal lit_id -> check_literal lit_id.Common.node
      | Ast.ATOM_pexp _ -> Common.bug () "Type.check_atom on ATOM_pexp"
  in

  let infer_slot (ty:Ast.ty) (slot_id:Common.node_id) : unit =
    ignore (internal_check_slot (TYPAT_ty ty) slot_id)
  in

  let infer_lval
      ?mut:(mut=Ast.MUT_immutable)
      (ty:Ast.ty)
      (lval:Ast.lval)
      : unit =
    ignore (generic_check_lval ?mut:mut ~deref:false
              (TYPAT_ty (Ast.TY_mutable ty)) lval)
  in

  (*
   * AST fragment checking
   *)

  let check_binop (binop:Ast.binop) (operand_ty:Ast.ty) =
    match binop with
        Ast.BINOP_eq | Ast.BINOP_ne | Ast.BINOP_lt | Ast.BINOP_le
      | Ast.BINOP_ge | Ast.BINOP_gt ->
          Ast.TY_bool
      | Ast.BINOP_or | Ast.BINOP_and | Ast.BINOP_xor | Ast.BINOP_lsl
      | Ast.BINOP_lsr | Ast.BINOP_asr ->
          demand_integer operand_ty;
          operand_ty
      | Ast.BINOP_add ->
          demand_number_or_str_or_vector operand_ty;
          operand_ty
      | Ast.BINOP_sub | Ast.BINOP_mul | Ast.BINOP_div | Ast.BINOP_mod ->
          demand_number operand_ty;
          operand_ty
      | Ast.BINOP_send ->
          Common.bug () "check_binop: BINOP_send found in expr"
  in

  let check_expr (expr:Ast.expr) : Ast.ty =
    match expr with
        Ast.EXPR_atom atom -> check_atom atom
      | Ast.EXPR_binary (binop, lhs, rhs) ->
          let operand_ty = check_atom ~deref:true lhs in
          demand operand_ty (check_atom ~deref:true rhs);
          check_binop binop operand_ty
      | Ast.EXPR_unary (Ast.UNOP_not, atom) ->
          demand Ast.TY_bool (check_atom ~deref:true atom);
          Ast.TY_bool
      | Ast.EXPR_unary (Ast.UNOP_bitnot, atom)
      | Ast.EXPR_unary (Ast.UNOP_neg, atom) ->
          let ty = check_atom atom in
          demand_integer ty;
          ty
      | Ast.EXPR_unary (Ast.UNOP_cast dst_ty_id, atom) ->
          (* TODO: probably we want to handle more cases here *)
          demand_bool_or_char_or_integer_or_native (check_atom atom);
          let dst_ty =
            Hashtbl.find cx.Semant.ctxt_all_cast_types dst_ty_id.Common.id
          in
          demand_bool_or_char_or_integer_or_native dst_ty;
          dst_ty
  in

  (* Checks a function call pattern, with arguments specified as atoms, and
   * returns the return type. *)
  let check_fn (callee:Ast.lval) (args:Ast.atom array) : Ast.ty =
    let arg_tys = Array.map check_atom args in
    let callee_ty =
      generic_check_lval
        ~mut:Ast.MUT_immutable
        ~deref:false
        (TYPAT_fn arg_tys)
        callee
    in
    demand_fn (Array.map (fun ty -> Some ty) arg_tys) callee_ty
  in

  let rec check_pat (expected:Ast.ty) (pat:Ast.pat) : unit =
    match pat with
        Ast.PAT_lit lit -> demand expected (check_literal lit)
      | Ast.PAT_tag (constr_fn, arg_pats) ->
          let constr_ty = check_lval constr_fn in
          let arg_tys =
            match constr_ty with
                Ast.TY_fn (ty_sig, _) ->
                  demand expected (get_slot_ty ty_sig.Ast.sig_output_slot);
                  Array.map get_slot_ty ty_sig.Ast.sig_input_slots
              | Ast.TY_tag _ ->
                  demand expected constr_ty;
                  [||]
              | _ -> type_error "constructor function" constr_ty
          in
          Common.arr_iter2 check_pat arg_tys arg_pats
      | Ast.PAT_slot (slot, _) -> infer_slot expected slot.Common.id
      | Ast.PAT_wild -> ()
  in

  let check_check_calls (calls:Ast.check_calls) : unit =
    let check_call (callee, args) =
      demand Ast.TY_bool (check_fn callee args)
    in
    Array.iter check_call calls
  in

  (*
   * Statement checking
   *)

  (* Again as above, we explicitly curry [fn_ctx] to avoid threading it
   * through these functions. *)
  let check_block (fn_ctx:fn_ctx) : (Ast.block -> unit) =
    let check_ret (stmt:Ast.stmt) : unit =
      fn_ctx.fnctx_just_saw_ret <-
        match stmt.Common.node with
            Ast.STMT_ret _ | Ast.STMT_be _ | Ast.STMT_fail
          | Ast.STMT_yield -> true
          | _ -> false
    in

    let rec check_block (block:Ast.block) : unit =
      Stack.push block.Common.id fn_ctx.fnctx_blocks;
      Array.iter check_stmt' block.Common.node

    and check_stmt (stmt:Ast.stmt) : unit =
      try
        check_stmt_full stmt
      with
          Common.Semant_err (None, msg) ->
            raise (Common.Semant_err ((Some stmt.Common.id), msg))

    and check_stmt_full (stmt:Ast.stmt) : unit =
      check_ret stmt;
      match stmt.Common.node with
          Ast.STMT_spawn (dst, _, _, callee, args) ->
            infer_lval Ast.TY_task dst;
            demand Ast.TY_nil (check_fn callee args)

        | Ast.STMT_new_rec (dst, fields, Some base) ->
            let ty = check_lval base in
            let ty_rec = demand_rec ty in
            let field_tys = Hashtbl.create (Array.length ty_rec) in
            Array.iter (fun (id, ty) -> Hashtbl.add field_tys id ty) ty_rec;
            let check_field (name, mut, atom) =
              let field_ty =
                try
                  Hashtbl.find field_tys name
                with Not_found ->
                  Common.err None
                    "field '%s' is not one of the base record's fields" name
              in
              demand field_ty (maybe_mutable mut (check_atom atom))
            in
            Array.iter check_field fields;
            infer_lval ty dst

        | Ast.STMT_new_rec (dst, fields, None) ->
            let check_field (name, mut, atom) =
              (name, maybe_mutable mut (check_atom atom))
            in
            let ty = Ast.TY_rec (Array.map check_field fields) in
            infer_lval ty dst

        | Ast.STMT_new_tup (dst, members) ->
            let check_member (mut, atom) =
              maybe_mutable mut (check_atom atom)
            in
            let ty = Ast.TY_tup (Array.map check_member members) in
            infer_lval ty dst

        | Ast.STMT_new_vec (dst, mut, [| |]) ->
            (* no inference allowed here *)
            let lval_ty = check_lval ~mut:Ast.MUT_mutable dst in
            ignore (demand_vec_with_mutability mut lval_ty)

        | Ast.STMT_new_vec (dst, mut, elems) ->
            let atom_ty = demand_all (Array.map check_atom elems) in
            let ty = Ast.TY_vec (maybe_mutable mut atom_ty) in
            infer_lval ty dst

        | Ast.STMT_new_str (dst, _) -> infer_lval Ast.TY_str dst

        | Ast.STMT_new_port dst ->
            (* we can't actually typecheck this *)
            ignore (check_lval dst);
            ()

        | Ast.STMT_new_chan (dst, Some port) ->
            let ty = Ast.TY_chan (demand_port (check_lval port)) in
            infer_lval ty dst

        | Ast.STMT_new_chan (dst, None) ->
            (* can't check this either *)
            ignore (check_lval dst);
            ()

        | Ast.STMT_new_box (dst, mut, src) ->
            let ty = Ast.TY_box (maybe_mutable mut (check_atom src)) in
            infer_lval ty dst

        | Ast.STMT_copy (dst, src) ->
            infer_lval (check_expr src) dst

        | Ast.STMT_copy_binop (dst, Ast.BINOP_add, src) ->
            begin
            let src_ty = check_atom ~deref:true src in
            let dst_ty = check_lval dst in
              match fundamental_ty dst_ty, fundamental_ty src_ty with
                  Ast.TY_vec elt1, Ast.TY_vec elt2 ->
                    if elt1 = elt2
                    then ()
                    else
                      Common.err None
                        "mismatched types in vec-append: %s += %s"
                        (pretty_ty_str dst_ty)
                        (pretty_ty_str src_ty)
                | Ast.TY_str, Ast.TY_str -> ()
                | _ ->
                    infer_lval src_ty dst;
                    demand src_ty (check_binop Ast.BINOP_add src_ty)
            end

        | Ast.STMT_copy_binop (dst, binop, src) ->
            let ty = check_atom ~deref:true src in
            infer_lval ty dst;
            demand ty (check_binop binop ty)

        | Ast.STMT_call (dst, callee, args) ->
            infer_lval (check_fn callee args) dst

        | Ast.STMT_bind (bound, callee, args) ->
            let check_arg arg_opt =
              match arg_opt with
                  None -> None
                | Some arg -> Some (check_atom arg)
            in
            let rec replace_args ty =
              match ty with
                  Ast.TY_fn (ty_sig, ty_fn_aux) ->
                    let orig_slots = ty_sig.Ast.sig_input_slots in
                    let take_arg i =
                      match args.(i) with
                          None -> Some orig_slots.(i)
                        | Some _ -> None
                    in
                    let new_slots = Array.init (Array.length args) take_arg in
                    let new_slots = Common.arr_filter_some new_slots in
                    let ty_sig =
                      { ty_sig with Ast.sig_input_slots = new_slots }
                    in
                    Ast.TY_fn (ty_sig, ty_fn_aux)
                | Ast.TY_mutable ty' -> Ast.TY_mutable (replace_args ty')
                | Ast.TY_constrained (ty', constrs) ->
                    Ast.TY_constrained (replace_args ty', constrs)
                | _ -> Common.bug () "replace_args: unexpected type"
            in
            let callee_ty = check_lval callee in
            ignore (demand_fn (Array.map check_arg args) callee_ty);
            infer_lval (replace_args callee_ty) bound

        | Ast.STMT_recv (dst, src) ->
            infer_lval (demand_port (check_lval src)) dst

        | Ast.STMT_slice (dst, src, slice) ->
            let src_ty = check_lval src in
            ignore (demand_vec src_ty);
            infer_lval src_ty dst;
            let check_index index = demand Ast.TY_int (check_atom index) in
            Common.may check_index slice.Ast.slice_start;
            Common.may check_index slice.Ast.slice_len

        | Ast.STMT_while w | Ast.STMT_do_while w ->
            let (stmts, expr) = w.Ast.while_lval in
            Array.iter check_stmt' stmts;
            demand Ast.TY_bool (check_expr expr);
            check_block w.Ast.while_body

        | Ast.STMT_for sf ->
            let elem_ty = demand_vec_or_str (check_lval sf.Ast.for_seq) in
            infer_slot elem_ty (fst sf.Ast.for_slot).Common.id;
            check_block sf.Ast.for_body

        | Ast.STMT_for_each sfe ->
            let (callee, args) = sfe.Ast.for_each_call in
            let elem_ty = check_fn callee args in
            infer_slot elem_ty (fst (sfe.Ast.for_each_slot)).Common.id;
            check_block sfe.Ast.for_each_head;
            check_block sfe.Ast.for_each_body

        | Ast.STMT_if si ->
            demand Ast.TY_bool (check_expr si.Ast.if_test);
            check_block si.Ast.if_then;
            Common.may check_block si.Ast.if_else

        | Ast.STMT_put _ when not fn_ctx.fnctx_is_iter ->
            Common.err None "'put' may only be used in an iterator function"

        | Ast.STMT_put (Some atom) ->
            demand fn_ctx.fnctx_return_type (check_atom atom)

        | Ast.STMT_put None -> () (* always well-typed *)

        | Ast.STMT_put_each (callee, args) -> ignore (check_fn callee args)

        | Ast.STMT_ret (Some atom) ->
            if fn_ctx.fnctx_is_iter then
              Common.err None
                "iterators can't return values; did you mean 'put'?";
            demand fn_ctx.fnctx_return_type (check_atom atom)

        | Ast.STMT_ret None ->
            if not fn_ctx.fnctx_is_iter then
              demand Ast.TY_nil fn_ctx.fnctx_return_type

        | Ast.STMT_be (callee, args) ->
            demand fn_ctx.fnctx_return_type (check_fn callee args)

        | Ast.STMT_alt_tag alt_tag ->
            let get_pat arm = fst arm.Common.node in
            let pats = Array.map get_pat alt_tag.Ast.alt_tag_arms in
            let ty = check_lval alt_tag.Ast.alt_tag_lval in
            let get_block arm = snd arm.Common.node in
            let blocks = Array.map get_block alt_tag.Ast.alt_tag_arms in
            Array.iter (check_pat ty) pats;
            Array.iter check_block blocks

        | Ast.STMT_alt_type _ -> () (* TODO *)

        | Ast.STMT_alt_port _ -> () (* TODO *)

        (* always well-typed *)
        | Ast.STMT_fail | Ast.STMT_yield 
        | Ast.STMT_break | Ast.STMT_cont -> ()

        | Ast.STMT_join lval -> infer_lval Ast.TY_task lval

        | Ast.STMT_send (chan, value) ->
            let value_ty = demand_chan (check_lval chan) in
            infer_lval ~mut:Ast.MUT_immutable value_ty value

        | Ast.STMT_log x | Ast.STMT_note x | Ast.STMT_log_err x ->
            (* always well-typed, just record type in passing. *)
            ignore (check_atom x)

        | Ast.STMT_prove _ ->
            (* always well-typed, static. Ignore. *)
            ()

        | Ast.STMT_check (_, calls) -> check_check_calls calls

        | Ast.STMT_check_expr expr -> demand Ast.TY_bool (check_expr expr)

        | Ast.STMT_check_if (_, calls, block) ->
            check_check_calls calls;
            check_block block

        | Ast.STMT_block block -> check_block block

        | Ast.STMT_decl (Ast.DECL_slot (slot_key, _)) ->
            Hashtbl.add fn_ctx.fnctx_slot_decls slot_key stmt.Common.id

        | Ast.STMT_decl (Ast.DECL_mod_item _) -> () (* always well-typed *)

    and check_stmt' stmt =
      try
        check_stmt stmt
      with Type_error (expected, actual) ->
        Common.err
          (Some stmt.Common.id)
          "mismatched types: expected %s but found %s"
          expected
          actual
    in
    check_block
  in
  check_block

let create_tag_graph_nodes (cx:Semant.ctxt) =
  let make_graph_node id _ =
    Hashtbl.add cx.Semant.ctxt_tag_containment id {
      Semant.tgn_index = None;
      Semant.tgn_children = Queue.create ()
    }
  in
  Hashtbl.iter make_graph_node cx.Semant.ctxt_all_tag_info

let populate_tag_graph_node (cx:Semant.ctxt) (id:Common.opaque_id) (n:int) =
  let tag_info = Hashtbl.find cx.Semant.ctxt_all_tag_info id in
  let (_, _, ty_tup) = Hashtbl.find tag_info.Semant.tag_nums n in
  let rec add_ty =
    function
        Ast.TY_tag { Ast.tag_id = id'; Ast.tag_args = tys } ->
          let tag_graph_node =
            Hashtbl.find cx.Semant.ctxt_tag_containment id
          in
          Queue.add id' tag_graph_node.Semant.tgn_children;
          Array.iter add_ty tys
      | Ast.TY_tup tys -> Array.iter add_ty tys
      | Ast.TY_rec ty_rec ->
          Array.iter (fun (_, ty) -> add_ty ty) ty_rec
      | Ast.TY_fn ty_fn -> add_ty_fn ty_fn
      | Ast.TY_chan ty | Ast.TY_port ty | Ast.TY_mutable ty
          | Ast.TY_constrained (ty, _) -> add_ty ty
      | Ast.TY_obj (_, ty_fns) ->
          Hashtbl.iter (fun _ ty_fn -> add_ty_fn ty_fn) ty_fns
      | _ -> ()
  and add_ty_fn (ty_sig, _) =
    let add_slot slot =
      match slot.Ast.slot_ty with
          None -> ()
        | Some ty -> add_ty ty
    in
    Array.iter add_slot ty_sig.Ast.sig_input_slots;
    add_slot ty_sig.Ast.sig_output_slot
  in
  Array.iter add_ty ty_tup

let stack_contains (stack:'a Stack.t) (elem:'a) : bool =
  try
    Stack.iter (fun elem' -> if elem = elem' then raise Exit) stack; false
  with Exit -> true

let report_infinitely_sized_tag
    (id:Common.opaque_id)
    (stack:Common.opaque_id Stack.t)
    : unit =
  let string_of_tag_id tag_id =
    let ty = Ast.TY_tag { Ast.tag_id = tag_id; Ast.tag_args = [| |] } in
    Ast.sprintf_ty () ty
  in
  let msg = Buffer.create 0 in
  Buffer.add_string msg "found tag of infinite size: ";
  while not (Stack.is_empty stack) do
    Buffer.add_string msg (string_of_tag_id (Stack.pop stack));
    Buffer.add_string msg " <- "
  done;
  Buffer.add_string msg (string_of_tag_id id);
  Buffer.add_string msg "; use '@' for recursive references";
  Common.err None "%s" (Buffer.contents msg)

let check_for_tag_cycles (cx:Semant.ctxt) =
  (* Find cycles in tags using Tarjan's strongly connected components
   * algorithm. *)
  let lowlinks = Hashtbl.create 0 in
  let next_index, stack = ref 0, Stack.create () in

  let rec check_node id node =
    if node.Semant.tgn_index = None then begin
      let index = !next_index in
      incr next_index;
      node.Semant.tgn_index <- Some index;

      Stack.push id stack;

      Hashtbl.add lowlinks id max_int;

      let check_outgoing_edge id' =
        let node' = Hashtbl.find cx.Semant.ctxt_tag_containment id' in
        if node'.Semant.tgn_index = None then begin
          check_node id' node';
          let lowlink = Hashtbl.find lowlinks id in
          let lowlink' = Hashtbl.find lowlinks id' in
          Hashtbl.replace lowlinks id (min lowlink lowlink')
        end else if stack_contains stack id' then
          let lowlink = Hashtbl.find lowlinks id in
          let index' =
            match node'.Semant.tgn_index with
                Some index' -> index'
              | None ->
                  Common.bug
                    ()
                    "check_for_tag_cycles: node in stack without index"
          in
          Hashtbl.replace lowlinks id (min lowlink index')
      in

      Queue.iter check_outgoing_edge node.Semant.tgn_children;

      if index == Hashtbl.find lowlinks id then
        report_infinitely_sized_tag id stack;

      ignore (Stack.pop stack)
    end
  in
  Hashtbl.iter check_node cx.Semant.ctxt_tag_containment

let process_crate (cx:Semant.ctxt) (crate:Ast.crate) : unit =
  (* Verify that, if main is present, it has the right form. *)
  let verify_main (item_id:Common.node_id) : unit =
    let path_name = Hashtbl.find cx.Semant.ctxt_all_item_names item_id in
    if cx.Semant.ctxt_main_name = Some path_name then
      try
        match Hashtbl.find cx.Semant.ctxt_all_item_types item_id with
            Ast.TY_fn (tsig, _) ->
              begin
                match tsig.Ast.sig_input_slots with
                    [| |]
                  | [| {
                         Ast.slot_mode = Ast.MODE_local;
                         Ast.slot_ty = Some (Ast.TY_vec Ast.TY_str)
                       } |] -> ()
                  | _ -> Common.err (Some item_id)
                      "main fn has bad type signature"
              end
          | _ -> Common.err (Some item_id) "main item has bad non-fn type"
      with Not_found ->
        Common.err (Some item_id) "main item has no type (is it a function?)"
  in

  let visitor (cx:Semant.ctxt) (inner:Walk.visitor) : Walk.visitor =
    let create_fn_ctx (ret_ty:Ast.ty) (is_iter:bool) =
      {
        fnctx_return_type = ret_ty;
        fnctx_is_iter = is_iter;
        fnctx_just_saw_ret = false;
        fnctx_blocks = Stack.create ();
        fnctx_slot_decls = Hashtbl.create 0;
      }
    in

    let create_fn_ctx_of_ty_fn (ty_fn:Ast.ty_fn) : fn_ctx =
      let (ty_sig, ty_fn_aux) = ty_fn in
      let ret_ty = ty_sig.Ast.sig_output_slot.Ast.slot_ty in
      let is_iter = ty_fn_aux.Ast.fn_is_iter in
      create_fn_ctx (Common.option_get ret_ty) is_iter
    in

    let finish_function (fn_ctx:fn_ctx) (item_id:Common.node_id option) =
      if not fn_ctx.fnctx_just_saw_ret &&
          fn_ctx.fnctx_return_type <> Ast.TY_nil &&
          not fn_ctx.fnctx_is_iter then
        Common.err item_id "this function must return a value";

      let check_for_slot_types_in_block block_id =
        let check_for_slot_type slot_key defn_id =
          match Hashtbl.find cx.Semant.ctxt_all_defns defn_id with
              Semant.DEFN_slot { Ast.slot_ty = None; Ast.slot_mode = _ } ->
                let stmt_id = Hashtbl.find fn_ctx.fnctx_slot_decls slot_key in
                Common.err
                  (Some stmt_id)
                  "no type could be inferred for this slot"
            | _ -> ()
        in
        let block_slots = Hashtbl.find cx.Semant.ctxt_block_slots block_id in
        Hashtbl.iter check_for_slot_type block_slots
      in
      Stack.iter check_for_slot_types_in_block fn_ctx.fnctx_blocks
    in

    let check_fn_ty_validity item_id (ty_sig, _) =
      let check_input_slot i slot =
        match slot with
            {
              Ast.slot_ty = Some (Ast.TY_param _);
              Ast.slot_mode = Ast.MODE_local
            } ->
              Common.err
                (Some item_id)
                "parameter %d of this type-parametric function must be \
                passed by reference, not by value"
                (i + 1)
          | _ -> ()
      in
      Array.iteri check_input_slot ty_sig.Ast.sig_input_slots
    in

    let visit_mod_item_pre _ _ item =
      let { Common.node = item; Common.id = item_id } = item in
      match item.Ast.decl_item with
          Ast.MOD_ITEM_fn fn when
              not (Hashtbl.mem cx.Semant.ctxt_required_items item_id) ->
            let fn_ty = Hashtbl.find cx.Semant.ctxt_all_item_types item_id in
            begin
              match fn_ty with
                  Ast.TY_fn ty_fn ->
                    check_fn_ty_validity item_id ty_fn;
                    let fn_ctx = create_fn_ctx_of_ty_fn ty_fn in
                    check_block cx fn_ctx fn.Ast.fn_body;
                    finish_function fn_ctx (Some item_id)
                | _ ->
                  Common.bug ()
                    "Type.visit_mod_item_pre: fn item didn't have a fn type"
            end
        | _ -> ()
    in
    let visit_mod_item_post _ _ item =
      let item_id = item.Common.id in
      verify_main item_id;
      match item.Common.node.Ast.decl_item with
          Ast.MOD_ITEM_tag (_, id, n) -> populate_tag_graph_node cx id n
        | _ -> ()
    in

    let visit_obj_fn_pre obj ident _ =
      let obj_ty = Hashtbl.find cx.Semant.ctxt_all_item_types obj.Common.id in
      let bad _ =
        Common.bug ()
          "Type.visit_obj_fn_pre: item doesn't have an object type (%a)"
          Ast.sprintf_ty obj_ty
      in
        match obj_ty with
            Ast.TY_fn (tsig, _) ->
              begin
                match tsig.Ast.sig_output_slot with
                    { Ast.slot_ty = Some (Ast.TY_obj (_, methods));
                      Ast.slot_mode = _ } ->
                      let fn_ty = Hashtbl.find methods ident in
                      let fn_ctx = create_fn_ctx_of_ty_fn fn_ty in
                      let obj_fns = obj.Common.node.Ast.obj_fns in
                      let fn = Hashtbl.find obj_fns ident in
                      check_block cx fn_ctx fn.Common.node.Ast.fn_body;
                      finish_function fn_ctx (Some fn.Common.id)
                  | _ -> bad()
              end
          | _ -> bad()
    in

    let visit_obj_drop_pre _ block =
      let fn_ctx = create_fn_ctx Ast.TY_nil false in
      check_block cx fn_ctx block;
      finish_function fn_ctx None
    in

    let visit_crate_pre _ : unit = create_tag_graph_nodes cx in

    let visit_crate_post _ : unit =
      (* Fill in the autoderef info for any lvals we didn't get to. *)
      let fill lval_id _ =
        if not (Hashtbl.mem cx.Semant.ctxt_auto_deref_lval lval_id) then
          Hashtbl.add cx.Semant.ctxt_auto_deref_lval lval_id false
      in
      Hashtbl.iter fill cx.Semant.ctxt_all_lvals;

      (* Check for tag cycles. *)
      check_for_tag_cycles cx
    in

    {
      inner with
        Walk.visit_mod_item_pre = visit_mod_item_pre;
        Walk.visit_mod_item_post = visit_mod_item_post;
        Walk.visit_obj_fn_pre = visit_obj_fn_pre;
        Walk.visit_obj_drop_pre = visit_obj_drop_pre;
        Walk.visit_crate_pre = visit_crate_pre;
        Walk.visit_crate_post = visit_crate_post
    }
  in

  let passes =
    [|
      (visitor cx Walk.empty_visitor)
    |]
  in
  let log_flag = cx.Semant.ctxt_sess.Session.sess_log_type in
    Semant.run_passes cx "type" passes log_flag log crate
;;

(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)

