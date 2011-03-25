open Semant;;
open Common;;

(*
 * Resolution passes:
 *
 *   - build multiple 'scope' hashtables mapping slot_key -> node_id
 *   - build single 'type inference' hashtable mapping node_id -> slot
 *
 *   (note: not every slot is identified; only those that are declared
 *    in statements and/or can participate in local type inference.
 *    Those in function signatures are not, f.e. Also no type values
 *    are identified, though module items are. )
 *
 *)

exception Resolution_failure of (Ast.name * Ast.name) list

let log cx = Session.log "resolve"
  (should_log cx cx.ctxt_sess.Session.sess_log_resolve)
  cx.ctxt_sess.Session.sess_log_out
;;

let iflog cx thunk =
  if (should_log cx cx.ctxt_sess.Session.sess_log_resolve)
  then thunk ()
  else ()
;;


let block_scope_forming_visitor
    (cx:ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =
  let visit_block_pre b =
    if not (Hashtbl.mem cx.ctxt_block_items b.id)
    then htab_put cx.ctxt_block_items b.id (Hashtbl.create 0);
    if not (Hashtbl.mem cx.ctxt_block_slots b.id)
    then htab_put cx.ctxt_block_slots b.id (Hashtbl.create 0);
    inner.Walk.visit_block_pre b
  in
    { inner with Walk.visit_block_pre = visit_block_pre }
;;


let stmt_collecting_visitor
    (cx:ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =
  let block_ids = Stack.create () in
  let visit_block_pre (b:Ast.block) =
    htab_put cx.ctxt_all_blocks b.id b.node;
    Stack.push b.id block_ids;
    inner.Walk.visit_block_pre b
  in
  let visit_block_post (b:Ast.block) =
    inner.Walk.visit_block_post b;
    ignore (Stack.pop block_ids)
  in

  let visit_for_block
      ((si:Ast.slot identified),(ident:Ast.ident))
      (block_id:node_id)
      : unit =
    let slots = Hashtbl.find cx.ctxt_block_slots block_id in
    let key = Ast.KEY_ident ident in
      log cx "found decl of '%s' in for-loop block header" ident;
      htab_put slots key si.id;
      htab_put cx.ctxt_slot_keys si.id key
  in

  let visit_stmt_pre stmt =
    begin
      htab_put cx.ctxt_all_stmts stmt.id stmt;
      match stmt.node with
          Ast.STMT_decl d ->
            begin
              let bid = Stack.top block_ids in
              let items = Hashtbl.find cx.ctxt_block_items bid in
              let slots = Hashtbl.find cx.ctxt_block_slots bid in
              let check_and_log_ident id ident =
                if Hashtbl.mem items ident ||
                  Hashtbl.mem slots (Ast.KEY_ident ident)
                then
                  err (Some id)
                    "duplicate declaration '%s' in block" ident
                else
                  log cx "found decl of '%s' in block" ident
              in
              let check_and_log_tmp id tmp =
                if Hashtbl.mem slots (Ast.KEY_temp tmp)
                then
                  err (Some id)
                    "duplicate declaration of temp #%d in block"
                    (int_of_temp tmp)
                else
                  log cx "found decl of temp #%d in block" (int_of_temp tmp)
              in
              let check_and_log_key id key =
                match key with
                    Ast.KEY_ident i -> check_and_log_ident id i
                  | Ast.KEY_temp t -> check_and_log_tmp id t
              in
                match d with
                    Ast.DECL_mod_item (ident, item) ->
                      check_and_log_ident item.id ident;
                      htab_put items ident item.id
                  | Ast.DECL_slot (key, sid) ->
                      check_and_log_key sid.id key;
                      htab_put slots key sid.id;
                      htab_put cx.ctxt_slot_keys sid.id key
            end
        | Ast.STMT_for f ->
            visit_for_block f.Ast.for_slot f.Ast.for_body.id
        | Ast.STMT_for_each f ->
            visit_for_block f.Ast.for_each_slot f.Ast.for_each_head.id
        | Ast.STMT_alt_tag { Ast.alt_tag_arms = arms;
                             Ast.alt_tag_lval = _ } ->
            let rec resolve_pat block pat =
              match pat with
                  Ast.PAT_slot ({ id = slot_id; node = _ }, ident) ->
                    let slots = Hashtbl.find cx.ctxt_block_slots block.id in
                    let key = Ast.KEY_ident ident in
                    htab_put slots key slot_id;
                    htab_put cx.ctxt_slot_keys slot_id key
                | Ast.PAT_tag (_, pats) -> Array.iter (resolve_pat block) pats
                | Ast.PAT_lit _
                | Ast.PAT_wild -> ()
            in
              Array.iter (fun { node = (p, b); id = _ } ->
                            resolve_pat b p) arms
        | _ -> ()
    end;
    inner.Walk.visit_stmt_pre stmt
  in
    { inner with
        Walk.visit_block_pre = visit_block_pre;
        Walk.visit_block_post = visit_block_post;
        Walk.visit_stmt_pre = visit_stmt_pre }
;;


let all_item_collecting_visitor
    (cx:ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =

  let items = Stack.create () in

  let push_on_item_arg_list item_id arg_id =
    let existing =
      match htab_search cx.ctxt_frame_args item_id with
          None -> []
        | Some x -> x
    in
      htab_put cx.ctxt_slot_is_arg arg_id ();
      Hashtbl.replace cx.ctxt_frame_args item_id (arg_id :: existing)
  in

  let note_header item_id header =
    Array.iter
      (fun (sloti,ident) ->
         let key = Ast.KEY_ident ident in
           htab_put cx.ctxt_slot_keys sloti.id key;
           push_on_item_arg_list item_id sloti.id)
      header;
  in

  let visit_mod_item_pre n p i =
    Stack.push i.id items;
    Array.iter (fun p -> htab_put cx.ctxt_all_defns p.id
                  (DEFN_ty_param p.node)) p;
    htab_put cx.ctxt_all_defns i.id (DEFN_item i.node);
    htab_put cx.ctxt_all_item_names i.id (path_to_name cx.ctxt_curr_path);
    log cx "collected item #%d: %s" (int_of_node i.id) n;
    begin
      match i.node.Ast.decl_item with
          Ast.MOD_ITEM_fn f ->
            note_header i.id f.Ast.fn_input_slots;
        | Ast.MOD_ITEM_obj ob ->
            note_header i.id ob.Ast.obj_state;
        | Ast.MOD_ITEM_tag (hdr, _, _) ->
            note_header i.id hdr
        | Ast.MOD_ITEM_type (_, Ast.TY_tag ttag) ->
            Hashtbl.replace cx.ctxt_user_tag_names ttag.Ast.tag_id
              (path_to_name cx.ctxt_curr_path)
        | _ -> ()
    end;
      inner.Walk.visit_mod_item_pre n p i
  in

  let visit_mod_item_post n p i =
    inner.Walk.visit_mod_item_post n p i;
    ignore (Stack.pop items)
  in

  let visit_obj_fn_pre obj ident fn =
    htab_put cx.ctxt_all_defns fn.id (DEFN_obj_fn (obj.id, fn.node));
    htab_put cx.ctxt_all_item_names fn.id (path_to_name cx.ctxt_curr_path);
    note_header fn.id fn.node.Ast.fn_input_slots;
    inner.Walk.visit_obj_fn_pre obj ident fn
  in

  let visit_obj_drop_pre obj b =
    htab_put cx.ctxt_all_defns b.id (DEFN_obj_drop obj.id);
    htab_put cx.ctxt_all_item_names b.id (path_to_name cx.ctxt_curr_path);
    inner.Walk.visit_obj_drop_pre obj b
  in

  let visit_stmt_pre s =
    begin
      match s.node with
          Ast.STMT_for_each fe ->
            let id = fe.Ast.for_each_body.id in
              htab_put cx.ctxt_all_defns id
                (DEFN_loop_body (Stack.top items));
              htab_put cx.ctxt_all_item_names id
                (path_to_name cx.ctxt_curr_path);
        | _ -> ()
    end;
    inner.Walk.visit_stmt_pre s;
  in

    { inner with
        Walk.visit_mod_item_pre = visit_mod_item_pre;
        Walk.visit_mod_item_post = visit_mod_item_post;
        Walk.visit_obj_fn_pre = visit_obj_fn_pre;
        Walk.visit_obj_drop_pre = visit_obj_drop_pre;
        Walk.visit_stmt_pre = visit_stmt_pre; }
;;

let lookup_type_node_by_name
    (cx:ctxt)
    (scopes:scope list)
    (name:Ast.name)
    : node_id =
  iflog cx (fun _ ->
              log cx "lookup_simple_type_by_name %a"
                Ast.sprintf_name name);
  match lookup_by_name cx [] scopes name with
      RES_failed name' -> raise (Resolution_failure [ name', name ])
    | RES_ok (_, id) ->
        match htab_search cx.ctxt_all_defns id with
            Some (DEFN_item { Ast.decl_item = Ast.MOD_ITEM_type _;
                              Ast.decl_params = _ })
          | Some (DEFN_item { Ast.decl_item = Ast.MOD_ITEM_obj _;
                              Ast.decl_params = _ })
          | Some (DEFN_ty_param _) -> id
          | _ ->
              err None "Found non-type binding for %a"
                Ast.sprintf_name name
;;

type recur_info =
    { recur_all_nodes: node_id list }
;;

let empty_recur_info =
  { recur_all_nodes = []; }
;;

let push_node r n =
  { recur_all_nodes = n :: r.recur_all_nodes }


let report_resolution_failure type_names =
  let rec recur type_names str =
    let stringify_pair (part, whole) =
      if part = whole then
        Printf.sprintf "'%a'" Ast.sprintf_name part
      else
        Printf.sprintf "'%a' in name '%a'" Ast.sprintf_name part
          Ast.sprintf_name whole
    in
    match type_names with
        [] -> bug () "no name in resolution failure"
      | [ pair ] -> err None "unbound name %s%s" (stringify_pair pair) str
      | pair::pairs ->
          recur pairs
            (Printf.sprintf " while resolving %s" (stringify_pair pair))
  in 
  recur type_names "" 

let rec lookup_type_by_name
    ?loc:loc
    (cx:ctxt)
    (scopes:scope list)
    (recur:recur_info)
    (name:Ast.name)
    : ((scope list) * node_id * Ast.ty) =
  iflog cx (fun _ ->
              log cx "+++ lookup_type_by_name %a"
                Ast.sprintf_name name);
  match lookup_by_name ?loc:loc cx [] scopes name with
      RES_failed name' -> raise (Resolution_failure [ name', name ])
    | RES_ok (scopes', id) ->
        let ty, params =
          match htab_search cx.ctxt_all_defns id with
              Some (DEFN_item { Ast.decl_item = Ast.MOD_ITEM_type (_, t);
                                Ast.decl_params = params }) ->
                (t, Array.map (fun p -> p.node) params)
            | Some (DEFN_item { Ast.decl_item = Ast.MOD_ITEM_obj ob;
                                Ast.decl_params = params }) ->
                (Ast.TY_obj (ty_obj_of_obj ob),
                 Array.map (fun p -> p.node) params)
            | Some (DEFN_ty_param (_, x)) ->
                (Ast.TY_param x, [||])
            | _ ->
                err loc "Found non-type binding for %a"
                  Ast.sprintf_name name
        in
        let args =
          match name with
              Ast.NAME_ext (_, Ast.COMP_app (_, args)) -> args
            | Ast.NAME_base (Ast.BASE_app (_, args)) -> args
            | _ -> [| |]
        in
        let args =
          iflog cx (fun _ -> log cx
                      "lookup_type_by_name %a resolving %d type args"
                      Ast.sprintf_name name
                      (Array.length args));
          Array.mapi
            begin
              fun i t ->
                let t =
                  resolve_type ?loc:loc cx scopes recur t
                in
                  iflog cx (fun _ -> log cx
                              "lookup_type_by_name resolved arg %d to %a" i
                              Ast.sprintf_ty t);
                  t
            end
            args
        in
          iflog cx
            begin
              fun _ ->
                log cx
                  "lookup_type_by_name %a found ty %a"
                  Ast.sprintf_name name Ast.sprintf_ty ty;
                log cx "applying %d type args to %d params"
                  (Array.length args) (Array.length params);
                log cx "params: %s"
                  (Fmt.fmt_to_str Ast.fmt_decl_params params);
                log cx "args: %s"
                  (Fmt.fmt_to_str Ast.fmt_app_args args);
            end;
          let ty =
            rebuild_ty_under_params ?node_id:loc cx None ty params args true
          in
            iflog cx (fun _ -> log cx "--- lookup_type_by_name %a ==> %a"
                        Ast.sprintf_name name
                        Ast.sprintf_ty ty);
            (scopes', id, ty)

and resolve_type
    ?loc:loc
    (cx:ctxt)
    (scopes:(scope list))
    (recur:recur_info)
    (t:Ast.ty)
    : Ast.ty =
  let _ = iflog cx (fun _ -> log cx "+++ resolve_type %a" Ast.sprintf_ty t) in
  let base = ty_fold_rebuild (fun t -> t) in
  let ty_fold_named name =
    let (scopes, node, t) =
      lookup_type_by_name ?loc:loc cx scopes recur name
    in
      iflog cx (fun _ ->
                  log cx "resolved type name '%a' to item %d with ty %a"
                  Ast.sprintf_name name (int_of_node node)
                  Ast.sprintf_ty t);
      if List.mem node recur.recur_all_nodes
      then (err (Some node) "infinite recursive type definition: '%a'"
              Ast.sprintf_name name)
      else
        let recur = push_node recur node in
          iflog cx (fun _ -> log cx "recursively resolving type %a"
                      Ast.sprintf_ty t);
          try
            resolve_type ?loc:loc cx scopes recur t
          with Resolution_failure names ->
            raise (Resolution_failure ((name, name)::names))
  in
  let fold =
    { base with
        ty_fold_named = ty_fold_named; }
  in
  let t' = fold_ty cx fold t in
    iflog cx (fun _ ->
                log cx "--- resolve_type %a ==> %a"
                  Ast.sprintf_ty t Ast.sprintf_ty t');
    t'
;;


let type_resolving_visitor
    (cx:ctxt)
    (scopes:(scope list) ref)
    (inner:Walk.visitor)
    : Walk.visitor =

  let tinfos = Hashtbl.create 0 in

  let resolve_ty ?(loc=id_of_scope (List.hd (!scopes))) (t:Ast.ty) : Ast.ty =
    try
      resolve_type ~loc:loc cx (!scopes) empty_recur_info t
    with Resolution_failure pairs ->
      report_resolution_failure pairs
  in

  let resolve_slot (s:Ast.slot) : Ast.slot =
    match s.Ast.slot_ty with
        None -> s
      | Some ty -> { s with Ast.slot_ty = Some (resolve_ty ty) }
  in

  let resolve_slot_identified
      (s:Ast.slot identified)
      : (Ast.slot identified) =
    try
      let slot = resolve_slot s.node in
        { s with node = slot }
    with
        Semant_err (None, e) -> raise (Semant_err ((Some s.id), e))
  in

  let visit_slot_identified_pre slot =
    let slot = resolve_slot_identified slot in
      htab_put cx.ctxt_all_defns slot.id (DEFN_slot slot.node);
      iflog cx
        (fun _ ->
           log cx "collected resolved slot #%d with type %s"
             (int_of_node slot.id)
             (match slot.node.Ast.slot_ty with
                  None -> "??"
                | Some t -> (Fmt.fmt_to_str Ast.fmt_ty t)));
      inner.Walk.visit_slot_identified_pre slot
  in

  let visit_mod_item_pre id params item =
    let resolve_and_store_type _ =
      let t = ty_of_mod_item item in
      let ty = resolve_ty ~loc:item.id t in
        iflog cx
          (fun _ ->
             log cx "resolved item %s, type as %a" id Ast.sprintf_ty ty);
        htab_put cx.ctxt_all_item_types item.id ty;
    in
    begin
      try
        match item.node.Ast.decl_item with
            Ast.MOD_ITEM_type (_, ty) ->
              let ty = resolve_ty ~loc:item.id ty in
                iflog cx
                  (fun _ ->
                     log cx "resolved item %s, defining type %a"
                       id Ast.sprintf_ty ty);
                htab_put cx.ctxt_all_type_items item.id ty;
                htab_put cx.ctxt_all_item_types item.id Ast.TY_type;
                if Hashtbl.mem cx.ctxt_all_item_names item.id then
                  Hashtbl.add cx.ctxt_user_type_names ty
                    (Hashtbl.find cx.ctxt_all_item_names item.id)

          (* 
           * Don't resolve the "type" of a mod item; just resolve its
           * members.
           *)
          | Ast.MOD_ITEM_mod _ -> ()

          | Ast.MOD_ITEM_tag (slots, oid, n) ->
              resolve_and_store_type ();
              let tinfo =
                htab_search_or_add
                  tinfos oid
                  (fun _ ->
                     { tag_idents = Hashtbl.create 0;
                       tag_nums = Hashtbl.create 0; } )
              in
              let ttup =
                Array.map
                  (fun (s,_) -> (slot_ty (resolve_slot_identified s).node))
                  slots
              in
                if not (Hashtbl.mem tinfo.tag_idents id)
                then
                  begin
                    htab_put tinfo.tag_idents id (n, item.id, ttup);
                    htab_put tinfo.tag_nums n (id, item.id, ttup);
                  end

          | _ -> resolve_and_store_type ()
      with
          Semant_err (None, e) -> raise (Semant_err ((Some item.id), e))
    end;
    inner.Walk.visit_mod_item_pre id params item
  in

  let visit_obj_fn_pre obj ident fn =
    let fty = resolve_ty ~loc:fn.id (Ast.TY_fn (ty_fn_of_fn fn.node)) in
      iflog cx
        (fun _ ->
           log cx "resolved obj fn %s as %a" ident Ast.sprintf_ty fty);
      htab_put cx.ctxt_all_item_types fn.id fty;
      inner.Walk.visit_obj_fn_pre obj ident fn
  in

  let visit_obj_drop_pre obj b =
    let fty = mk_simple_ty_fn [| |] in
      htab_put cx.ctxt_all_item_types b.id fty;
      inner.Walk.visit_obj_drop_pre obj b
  in

  let visit_stmt_pre stmt =
    begin
      match stmt.node with
          Ast.STMT_for_each fe ->
            let id = fe.Ast.for_each_body.id in
            let fty = mk_simple_ty_iter [| |] in
              htab_put cx.ctxt_all_item_types id fty;
        | Ast.STMT_copy (_, Ast.EXPR_unary (Ast.UNOP_cast t, _)) ->
            let ty = resolve_ty t.node in
              htab_put cx.ctxt_all_cast_types t.id ty
        | _ -> ()
    end;
    inner.Walk.visit_stmt_pre stmt
  in

  let rebuilt_pexps = Hashtbl.create 0 in
  let get_rebuilt_pexp p =
    Hashtbl.find rebuilt_pexps p.id
  in

  let visit_pexp_post p =
    inner.Walk.visit_pexp_post p;
    let rebuild_plval pl =
      match pl with
          Ast.PLVAL_base (Ast.BASE_app (id, tys)) ->
            Ast.PLVAL_base (Ast.BASE_app (id, Array.map resolve_ty tys))
        | Ast.PLVAL_base _ -> pl
        | Ast.PLVAL_ext_name (pexp, nc) ->
            let pexp = get_rebuilt_pexp pexp in
            let nc =
              match nc with
                  Ast.COMP_ident _
                | Ast.COMP_idx _ -> nc
                | Ast.COMP_app (id, tys) ->
                    Ast.COMP_app (id, Array.map resolve_ty tys)
            in
              Ast.PLVAL_ext_name (pexp, nc)

        | Ast.PLVAL_ext_pexp (a, b) ->
            Ast.PLVAL_ext_pexp (get_rebuilt_pexp a,
                                get_rebuilt_pexp b)
        | Ast.PLVAL_ext_deref p ->
            Ast.PLVAL_ext_deref (get_rebuilt_pexp p)
    in
    let p =
      match p.node with
          Ast.PEXP_lval pl ->
            let pl' = rebuild_plval pl in
              iflog cx (fun _ -> log cx "rebuilt plval %a as %a (#%d)"
                          Ast.sprintf_plval pl Ast.sprintf_plval pl'
                          (int_of_node p.id));
              { p with node = Ast.PEXP_lval pl' }

        | _ -> p
    in
      htab_put rebuilt_pexps p.id p
  in


  let visit_lval_pre lv =
    let rec rebuild_lval' lv =
      match lv with
          Ast.LVAL_ext (base, ext) ->
            let ext =
              match ext with
                  Ast.COMP_deref
                | Ast.COMP_named (Ast.COMP_ident _)
                | Ast.COMP_named (Ast.COMP_idx _)
                | Ast.COMP_atom (Ast.ATOM_literal _) -> ext
                | Ast.COMP_atom (Ast.ATOM_lval lv) ->
                    Ast.COMP_atom (Ast.ATOM_lval (rebuild_lval lv))
                | Ast.COMP_atom (Ast.ATOM_pexp _) ->
                    bug () "Resolve.rebuild_lval' on ATOM_pexp"

                | Ast.COMP_named (Ast.COMP_app (ident, params)) ->
                    Ast.COMP_named
                      (Ast.COMP_app (ident, Array.map resolve_ty params))
            in
              Ast.LVAL_ext (rebuild_lval' base, ext)

        | Ast.LVAL_base nb ->
            let node =
              match nb.node with
                  Ast.BASE_ident _
                | Ast.BASE_temp _ -> nb.node
                | Ast.BASE_app (ident, params) ->
                    Ast.BASE_app (ident, Array.map resolve_ty params)
            in
              Ast.LVAL_base {nb with node = node}

    and rebuild_lval lv =
      let id = lval_base_id lv in
      let lv' = rebuild_lval' lv in
        iflog cx (fun _ -> log cx "rebuilt lval %a as %a (#%d)"
                    Ast.sprintf_lval lv Ast.sprintf_lval lv'
                    (int_of_node id));
        htab_put cx.ctxt_all_lvals id lv';
        lv'
    in
      ignore (rebuild_lval lv);
      inner.Walk.visit_lval_pre lv
  in

  let visit_crate_post c =
    inner.Walk.visit_crate_post c;
    Hashtbl.iter (fun k v -> Hashtbl.add cx.ctxt_all_tag_info k v) tinfos
  in

    { inner with
        Walk.visit_slot_identified_pre = visit_slot_identified_pre;
        Walk.visit_mod_item_pre = visit_mod_item_pre;
        Walk.visit_obj_fn_pre = visit_obj_fn_pre;
        Walk.visit_obj_drop_pre = visit_obj_drop_pre;
        Walk.visit_stmt_pre = visit_stmt_pre;
        Walk.visit_lval_pre = visit_lval_pre;
        Walk.visit_pexp_post = visit_pexp_post;
        Walk.visit_crate_post = visit_crate_post }
;;


let lval_base_resolving_visitor
    (cx:ctxt)
    (scopes:(scope list) ref)
    (inner:Walk.visitor)
    : Walk.visitor =
  let lookup_defn_by_ident id ident =
    iflog cx
      (fun _ -> log cx "looking up slot or item with ident '%s'" ident);
    match lookup cx (!scopes) (Ast.KEY_ident ident) with
        RES_failed _ -> err (Some id) "unresolved identifier '%s'" ident
      | RES_ok (_, id) ->
          ((iflog cx (fun _ -> log cx "resolved to node id #%d"
                        (int_of_node id))); id)
  in
  let lookup_slot_by_temp id temp =
    iflog cx (fun _ -> log cx "looking up temp slot #%d" (int_of_temp temp));
    let res = lookup cx (!scopes) (Ast.KEY_temp temp) in
      match res with
          RES_failed _ -> err
            (Some id) "unresolved temp node #%d" (int_of_temp temp)
        | RES_ok (_, id) ->
            (iflog cx
               (fun _ -> log cx "resolved to node id #%d" (int_of_node id));
             id)
  in
  let lookup_defn_by_name_base id nb =
    match nb with
        Ast.BASE_ident ident
      | Ast.BASE_app (ident, _) -> lookup_defn_by_ident id ident
      | Ast.BASE_temp temp -> lookup_slot_by_temp id temp
  in

  let visit_lval_pre lv =
    let rec lookup_lval lv =
      iflog cx (fun _ ->
                  log cx "looking up lval #%d"
                    (int_of_node (lval_base_id lv)));
      match lv with
          Ast.LVAL_ext (base, ext) ->
            begin
              lookup_lval base;
              match ext with
                  Ast.COMP_atom (Ast.ATOM_lval lv') -> lookup_lval lv'

                | _ -> ()
            end
        | Ast.LVAL_base nb ->
            let defn_id = lookup_defn_by_name_base nb.id nb.node in
              iflog cx (fun _ -> log cx "resolved lval #%d to defn #%d"
                          (int_of_node nb.id) (int_of_node defn_id));
              htab_put cx.ctxt_lval_base_id_to_defn_base_id nb.id defn_id
    in

    (*
     * The point here is just to tickle the reference-a-name machinery in
     * lookup that makes sure that all and only those items referenced get
     * processed by later stages. An lval that happens to be an item will
     * mark the item in question here.
     *)
    let reference_any_name lv =
      let rec lval_is_name lv =
        match lv with
            Ast.LVAL_base {node = Ast.BASE_ident _; id = _}
          | Ast.LVAL_base {node = Ast.BASE_app _; id = _} -> true
          | Ast.LVAL_ext (lv', Ast.COMP_named (Ast.COMP_ident _))
          | Ast.LVAL_ext (lv', Ast.COMP_named (Ast.COMP_app _))
            -> lval_is_name lv'
          | _ -> false
      in
        if lval_is_name lv && lval_base_is_item cx lv
        then ignore (lookup_by_name cx [] (!scopes) (lval_to_name lv))
    in

      lookup_lval lv;
      reference_any_name lv;
      inner.Walk.visit_lval_pre lv
  in

  let visit_pexp_pre p =
    begin
    match p.node with
        Ast.PEXP_lval pl ->
          begin
            match pl with
                (Ast.PLVAL_base (Ast.BASE_ident ident))
              | (Ast.PLVAL_base (Ast.BASE_app (ident, _))) ->
                  let id = lookup_defn_by_ident p.id ident in

                    iflog cx
                      (fun _ ->
                         log cx "resolved plval %a = #%d to defn #%d"
                           Ast.sprintf_plval pl
                           (int_of_node p.id) (int_of_node id));

                    (* Record the pexp -> defn mapping. *)
                    htab_put cx.ctxt_lval_base_id_to_defn_base_id p.id id;

                    (* Tickle the referenced-ness table if it's an item. *)
                    if defn_id_is_item cx id
                    then ignore (lookup_by_name cx [] (!scopes)
                                   (plval_to_name pl))
              | _ -> ()
          end

      | _ -> ()
    end;
    inner.Walk.visit_pexp_pre p
  in

    { inner with
        Walk.visit_lval_pre = visit_lval_pre;
        Walk.visit_pexp_pre = visit_pexp_pre
    };
;;


let pattern_resolving_visitor
    (cx:ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =

  let not_tag_ctor nm id : unit =
    err (Some id) "'%s' is not a tag constructor" (string_of_name nm)
  in

  let resolve_pat_tag
      (name:Ast.name)
      (id:node_id)
      (pats:Ast.pat array)
      (tag_ctor_id:node_id)
      : unit =

    (* NB this isn't really the proper tag type, since we aren't applying any
     * type parameters from the tag constructor in the pattern, but since we
     * are only looking at the fact that it's a tag-like type at all, and
     * asking for its arity, it doesn't matter that the possibly parametric
     * tag type has its parameters unbound here. *)
    let tag_ty =
      match Hashtbl.find cx.ctxt_all_item_types tag_ctor_id with
          Ast.TY_tag t -> Ast.TY_tag t
        | ft -> fn_output_ty ft
    in
      begin
        match tag_ty with
            Ast.TY_tag ttag ->
              let ident =
                match name with
                    Ast.NAME_ext (_, Ast.COMP_ident id)
                  | Ast.NAME_ext (_, Ast.COMP_app (id, _))
                  | Ast.NAME_base (Ast.BASE_ident id)
                  | Ast.NAME_base (Ast.BASE_app (id, _)) -> id
                  | _ -> err (Some id) "pattern-name ends in non-ident"
              in
              let tinfo = Hashtbl.find cx.ctxt_all_tag_info ttag.Ast.tag_id in
              let (_, _, ttup) = Hashtbl.find tinfo.tag_idents ident in
              let arity = Array.length ttup in
                if (Array.length pats) != arity
                then
                  err (Some id)
                    "tag pattern '%s' with wrong number of components"
                    (string_of_name name)
                else ()
          | _ -> not_tag_ctor name id
      end
  in

  let resolve_arm { node = arm; id = id } =
    match fst arm with
        Ast.PAT_tag (lval, pats) ->
          let lval_nm = lval_to_name lval in
          let lval_id = lval_base_id lval in
          let tag_ctor_id = (lval_item ~node_id:id cx lval).id in
            if defn_id_is_item cx tag_ctor_id

            (* FIXME (issue #76): we should actually check here that the
             * function is a tag value-ctor.  For now this actually allows
             * any function returning a tag type to pass as a tag
             * pattern.  *)
            then resolve_pat_tag lval_nm lval_id pats tag_ctor_id
            else not_tag_ctor lval_nm lval_id
      | _ -> ()
  in

  let visit_stmt_pre stmt =
    begin
      match stmt.node with
          Ast.STMT_alt_tag { Ast.alt_tag_lval = _;
                             Ast.alt_tag_arms = arms } ->
            Array.iter resolve_arm arms
        | _ -> ()
    end;
    inner.Walk.visit_stmt_pre stmt
  in
    { inner with Walk.visit_stmt_pre = visit_stmt_pre }
;;

let export_referencing_visitor
    (cx:ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =
  let visit_mod_item_pre id params item =
    begin
      match item.node.Ast.decl_item with
          Ast.MOD_ITEM_mod (view, items) ->
            let is_defining_mod =
              (* auto-ref the default-export cases only if
               * the containing mod is 'defining', meaning
               * not-native / not-use
               *)
                 not (Hashtbl.mem cx.ctxt_required_items item.id)
              in
              let reference _ item =
                Hashtbl.replace cx.ctxt_node_referenced item.id ();
              in
              let reference_export e _ =
                match e with
                    Ast.EXPORT_ident ident ->
                      let item = Hashtbl.find items ident in
                        reference ident item
                  | Ast.EXPORT_all_decls ->
                      if is_defining_mod
                      then Hashtbl.iter reference items
              in
                Hashtbl.iter reference_export view.Ast.view_exports
          | _ -> ()
      end;
      inner.Walk.visit_mod_item_pre id params item
    in
      { inner with Walk.visit_mod_item_pre = visit_mod_item_pre }


;;

let process_crate
    (cx:ctxt)
    (crate:Ast.crate)
    : unit =
  let (scopes:(scope list) ref) = ref [] in

  let passes_0 =
    [|
      (block_scope_forming_visitor cx Walk.empty_visitor);
      (stmt_collecting_visitor cx
         (all_item_collecting_visitor cx
            Walk.empty_visitor));
    |]
  in

  let passes_1 =
    [|
      (scope_stack_managing_visitor scopes
         (type_resolving_visitor cx scopes
            (lval_base_resolving_visitor cx scopes
               Walk.empty_visitor)));
    |]
  in

  let passes_2 =
    [|
      (scope_stack_managing_visitor scopes
         (pattern_resolving_visitor cx
            Walk.empty_visitor));
      export_referencing_visitor cx Walk.empty_visitor
    |]
  in
  let log_flag = cx.ctxt_sess.Session.sess_log_resolve in
    log cx "running primary resolve passes";
    run_passes cx "resolve collect" passes_0 log_flag log crate;
    log cx "running secondary resolve passes";
    run_passes cx "resolve bind" passes_1 log_flag log crate;
    log cx "running tertiary resolve passes";
    run_passes cx "resolve patterns" passes_2 log_flag log crate;

    iflog cx
      begin
        fun _ ->
          Hashtbl.iter
            begin
              fun n _ ->
                if defn_id_is_item cx n
                then
                  log cx "referenced: %a"
                    Ast.sprintf_name
                    (Hashtbl.find cx.ctxt_all_item_names n)
            end
            cx.ctxt_node_referenced;
      end;
    (* Post-resolve, we can establish a tag cache. *)
    cx.ctxt_tag_cache <- Some (Hashtbl.create 0);
    cx.ctxt_rebuild_cache <- Some (Hashtbl.create 0)
;;

(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)

