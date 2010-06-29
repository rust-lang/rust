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


let log cx = Session.log "resolve"
  cx.ctxt_sess.Session.sess_log_resolve
  cx.ctxt_sess.Session.sess_log_out
;;

let iflog cx thunk =
  if cx.ctxt_sess.Session.sess_log_resolve
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
        | Ast.STMT_alt_tag { Ast.alt_tag_arms = arms } ->
            let rec resolve_pat block pat =
              match pat with
                  Ast.PAT_slot ({ id = slot_id }, ident) ->
                    let slots = Hashtbl.find cx.ctxt_block_slots block.id in
                    let key = Ast.KEY_ident ident in
                    htab_put slots key slot_id;
                    htab_put cx.ctxt_slot_keys slot_id key
                | Ast.PAT_tag (_, pats) -> Array.iter (resolve_pat block) pats
                | Ast.PAT_lit _ | Ast.PAT_wild -> ()
            in
            Array.iter (fun { node = (p, b) } -> resolve_pat b p) arms
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
    (path:Ast.name_component Stack.t)
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
    htab_put cx.ctxt_all_item_names i.id (Walk.path_to_name path);
    log cx "collected item #%d: %s" (int_of_node i.id) n;
    begin
      match i.node.Ast.decl_item with
          Ast.MOD_ITEM_fn f ->
            note_header i.id f.Ast.fn_input_slots;
        | Ast.MOD_ITEM_obj ob ->
            note_header i.id ob.Ast.obj_state;
        | Ast.MOD_ITEM_tag (header_slots, _, _) ->
            let skey i = Printf.sprintf "_%d" i in
              note_header i.id
                (Array.mapi (fun i s -> (s, skey i)) header_slots)
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
    htab_put cx.ctxt_all_item_names fn.id (Walk.path_to_name path);
    note_header fn.id fn.node.Ast.fn_input_slots;
    inner.Walk.visit_obj_fn_pre obj ident fn
  in

  let visit_obj_drop_pre obj b =
    htab_put cx.ctxt_all_defns b.id (DEFN_obj_drop obj.id);
    htab_put cx.ctxt_all_item_names b.id (Walk.path_to_name path);
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
                (Walk.path_to_name path);
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
  match lookup_by_name cx scopes name with
      None -> err None "unknown name: %a" Ast.sprintf_name name
    | Some (_, id) ->
        match htab_search cx.ctxt_all_defns id with
            Some (DEFN_item { Ast.decl_item = Ast.MOD_ITEM_type _ })
          | Some (DEFN_item { Ast.decl_item = Ast.MOD_ITEM_obj _ })
          | Some (DEFN_ty_param _) -> id
          | _ ->
              err None "Found non-type binding for %a"
                Ast.sprintf_name name
;;


let get_ty_references
    (t:Ast.ty)
    (cx:ctxt)
    (scopes:scope list)
    : node_id list =
  let base = ty_fold_list_concat () in
  let ty_fold_named n =
    [ lookup_type_node_by_name cx scopes n ]
  in
  let fold = { base with ty_fold_named = ty_fold_named } in
    fold_ty fold t
;;


let type_reference_and_tag_extracting_visitor
    (cx:ctxt)
    (scopes:(scope list) ref)
    (node_to_references:(node_id,node_id list) Hashtbl.t)
    (all_tags:(node_id,(Ast.ty_tag * (scope list))) Hashtbl.t)
    (inner:Walk.visitor)
    : Walk.visitor =
  let visit_mod_item_pre id params item =
    begin
      match item.node.Ast.decl_item with
          Ast.MOD_ITEM_type ty ->
            begin
              log cx "extracting references for type node %d"
                (int_of_node item.id);
              let referenced = get_ty_references ty cx (!scopes) in
                List.iter
                  (fun i -> log cx "type %d references type %d"
                     (int_of_node item.id) (int_of_node i)) referenced;
                htab_put node_to_references item.id referenced;
                match ty with
                    Ast.TY_tag ttag ->
                      htab_put all_tags item.id (ttag, (!scopes))
                  | _ -> ()
            end
        | _ -> ()
    end;
    inner.Walk.visit_mod_item_pre id params item
  in
    { inner with
        Walk.visit_mod_item_pre = visit_mod_item_pre }
;;


type recur_info =
    { recur_all_nodes: node_id list;
      recur_curr_iso: (node_id array) option; }
;;

let empty_recur_info =
  { recur_all_nodes = [];
    recur_curr_iso = None }
;;

let push_node r n =
  { r with recur_all_nodes = n :: r.recur_all_nodes }
;;

let set_iso r i =
  { r with recur_curr_iso = Some i }
;;


let index_in_curr_iso (recur:recur_info) (node:node_id) : int option =
  match recur.recur_curr_iso with
      None -> None
    | Some iso ->
        let rec search i =
          if i >= (Array.length iso)
          then None
          else
            if iso.(i) = node
            then Some i
            else search (i+1)
        in
          search 0
;;

let need_ty_tag t =
  match t with
      Ast.TY_tag ttag -> ttag
    | _ -> err None "needed ty_tag"
;;


let rec ty_iso_of
    (cx:ctxt)
    (recursive_tag_groups:(node_id,(node_id,unit) Hashtbl.t) Hashtbl.t)
    (all_tags:(node_id,(Ast.ty_tag * (scope list))) Hashtbl.t)
    (n:node_id)
    : Ast.ty =
  let _ = iflog cx (fun _ -> log cx "+++ ty_iso_of #%d" (int_of_node n)) in
  let group_table = Hashtbl.find recursive_tag_groups n in
  let group_array = Array.of_list (htab_keys group_table) in
  let compare_nodes a_id b_id =
    let a_name = Hashtbl.find cx.ctxt_all_item_names a_id in
    let b_name = Hashtbl.find cx.ctxt_all_item_names b_id in
      compare a_name b_name
  in
  let recur = set_iso (push_node empty_recur_info n) group_array in
  let resolve_member member =
    let (tag, scopes) = Hashtbl.find all_tags member in
    let ty = Ast.TY_tag tag in
    let ty = resolve_type cx scopes recursive_tag_groups all_tags recur ty in
      need_ty_tag ty
  in
    Array.sort compare_nodes group_array;
    log cx "resolving node %d, %d-member iso group"
      (int_of_node n) (Array.length group_array);
    Array.iteri (fun i n -> log cx "member %d: %d" i
                   (int_of_node n)) group_array;
    let group = Array.map resolve_member group_array in
    let rec search i =
      if i >= (Array.length group_array)
      then err None "node is not a member of its own iso group"
      else
        if group_array.(i) = n
        then i
        else search (i+1)
    in
    let iso =
      Ast.TY_iso { Ast.iso_index = (search 0);
                   Ast.iso_group = group }
    in
    iflog cx (fun _ ->
                log cx "--- ty_iso_of #%d ==> %a"
                  (int_of_node n) Ast.sprintf_ty iso);
      iso


and lookup_type_by_name
    (cx:ctxt)
    (scopes:scope list)
    (recursive_tag_groups:(node_id,(node_id,unit) Hashtbl.t) Hashtbl.t)
    (all_tags:(node_id,(Ast.ty_tag * (scope list))) Hashtbl.t)
    (recur:recur_info)
    (name:Ast.name)
    : ((scope list) * node_id * Ast.ty) =
  iflog cx (fun _ ->
              log cx "+++ lookup_type_by_name %a"
                Ast.sprintf_name name);
  match lookup_by_name cx scopes name with
      None -> err None "unknown name: %a" Ast.sprintf_name name
    | Some (scopes', id) ->
        let ty, params =
          match htab_search cx.ctxt_all_defns id with
              Some (DEFN_item { Ast.decl_item = Ast.MOD_ITEM_type t;
                                Ast.decl_params = params }) ->
                (t, Array.map (fun p -> p.node) params)
            | Some (DEFN_item { Ast.decl_item = Ast.MOD_ITEM_obj ob;
                                Ast.decl_params = params }) ->
                (Ast.TY_obj (ty_obj_of_obj ob),
                 Array.map (fun p -> p.node) params)
            | Some (DEFN_ty_param (_, x)) ->
                (Ast.TY_param x, [||])
            | _ ->
                err None "Found non-type binding for %a"
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
                  resolve_type cx scopes recursive_tag_groups
                    all_tags recur t
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
          let ty = rebuild_ty_under_params ty params args true in
            iflog cx (fun _ -> log cx "--- lookup_type_by_name %a ==> %a"
                        Ast.sprintf_name name
                        Ast.sprintf_ty ty);
            (scopes', id, ty)

and resolve_type
    (cx:ctxt)
    (scopes:(scope list))
    (recursive_tag_groups:(node_id,(node_id,unit) Hashtbl.t) Hashtbl.t)
    (all_tags:(node_id,(Ast.ty_tag * (scope list))) Hashtbl.t)
    (recur:recur_info)
    (t:Ast.ty)
    : Ast.ty =
  let _ = iflog cx (fun _ -> log cx "+++ resolve_type %a" Ast.sprintf_ty t) in
  let base = ty_fold_rebuild (fun t -> t) in
  let ty_fold_named name =
    let (scopes, node, t) =
      lookup_type_by_name cx scopes recursive_tag_groups all_tags recur name
    in
      iflog cx (fun _ ->
                  log cx "resolved type name '%a' to item %d with ty %a"
                  Ast.sprintf_name name (int_of_node node) Ast.sprintf_ty t);
      match index_in_curr_iso recur node with
          Some i -> Ast.TY_idx i
        | None ->
            if Hashtbl.mem recursive_tag_groups node
            then
              begin
                let ttag = need_ty_tag t in
                  Hashtbl.replace all_tags node (ttag, scopes);
                  ty_iso_of cx recursive_tag_groups all_tags node
              end
            else
              if List.mem node recur.recur_all_nodes
              then (err (Some node) "infinite recursive type definition: '%a'"
                      Ast.sprintf_name name)
              else
                let recur = push_node recur node in
                  iflog cx (fun _ -> log cx "recursively resolving type %a"
                              Ast.sprintf_ty t);
                  resolve_type cx scopes recursive_tag_groups all_tags recur t
  in
  let fold =
    { base with
        ty_fold_named = ty_fold_named; }
  in
  let t' = fold_ty fold t in
    iflog cx (fun _ ->
                log cx "--- resolve_type %a ==> %a"
                  Ast.sprintf_ty t Ast.sprintf_ty t');
    t'
;;


let type_resolving_visitor
    (cx:ctxt)
    (scopes:(scope list) ref)
    (recursive_tag_groups:(node_id,(node_id,unit) Hashtbl.t) Hashtbl.t)
    (all_tags:(node_id,(Ast.ty_tag * (scope list))) Hashtbl.t)
    (inner:Walk.visitor)
    : Walk.visitor =

  let resolve_ty (t:Ast.ty) : Ast.ty =
    resolve_type cx (!scopes) recursive_tag_groups all_tags empty_recur_info t
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
      log cx "collected resolved slot #%d with type %s" (int_of_node slot.id)
        (match slot.node.Ast.slot_ty with
             None -> "??"
           | Some t -> (Fmt.fmt_to_str Ast.fmt_ty t));
      inner.Walk.visit_slot_identified_pre slot
  in

  let visit_mod_item_pre id params item =
    begin
      try
        match item.node.Ast.decl_item with
            Ast.MOD_ITEM_type ty ->
              let ty =
                resolve_type cx (!scopes) recursive_tag_groups
                  all_tags empty_recur_info ty
              in
                log cx "resolved item %s, defining type %a"
                  id Ast.sprintf_ty ty;
                htab_put cx.ctxt_all_type_items item.id ty;
                htab_put cx.ctxt_all_item_types item.id Ast.TY_type

          (* 
           * Don't resolve the "type" of a mod item; just resolve its
           * members.
           *)
          | Ast.MOD_ITEM_mod _ -> ()

          | Ast.MOD_ITEM_tag (header_slots, _, nid)
              when Hashtbl.mem recursive_tag_groups nid ->
              begin
                match ty_of_mod_item true item with
                    Ast.TY_fn (tsig, taux) ->
                      let input_slots =
                        Array.map
                          (fun sloti -> resolve_slot sloti.node)
                          header_slots
                      in
                      let output_slot =
                        interior_slot (ty_iso_of cx recursive_tag_groups
                                         all_tags nid)
                      in
                      let ty =
                        Ast.TY_fn
                          ({tsig with
                              Ast.sig_input_slots = input_slots;
                              Ast.sig_output_slot = output_slot }, taux)
                      in
                        log cx "resolved recursive tag %s, type as %a"
                          id Ast.sprintf_ty ty;
                        htab_put cx.ctxt_all_item_types item.id ty
                  | _ -> bug () "recursive tag with non-function type"
              end

          | _ ->
              let t = ty_of_mod_item true item in
              let ty =
                resolve_type cx (!scopes) recursive_tag_groups
                  all_tags empty_recur_info t
              in
                log cx "resolved item %s, type as %a" id Ast.sprintf_ty ty;
                htab_put cx.ctxt_all_item_types item.id ty;
      with
          Semant_err (None, e) -> raise (Semant_err ((Some item.id), e))
    end;
    inner.Walk.visit_mod_item_pre id params item
  in

  let visit_obj_fn_pre obj ident fn =
    let fty =
      resolve_type cx (!scopes) recursive_tag_groups all_tags
        empty_recur_info (Ast.TY_fn (ty_fn_of_fn fn.node))
    in
      log cx "resolved obj fn %s as %a" ident Ast.sprintf_ty fty;
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

  let visit_lval_pre lv =
    let rec rebuild_lval' lv =
      match lv with
          Ast.LVAL_ext (base, ext) ->
            let ext =
              match ext with
                  Ast.COMP_named (Ast.COMP_ident _)
                | Ast.COMP_named (Ast.COMP_idx _)
                | Ast.COMP_atom (Ast.ATOM_literal _) -> ext
                | Ast.COMP_atom (Ast.ATOM_lval lv) ->
                    Ast.COMP_atom (Ast.ATOM_lval (rebuild_lval lv))
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

    { inner with
        Walk.visit_slot_identified_pre = visit_slot_identified_pre;
        Walk.visit_mod_item_pre = visit_mod_item_pre;
        Walk.visit_obj_fn_pre = visit_obj_fn_pre;
        Walk.visit_obj_drop_pre = visit_obj_drop_pre;
        Walk.visit_stmt_pre = visit_stmt_pre;
        Walk.visit_lval_pre = visit_lval_pre; }
;;


let lval_base_resolving_visitor
    (cx:ctxt)
    (scopes:(scope list) ref)
    (inner:Walk.visitor)
    : Walk.visitor =
  let lookup_referent_by_ident id ident =
    log cx "looking up slot or item with ident '%s'" ident;
    match lookup cx (!scopes) (Ast.KEY_ident ident) with
        None -> err (Some id) "unresolved identifier '%s'" ident
      | Some (_, id) -> (log cx "resolved to node id #%d"
                           (int_of_node id); id)
  in
  let lookup_slot_by_temp id temp =
    log cx "looking up temp slot #%d" (int_of_temp temp);
    let res = lookup cx (!scopes) (Ast.KEY_temp temp) in
      match res with
          None -> err
            (Some id) "unresolved temp node #%d" (int_of_temp temp)
        | Some (_, id) ->
            (log cx "resolved to node id #%d" (int_of_node id); id)
  in
  let lookup_referent_by_name_base id nb =
    match nb with
        Ast.BASE_ident ident
      | Ast.BASE_app (ident, _) -> lookup_referent_by_ident id ident
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
            let referent_id = lookup_referent_by_name_base nb.id nb.node in
              iflog cx (fun _ -> log cx "resolved lval #%d to referent #%d"
                          (int_of_node nb.id) (int_of_node referent_id));
              htab_put cx.ctxt_lval_to_referent nb.id referent_id
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
            Ast.LVAL_base {node = Ast.BASE_ident _}
          | Ast.LVAL_base {node = Ast.BASE_app _} -> true
          | Ast.LVAL_ext (lv', Ast.COMP_named (Ast.COMP_ident _))
          | Ast.LVAL_ext (lv', Ast.COMP_named (Ast.COMP_app _))
            -> lval_is_name lv'
          | _ -> false
      in
        if lval_is_name lv && lval_is_item cx lv
        then ignore (lookup_by_name cx (!scopes) (lval_to_name lv))
    in

      lookup_lval lv;
      reference_any_name lv;
      inner.Walk.visit_lval_pre lv
  in
    { inner with
        Walk.visit_lval_pre = visit_lval_pre };
;;



(*
 * iso-recursion groups are very complicated.
 * 
 *   - iso groups are always rooted at *named* ty_tag nodes
 * 
 *   - consider: 
 * 
 *    type colour = tag(red, green, blue);
 *    type list = tag(cons(colour, @list), nil())
 * 
 *    this should include list as an iso but not colour,
 *    should result in:
 * 
 *    type list = iso[<0>:tag(cons(tag(red,green,blue),@#1))]
 * 
 *   - consider:
 * 
 *    type colour = tag(red, green, blue);
 *    type tree = tag(children(@list), leaf(colour))
 *    type list = tag(cons(@tree, @list), nil())
 * 
 *    this should result in:
 * 
 *    type list = iso[<0>:tag(cons(@#2, @#1),nil());
 *                    1: tag(children(@#1),leaf(tag(red,green,blue)))]
 * 
 *  - how can you calculate these?
 * 
 *    - start by making a map from named-tag-node-id -> referenced-other-nodes
 *    - for each member in the set, if you can get from itself to itself, keep
 *      it, otherwise it's non-recursive => non-interesting, delete it.
 *    - group the members (now all recursive) by dependency
 *    - assign index-number to each elt of group
 *    - fully resolve each elt of group, turning names into numbers or chasing
 *      through to fully-resolving targets as necessary
 *    - place group in iso, store differently-indexed value in table for each
 * 
 * 
 *  - what are the illegal forms?
 *    - recursion that takes indefinite storage to form a tag, eg.
 * 
 *      type t = tag(foo(t));
 *
 *    - recursion that makes a tag unconstructable, eg:
 * 
 *      type t = tag(foo(@t));
 *)

let resolve_recursion
    (cx:ctxt)
    (node_to_references:(node_id,node_id list) Hashtbl.t)
    (recursive_tag_groups:(node_id,(node_id,unit) Hashtbl.t) Hashtbl.t)
    : unit =

  let recursive_tag_types = Hashtbl.create 0 in

  let rec can_reach
      (target:node_id)
      (visited:node_id list)
      (curr:node_id)
      : bool =
    if List.mem curr visited
    then false
    else
      match htab_search node_to_references curr with
          None -> false
        | Some referenced ->
            if List.mem target referenced
            then true
            else List.exists (can_reach target (curr :: visited)) referenced
  in

  let extract_recursive_tags _ =
    Hashtbl.iter
      begin fun id _ ->
        if can_reach id [] id
        then begin
          match Hashtbl.find cx.ctxt_all_defns id with
              DEFN_item
                { Ast.decl_item = Ast.MOD_ITEM_type (Ast.TY_tag _) } ->
                log cx "type %d is a recursive tag" (int_of_node id);
                Hashtbl.replace recursive_tag_types id ()
            | _ ->
                log cx "type %d is recursive, but not a tag" (int_of_node id);
        end
        else log cx "type %d is non-recursive" (int_of_node id);
      end
      node_to_references
  in

  let group_recursive_tags _ =
    while (Hashtbl.length recursive_tag_types) != 0 do
      let keys = htab_keys recursive_tag_types in
      let root = List.hd keys in
      let group = Hashtbl.create 0 in
      let rec walk visited node =
        if List.mem node visited
        then ()
        else
          begin
            if Hashtbl.mem recursive_tag_types node
            then
              begin
                Hashtbl.remove recursive_tag_types node;
                htab_put recursive_tag_groups node group;
                htab_put group node ();
                log cx "recursion group rooted at tag %d contains tag %d"
                  (int_of_node root) (int_of_node node);
              end;
            match htab_search node_to_references node with
                None -> ()
              | Some referenced ->
                  List.iter (walk (node :: visited)) referenced
          end
      in
        walk [] root;
    done
  in

    begin
      extract_recursive_tags ();
      group_recursive_tags ();
      log cx "found %d independent type-recursion groups"
        (Hashtbl.length recursive_tag_groups);
    end
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
      fn_output_ty (Hashtbl.find cx.ctxt_all_item_types tag_ctor_id)
    in
      begin
        match tag_ty with
            Ast.TY_tag _
          | Ast.TY_iso _ ->
              let tag_ty_tup = tag_or_iso_ty_tup_by_name tag_ty name in
              let arity = Array.length tag_ty_tup in
                if (Array.length pats) != arity
                then
                  err (Some id)
                    "tag pattern '%s' with wrong number of components"
                    (string_of_name name)
                else ()
          | _ -> not_tag_ctor name id
      end
  in

  let resolve_arm { node = arm } =
    match fst arm with
        Ast.PAT_tag (lval, pats) ->
          let lval_nm = lval_to_name lval in
          let lval_id = lval_base_id lval in
          let tag_ctor_id = lval_to_referent cx lval_id in
            if referent_is_item cx tag_ctor_id

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
  let path = Stack.create () in

  let node_to_references = Hashtbl.create 0 in
  let all_tags = Hashtbl.create 0 in
  let recursive_tag_groups = Hashtbl.create 0 in

  let passes_0 =
    [|
      (block_scope_forming_visitor cx Walk.empty_visitor);
      (stmt_collecting_visitor cx
         (all_item_collecting_visitor cx path
            Walk.empty_visitor));
      (scope_stack_managing_visitor scopes
         (type_reference_and_tag_extracting_visitor
            cx scopes node_to_references all_tags
            Walk.empty_visitor))
    |]
  in

  let passes_1 =
    [|
      (scope_stack_managing_visitor scopes
         (type_resolving_visitor cx scopes
            recursive_tag_groups all_tags
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

    log cx "running primary resolve passes";
    run_passes cx "resolve collect" path passes_0 (log cx "%s") crate;
    resolve_recursion cx node_to_references recursive_tag_groups;
    log cx "running secondary resolve passes";
    run_passes cx "resolve bind" path passes_1 (log cx "%s") crate;
    log cx "running tertiary resolve passes";
    run_passes cx "resolve patterns" path passes_2 (log cx "%s") crate;

    iflog cx
      begin
        fun _ ->
          Hashtbl.iter
            begin
              fun n _ ->
                if referent_is_item cx n
                then
                  log cx "referenced: %a"
                    Ast.sprintf_name
                    (Hashtbl.find cx.ctxt_all_item_names n)
            end
            cx.ctxt_node_referenced;
      end
;;

(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)

