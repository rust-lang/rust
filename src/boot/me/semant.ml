
open Common;;

type slots_table = (Ast.slot_key,node_id) Hashtbl.t
type items_table = (Ast.ident,node_id) Hashtbl.t
type block_slots_table = (node_id,slots_table) Hashtbl.t
type block_items_table = (node_id,items_table) Hashtbl.t
;;


type code = {
  code_fixup: fixup;
  code_quads: Il.quads;
  code_vregs_and_spill: (int * fixup) option;
}
;;

type glue =
    GLUE_activate
  | GLUE_yield
  | GLUE_exit_main_task
  | GLUE_exit_task
  | GLUE_copy of Ast.ty           (* One-level copy.                    *)
  | GLUE_drop of Ast.ty           (* De-initialize local memory.        *)
  | GLUE_free of Ast.ty           (* Drop body + free() box ptr.        *)
  | GLUE_sever of Ast.ty          (* Null all box state slots.          *)
  | GLUE_mark of Ast.ty           (* Mark all box state slots.          *)
  | GLUE_clone of Ast.ty          (* Deep copy.                         *)
  | GLUE_compare of Ast.ty
  | GLUE_hash of Ast.ty
  | GLUE_write of Ast.ty
  | GLUE_read of Ast.ty
  | GLUE_unwind
  | GLUE_gc
  | GLUE_get_next_pc
  | GLUE_mark_frame of node_id    (* Node is the frame.                 *)
  | GLUE_drop_frame of node_id    (* Node is the frame.                 *)
  | GLUE_reloc_frame of node_id   (* Node is the frame.                 *)
  | GLUE_fn_binding of node_id    (* Node is the 'bind' stmt.           *)
  | GLUE_obj_drop of node_id      (* Node is the obj.                   *)
  | GLUE_loop_body of node_id     (* Node is the 'for each' body block. *)
  | GLUE_forward of (Ast.ident * Ast.ty_obj * Ast.ty_obj)
;;

type data =
    DATA_str of string
  | DATA_name of Ast.name
  | DATA_tydesc of Ast.ty
  | DATA_frame_glue_fns of node_id
  | DATA_obj_vtbl of node_id
  | DATA_forwarding_vtbl of (Ast.ty_obj * Ast.ty_obj)
  | DATA_crate
;;

type defn =
    DEFN_slot of Ast.slot
  | DEFN_item of Ast.mod_item_decl
  | DEFN_ty_param of Ast.ty_param
  | DEFN_obj_fn of (node_id * Ast.fn)
  | DEFN_obj_drop of node_id
  | DEFN_loop_body of node_id
;;

type glue_code = (glue, code) Hashtbl.t;;
type item_code = (node_id, code) Hashtbl.t;;
type file_code = (node_id, item_code) Hashtbl.t;;
type data_frags = (data, (fixup * Asm.frag)) Hashtbl.t;;

let string_of_name (n:Ast.name) : string =
  Fmt.fmt_to_str Ast.fmt_name n
;;

(* The only need for a carg is to uniquely identify a constraint-arg
 * in a scope-independent fashion. So we just look up the node that's
 * used as the base of any such arg and glue it on the front of the 
 * symbolic name.
 *)

type constr_key_arg = Constr_arg_node of (node_id * Ast.carg_path)
                      | Constr_arg_lit of Ast.lit
type constr_key =
    Constr_pred of (node_id * constr_key_arg array)
  | Constr_init of node_id

type ctxt =
    { ctxt_sess: Session.sess;
      ctxt_frame_args: (node_id,node_id list) Hashtbl.t;
      ctxt_frame_blocks: (node_id,node_id list) Hashtbl.t;
      ctxt_block_slots: block_slots_table;
      ctxt_block_items: block_items_table;
      ctxt_slot_is_arg: (node_id,unit) Hashtbl.t;
      ctxt_slot_keys: (node_id,Ast.slot_key) Hashtbl.t;
      ctxt_node_referenced: (node_id, unit) Hashtbl.t;
      ctxt_auto_deref_lval: (node_id, bool) Hashtbl.t;
      ctxt_all_item_names: (node_id,Ast.name) Hashtbl.t;
      ctxt_all_item_types: (node_id,Ast.ty) Hashtbl.t;
      ctxt_all_lval_types: (node_id,Ast.ty) Hashtbl.t;
      ctxt_all_cast_types: (node_id,Ast.ty) Hashtbl.t;
      ctxt_all_type_items: (node_id,Ast.ty) Hashtbl.t;
      ctxt_all_stmts: (node_id,Ast.stmt) Hashtbl.t;
      ctxt_item_files: (node_id,filename) Hashtbl.t;
      ctxt_all_lvals: (node_id,Ast.lval) Hashtbl.t;

      (* definition id --> definition *)
      ctxt_all_defns: (node_id,defn) Hashtbl.t;

      (* reference id --> definition id *)
      ctxt_lval_to_referent: (node_id,node_id) Hashtbl.t;

      ctxt_required_items: (node_id, (required_lib * nabi_conv)) Hashtbl.t;
      ctxt_required_syms: (node_id, string) Hashtbl.t;

      (* Layout-y stuff. *)
      ctxt_slot_aliased: (node_id,unit) Hashtbl.t;
      ctxt_slot_is_obj_state: (node_id,unit) Hashtbl.t;
      ctxt_slot_vregs: (node_id,((int option) ref)) Hashtbl.t;
      ctxt_slot_offsets: (node_id,size) Hashtbl.t;
      ctxt_frame_sizes: (node_id,size) Hashtbl.t;
      ctxt_call_sizes: (node_id,size) Hashtbl.t;
      ctxt_block_is_loop_body: (node_id,unit) Hashtbl.t;
      ctxt_stmt_loop_depths: (node_id,int) Hashtbl.t;
      ctxt_slot_loop_depths: (node_id,int) Hashtbl.t;

      (* Typestate-y stuff. *)
      ctxt_constrs: (constr_id,constr_key) Hashtbl.t;
      ctxt_constr_ids: (constr_key,constr_id) Hashtbl.t;
      ctxt_preconditions: (node_id,Bits.t) Hashtbl.t;
      ctxt_postconditions: (node_id,Bits.t) Hashtbl.t;
      ctxt_prestates: (node_id,Bits.t) Hashtbl.t;
      ctxt_poststates: (node_id,Bits.t) Hashtbl.t;
      ctxt_call_lval_params: (node_id,Ast.ty array) Hashtbl.t;
      ctxt_copy_stmt_is_init: (node_id,unit) Hashtbl.t;
      ctxt_post_stmt_slot_drops: (node_id,node_id list) Hashtbl.t;

      (* Translation-y stuff. *)
      ctxt_fn_fixups: (node_id,fixup) Hashtbl.t;
      ctxt_block_fixups: (node_id,fixup) Hashtbl.t;
      ctxt_file_fixups: (node_id,fixup) Hashtbl.t;
      ctxt_spill_fixups: (node_id,fixup) Hashtbl.t;
      ctxt_abi: Abi.abi;
      ctxt_activate_fixup: fixup;
      ctxt_gc_fixup: fixup;
      ctxt_yield_fixup: fixup;
      ctxt_unwind_fixup: fixup;
      ctxt_exit_task_fixup: fixup;

      ctxt_debug_aranges_fixup: fixup;
      ctxt_debug_pubnames_fixup: fixup;
      ctxt_debug_info_fixup: fixup;
      ctxt_debug_abbrev_fixup: fixup;
      ctxt_debug_line_fixup: fixup;
      ctxt_debug_frame_fixup: fixup;

      ctxt_image_base_fixup: fixup;
      ctxt_crate_fixup: fixup;

      ctxt_file_code: file_code;
      ctxt_all_item_code: item_code;
      ctxt_glue_code: glue_code;
      ctxt_data: data_frags;

      ctxt_native_required:
        (required_lib,((string,fixup) Hashtbl.t)) Hashtbl.t;
      ctxt_native_provided:
        (segment,((string, fixup) Hashtbl.t)) Hashtbl.t;

      ctxt_required_rust_sym_num: (node_id, int) Hashtbl.t;
      ctxt_required_c_sym_num: ((required_lib * string), int) Hashtbl.t;
      ctxt_required_lib_num: (required_lib, int) Hashtbl.t;

      ctxt_main_fn_fixup: fixup option;
      ctxt_main_name: string option;
    }
;;

let new_ctxt sess abi crate =
  { ctxt_sess = sess;
    ctxt_frame_args = Hashtbl.create 0;
    ctxt_frame_blocks = Hashtbl.create 0;
    ctxt_block_slots = Hashtbl.create 0;
    ctxt_block_items = Hashtbl.create 0;
    ctxt_slot_is_arg = Hashtbl.create 0;
    ctxt_slot_keys = Hashtbl.create 0;
    ctxt_node_referenced = Hashtbl.create 0;
    ctxt_auto_deref_lval = Hashtbl.create 0;
    ctxt_all_item_names = Hashtbl.create 0;
    ctxt_all_item_types = Hashtbl.create 0;
    ctxt_all_lval_types = Hashtbl.create 0;
    ctxt_all_cast_types = Hashtbl.create 0;
    ctxt_all_type_items = Hashtbl.create 0;
    ctxt_all_stmts = Hashtbl.create 0;
    ctxt_item_files = crate.Ast.crate_files;
    ctxt_all_lvals = Hashtbl.create 0;
    ctxt_all_defns = Hashtbl.create 0;
    ctxt_lval_to_referent = Hashtbl.create 0;
    ctxt_required_items = crate.Ast.crate_required;
    ctxt_required_syms = crate.Ast.crate_required_syms;

    ctxt_constrs = Hashtbl.create 0;
    ctxt_constr_ids = Hashtbl.create 0;
    ctxt_preconditions = Hashtbl.create 0;
    ctxt_postconditions = Hashtbl.create 0;
    ctxt_prestates = Hashtbl.create 0;
    ctxt_poststates = Hashtbl.create 0;
    ctxt_copy_stmt_is_init = Hashtbl.create 0;
    ctxt_post_stmt_slot_drops = Hashtbl.create 0;
    ctxt_call_lval_params = Hashtbl.create 0;

    ctxt_slot_aliased = Hashtbl.create 0;
    ctxt_slot_is_obj_state = Hashtbl.create 0;
    ctxt_slot_vregs = Hashtbl.create 0;
    ctxt_slot_offsets = Hashtbl.create 0;
    ctxt_frame_sizes = Hashtbl.create 0;
    ctxt_call_sizes = Hashtbl.create 0;

    ctxt_block_is_loop_body = Hashtbl.create 0;
    ctxt_slot_loop_depths = Hashtbl.create 0;
    ctxt_stmt_loop_depths = Hashtbl.create 0;

    ctxt_fn_fixups = Hashtbl.create 0;
    ctxt_block_fixups = Hashtbl.create 0;
    ctxt_file_fixups = Hashtbl.create 0;
    ctxt_spill_fixups = Hashtbl.create 0;
    ctxt_abi = abi;
    ctxt_activate_fixup = new_fixup "activate glue";
    ctxt_yield_fixup = new_fixup "yield glue";
    ctxt_unwind_fixup = new_fixup "unwind glue";
    ctxt_gc_fixup = new_fixup "gc glue";
    ctxt_exit_task_fixup = new_fixup "exit-task glue";

    ctxt_debug_aranges_fixup = new_fixup "debug_aranges section";
    ctxt_debug_pubnames_fixup = new_fixup "debug_pubnames section";
    ctxt_debug_info_fixup = new_fixup "debug_info section";
    ctxt_debug_abbrev_fixup = new_fixup "debug_abbrev section";
    ctxt_debug_line_fixup = new_fixup "debug_line section";
    ctxt_debug_frame_fixup = new_fixup "debug_frame section";

    ctxt_image_base_fixup = new_fixup "loaded image base";
    ctxt_crate_fixup = new_fixup "root crate structure";
    ctxt_file_code = Hashtbl.create 0;
    ctxt_all_item_code = Hashtbl.create 0;
    ctxt_glue_code = Hashtbl.create 0;
    ctxt_data = Hashtbl.create 0;

    ctxt_native_required = Hashtbl.create 0;
    ctxt_native_provided = Hashtbl.create 0;

    ctxt_required_rust_sym_num = Hashtbl.create 0;
    ctxt_required_c_sym_num = Hashtbl.create 0;
    ctxt_required_lib_num = Hashtbl.create 0;

    ctxt_main_fn_fixup =
      (match crate.Ast.crate_main with
           None -> None
         | Some n -> Some (new_fixup (string_of_name n)));

    ctxt_main_name =
      (match crate.Ast.crate_main with
           None -> None
         | Some n -> Some (string_of_name n));
  }
;;

let report_err cx ido str =
  let sess = cx.ctxt_sess in
  let spano = match ido with
      None -> None
    | Some id -> (Session.get_span sess id)
  in
    match spano with
        None ->
          Session.fail sess "Error: %s\n%!" str
      | Some span ->
          Session.fail sess "%s:E:Error: %s\n%!"
            (Session.string_of_span span) str
;;

let bugi (cx:ctxt) (i:node_id) =
  let k s =
    report_err cx (Some i) s;
    failwith s
  in Printf.ksprintf k
;;

(* Convenience accessors. *)

(* resolve an lval reference id to the id of its definition *)
let lval_to_referent (cx:ctxt) (id:node_id) : node_id =
  if Hashtbl.mem cx.ctxt_lval_to_referent id
  then Hashtbl.find cx.ctxt_lval_to_referent id
  else bug () "unresolved lval"
;;

(* resolve an lval reference id to its definition *)
let resolve_lval_id (cx:ctxt) (id:node_id) : defn =
  Hashtbl.find cx.ctxt_all_defns (lval_to_referent cx id)
;;

let referent_is_slot (cx:ctxt) (id:node_id) : bool =
  match Hashtbl.find cx.ctxt_all_defns id with
      DEFN_slot _ -> true
    | _ -> false
;;

let referent_is_item (cx:ctxt) (id:node_id) : bool =
  match Hashtbl.find cx.ctxt_all_defns id with
      DEFN_item _ -> true
    | _ -> false
;;

let rec lval_base_id (lv:Ast.lval) : node_id =
  match lv with
      Ast.LVAL_base nbi -> nbi.id
    | Ast.LVAL_ext (lv, _) -> lval_base_id lv
;;

let get_item (cx:ctxt) (node:node_id) : Ast.mod_item_decl =
  match htab_search cx.ctxt_all_defns node with
      Some (DEFN_item item) -> item
    | Some _ -> bugi cx node "defn is not an item"
    | None -> bugi cx node "missing defn"
;;

let get_slot (cx:ctxt) (node:node_id) : Ast.slot =
  match htab_search cx.ctxt_all_defns node with
      Some (DEFN_slot slot) -> slot
    | Some _ -> bugi cx node "defn is not a slot"
    | None -> bugi cx node "missing defn"
;;

(* coerce an lval reference id to its definition slot *)
let lval_base_to_slot (cx:ctxt) (lval:Ast.lval) : Ast.slot identified =
  let lid = lval_base_id lval in
  let rid = lval_to_referent cx lid in
  let slot = get_slot cx rid in
    { node = slot; id = rid }
;;

let get_stmt_depth (cx:ctxt) (id:node_id) : int =
  Hashtbl.find cx.ctxt_stmt_loop_depths id
;;

let get_slot_depth (cx:ctxt) (id:node_id) : int =
  Hashtbl.find cx.ctxt_slot_loop_depths id
;;

let get_fn_fixup (cx:ctxt) (id:node_id) : fixup =
  if Hashtbl.mem cx.ctxt_fn_fixups id
  then Hashtbl.find cx.ctxt_fn_fixups id
  else bugi cx id "fn without fixup"
;;

let get_framesz (cx:ctxt) (id:node_id) : size =
  if Hashtbl.mem cx.ctxt_frame_sizes id
  then Hashtbl.find cx.ctxt_frame_sizes id
  else bugi cx id "missing framesz"
;;

let get_callsz (cx:ctxt) (id:node_id) : size =
  if Hashtbl.mem cx.ctxt_call_sizes id
  then Hashtbl.find cx.ctxt_call_sizes id
  else bugi cx id "missing callsz"
;;

let rec n_item_ty_params (cx:ctxt) (id:node_id) : int =
  match Hashtbl.find cx.ctxt_all_defns id with
      DEFN_item i -> Array.length i.Ast.decl_params
    | DEFN_obj_fn (oid,_) -> n_item_ty_params cx oid
    | DEFN_obj_drop oid -> n_item_ty_params cx oid
    | DEFN_loop_body fid -> n_item_ty_params cx fid
    | _ -> bugi cx id "n_item_ty_params on non-item"
;;

let item_is_obj_fn (cx:ctxt) (id:node_id) : bool =
  match Hashtbl.find cx.ctxt_all_defns id with
      DEFN_obj_fn _
    | DEFN_obj_drop _ -> true
    | _ -> false
;;

let get_spill (cx:ctxt) (id:node_id) : fixup =
  if Hashtbl.mem cx.ctxt_spill_fixups id
  then Hashtbl.find cx.ctxt_spill_fixups id
  else bugi cx id "missing spill fixup"
;;

let require_native (cx:ctxt) (lib:required_lib) (name:string) : fixup =
  let lib_tab = (htab_search_or_add cx.ctxt_native_required lib
                   (fun _ -> Hashtbl.create 0))
  in
    htab_search_or_add lib_tab name
      (fun _ -> new_fixup ("require: " ^ name))
;;

let provide_native (cx:ctxt) (seg:segment) (name:string) : fixup =
  let seg_tab = (htab_search_or_add cx.ctxt_native_provided seg
                   (fun _ -> Hashtbl.create 0))
  in
    htab_search_or_add seg_tab name
      (fun _ -> new_fixup ("provide: " ^ name))
;;

let provide_existing_native
    (cx:ctxt)
    (seg:segment)
    (name:string)
    (fix:fixup)
    : unit =
  let seg_tab = (htab_search_or_add cx.ctxt_native_provided seg
                   (fun _ -> Hashtbl.create 0))
  in
    htab_put seg_tab name fix
;;

let slot_ty (s:Ast.slot) : Ast.ty =
  match s.Ast.slot_ty with
      Some t -> t
    | None -> bug () "untyped slot"
;;

let fn_output_ty (fn_ty:Ast.ty) : Ast.ty =
  match fn_ty with
      Ast.TY_fn ({ Ast.sig_output_slot = slot }, _) ->
        begin
          match slot.Ast.slot_ty with
              Some ty -> ty
            | None -> bug () "function has untyped output slot"
        end
    | _ -> bug () "fn_output_ty on non-TY_fn"
;;

(* name of tag constructor function -> name for indexing in the ty_tag *)
let rec tag_ctor_name_to_tag_name (name:Ast.name) : Ast.name =
  match name with
      Ast.NAME_base (Ast.BASE_ident _) -> name
    | Ast.NAME_base (Ast.BASE_app (id, _)) ->
        Ast.NAME_base (Ast.BASE_ident id)

    | Ast.NAME_ext (_, Ast.COMP_ident id)
    | Ast.NAME_ext (_, Ast.COMP_app (id, _)) ->
        Ast.NAME_base (Ast.BASE_ident id)

    | _ -> bug () "tag_or_iso_ty_tup_by_name with non-tag-ctor name"
;;

let tag_or_iso_ty_tup_by_name (ty:Ast.ty) (name:Ast.name) : Ast.ty_tup =
  let tagname = tag_ctor_name_to_tag_name name in
    match ty with
        Ast.TY_tag tags ->
          Hashtbl.find tags tagname
      | Ast.TY_iso { Ast.iso_index = i; Ast.iso_group = gp } ->
          Hashtbl.find gp.(i) tagname
      | _ ->
          bug () "tag_or_iso_ty_tup_by_name called with non-tag or -iso type"
;;

let defn_is_slot (d:defn) : bool =
  match d with
      DEFN_slot _ -> true
    | _ -> false
;;

let defn_is_item (d:defn) : bool =
  match d with
      DEFN_item _ -> true
    | _ -> false
;;

let slot_is_obj_state (cx:ctxt) (sid:node_id) : bool =
  Hashtbl.mem cx.ctxt_slot_is_obj_state sid
;;


(* determines whether d defines a statically-known value *)
let defn_is_static (d:defn) : bool =
  not (defn_is_slot d)
;;

let defn_is_callable (d:defn) : bool =
  match d with
      DEFN_slot { Ast.slot_ty = Some Ast.TY_fn _ }
    | DEFN_item { Ast.decl_item = (Ast.MOD_ITEM_fn _ ) } -> true
    | _ -> false
;;

(* Constraint manipulation. *)

let rec apply_names_to_carg_path
    (names:(Ast.name_base option) array)
    (cp:Ast.carg_path)
    : Ast.carg_path =
  match cp with
      Ast.CARG_ext (Ast.CARG_base Ast.BASE_formal,
                    Ast.COMP_idx i) ->
        begin
          match names.(i) with
              Some nb ->
                Ast.CARG_base (Ast.BASE_named nb)
            | None -> bug () "Indexing off non-named carg"
        end
    | Ast.CARG_ext (cp', e) ->
        Ast.CARG_ext (apply_names_to_carg_path names cp', e)
    | _ -> cp
;;

let apply_names_to_carg
    (names:(Ast.name_base option) array)
    (carg:Ast.carg)
    : Ast.carg =
  match carg with
      Ast.CARG_path cp ->
        Ast.CARG_path (apply_names_to_carg_path names cp)
    | Ast.CARG_lit _ -> carg
;;

let apply_names_to_constr
    (names:(Ast.name_base option) array)
    (constr:Ast.constr)
    : Ast.constr =
  { constr with
      Ast.constr_args =
      Array.map (apply_names_to_carg names) constr.Ast.constr_args }
;;

let atoms_to_names (atoms:Ast.atom array)
    : (Ast.name_base option) array =
  Array.map
    begin
      fun atom ->
        match atom with
            Ast.ATOM_lval (Ast.LVAL_base nbi) -> Some nbi.node
          | _ -> None
    end
    atoms
;;

let rec lval_to_name (lv:Ast.lval) : Ast.name =
  match lv with
      Ast.LVAL_base { node = nb } ->
        Ast.NAME_base nb
    | Ast.LVAL_ext (lv, lv_comp) ->
        let comp =
          begin
            match lv_comp with
                Ast.COMP_named comp -> comp
              | _ -> bug ()
                  "lval_to_name with lval that contains non-name components"
          end
        in
          Ast.NAME_ext (lval_to_name lv, comp)
;;

let rec lval_slots (cx:ctxt) (lv:Ast.lval) : node_id array =
  match lv with
      Ast.LVAL_base nbi ->
        let referent = lval_to_referent cx nbi.id in
          if referent_is_slot cx referent
          then [| referent |]
          else [| |]
    | Ast.LVAL_ext (lv, Ast.COMP_named _)
    | Ast.LVAL_ext (lv, Ast.COMP_deref) -> lval_slots cx lv
    | Ast.LVAL_ext (lv, Ast.COMP_atom a) ->
        Array.append (lval_slots cx lv) (atom_slots cx a)

and atom_slots (cx:ctxt) (a:Ast.atom) : node_id array =
  match a with
      Ast.ATOM_literal _ -> [| |]
    | Ast.ATOM_lval lv -> lval_slots cx lv
;;

let lval_option_slots (cx:ctxt) (lv:Ast.lval option) : node_id array =
  match lv with
      None -> [| |]
    | Some lv -> lval_slots cx lv
;;

let resolve_lval (cx:ctxt) (lv:Ast.lval) : defn =
  resolve_lval_id cx (lval_base_id lv)
;;

let atoms_slots (cx:ctxt) (az:Ast.atom array) : node_id array =
  Array.concat (List.map (atom_slots cx) (Array.to_list az))
;;

let tup_inputs_slots (cx:ctxt) (az:Ast.tup_input array) : node_id array =
  Array.concat (List.map (atom_slots cx) (Array.to_list az))
;;

let rec_inputs_slots (cx:ctxt)
    (inputs:Ast.rec_input array) : node_id array =
  Array.concat (List.map
                  (fun (_, atom) -> atom_slots cx atom)
                  (Array.to_list inputs))
;;

let expr_slots (cx:ctxt) (e:Ast.expr) : node_id array =
    match e with
        Ast.EXPR_binary (_, a, b) ->
          Array.append (atom_slots cx a) (atom_slots cx b)
      | Ast.EXPR_unary (_, u) -> atom_slots cx u
      | Ast.EXPR_atom a -> atom_slots cx a
;;


(* Type extraction. *)

let local_slot_full mut ty : Ast.slot =
  let ty =
    if mut
    then Ast.TY_mutable ty
    else ty
  in
    { Ast.slot_mode = Ast.MODE_local;
      Ast.slot_ty = Some ty }
;;

let box_slot_full mut ty : Ast.slot =
  let ty =
    match ty with
        Ast.TY_box _ -> ty
      | _ -> Ast.TY_box ty
  in
  let ty =
    if mut
    then Ast.TY_mutable ty
    else ty
  in
  { Ast.slot_mode = Ast.MODE_local;
    Ast.slot_ty = Some ty }
;;

let local_slot ty : Ast.slot = local_slot_full false ty
;;

let box_slot ty : Ast.slot = box_slot_full false ty
;;


(* General folds of Ast.ty. *)

type ('ty, 'tys, 'slot, 'slots, 'tag) ty_fold =
    {
      (* Functions that correspond to local nodes in Ast.ty. *)
      ty_fold_slot : (Ast.mode * 'ty) -> 'slot;
      ty_fold_slots : ('slot array) -> 'slots;
      ty_fold_tys : ('ty array) -> 'tys;
      ty_fold_tags : (Ast.name, 'tys) Hashtbl.t -> 'tag;

      (* Functions that correspond to the Ast.ty constructors. *)
      ty_fold_any: unit -> 'ty;
      ty_fold_nil : unit -> 'ty;
      ty_fold_bool : unit -> 'ty;
      ty_fold_mach : ty_mach -> 'ty;
      ty_fold_int : unit -> 'ty;
      ty_fold_uint : unit -> 'ty;
      ty_fold_char : unit -> 'ty;
      ty_fold_str : unit -> 'ty;
      ty_fold_tup : 'tys -> 'ty;
      ty_fold_vec : 'ty -> 'ty;
      ty_fold_rec : (Ast.ident * 'ty) array -> 'ty;
      ty_fold_tag : 'tag -> 'ty;
      ty_fold_iso : (int * 'tag array) -> 'ty;
      ty_fold_idx : int -> 'ty;
      ty_fold_fn : (('slots * Ast.constrs * 'slot) * Ast.ty_fn_aux) -> 'ty;
      ty_fold_obj : (Ast.effect
                     * (Ast.ident, (('slots * Ast.constrs * 'slot) *
                                      Ast.ty_fn_aux)) Hashtbl.t) -> 'ty;
      ty_fold_chan : 'ty -> 'ty;
      ty_fold_port : 'ty -> 'ty;
      ty_fold_task : unit -> 'ty;
      ty_fold_native : opaque_id -> 'ty;
      ty_fold_param : (int * Ast.effect) -> 'ty;
      ty_fold_named : Ast.name -> 'ty;
      ty_fold_type : unit -> 'ty;
      ty_fold_box : 'ty -> 'ty;
      ty_fold_mutable : 'ty -> 'ty;
      ty_fold_constrained : ('ty * Ast.constrs) -> 'ty }
;;

let rec fold_ty
    (f:('ty, 'tys, 'slot, 'slots, 'tag) ty_fold)
    (ty:Ast.ty)
    : 'ty =
  let fold_slot (s:Ast.slot) : 'slot =
    f.ty_fold_slot (s.Ast.slot_mode,
                    fold_ty f (slot_ty s))
  in

  let fold_slots (slots:Ast.slot array) : 'slots =
    f.ty_fold_slots (Array.map fold_slot slots)
  in

  let fold_tys (tys:Ast.ty array) : 'tys =
    f.ty_fold_tys (Array.map (fold_ty f) tys)
  in

  let fold_tags (ttag:Ast.ty_tag) : 'tag =
    f.ty_fold_tags (htab_map ttag (fun k v -> (k, fold_tys v)))
  in

  let fold_sig tsig =
    (fold_slots tsig.Ast.sig_input_slots,
     tsig.Ast.sig_input_constrs,
     fold_slot tsig.Ast.sig_output_slot)
  in
  let fold_obj fns =
    htab_map fns (fun i (tsig, taux) -> (i, (fold_sig tsig, taux)))
  in
    match ty with
    Ast.TY_any -> f.ty_fold_any ()
  | Ast.TY_nil -> f.ty_fold_nil ()
  | Ast.TY_bool -> f.ty_fold_bool ()
  | Ast.TY_mach m -> f.ty_fold_mach m
  | Ast.TY_int -> f.ty_fold_int ()
  | Ast.TY_uint -> f.ty_fold_uint ()
  | Ast.TY_char -> f.ty_fold_char ()
  | Ast.TY_str -> f.ty_fold_str ()

  | Ast.TY_tup t -> f.ty_fold_tup (fold_tys t)
  | Ast.TY_vec t -> f.ty_fold_vec (fold_ty f t)
  | Ast.TY_rec r ->
      f.ty_fold_rec (Array.map (fun (k,v) -> (k,fold_ty f v)) r)

  | Ast.TY_tag tt -> f.ty_fold_tag (fold_tags tt)
  | Ast.TY_iso ti ->
      f.ty_fold_iso (ti.Ast.iso_index,
                     (Array.map fold_tags ti.Ast.iso_group))
  | Ast.TY_idx i -> f.ty_fold_idx i

  | Ast.TY_fn (tsig,taux) -> f.ty_fold_fn (fold_sig tsig, taux)
  | Ast.TY_chan t -> f.ty_fold_chan (fold_ty f t)
  | Ast.TY_port t -> f.ty_fold_port (fold_ty f t)

  | Ast.TY_obj (eff,t) -> f.ty_fold_obj (eff, (fold_obj t))
  | Ast.TY_task -> f.ty_fold_task ()

  | Ast.TY_native x -> f.ty_fold_native x
  | Ast.TY_param x -> f.ty_fold_param x
  | Ast.TY_named n -> f.ty_fold_named n
  | Ast.TY_type -> f.ty_fold_type ()

  | Ast.TY_box t -> f.ty_fold_box (fold_ty f t)
  | Ast.TY_mutable t -> f.ty_fold_mutable (fold_ty f t)

  | Ast.TY_constrained (t, constrs) ->
      f.ty_fold_constrained (fold_ty f t, constrs)

;;

type 'a simple_ty_fold = ('a, 'a, 'a, 'a, 'a) ty_fold
;;

let ty_fold_default (default:'a) : 'a simple_ty_fold =
    { ty_fold_tys = (fun _ -> default);
      ty_fold_slot = (fun _ -> default);
      ty_fold_slots = (fun _ -> default);
      ty_fold_tags = (fun _ -> default);
      ty_fold_any = (fun _ -> default);
      ty_fold_nil = (fun _ -> default);
      ty_fold_bool = (fun _ -> default);
      ty_fold_mach = (fun _ -> default);
      ty_fold_int = (fun _ -> default);
      ty_fold_uint = (fun _ -> default);
      ty_fold_char = (fun _ -> default);
      ty_fold_str = (fun _ -> default);
      ty_fold_tup = (fun _ -> default);
      ty_fold_vec = (fun _ -> default);
      ty_fold_rec = (fun _ -> default);
      ty_fold_tag = (fun _ -> default);
      ty_fold_iso = (fun _ -> default);
      ty_fold_idx = (fun _ -> default);
      ty_fold_fn = (fun _ -> default);
      ty_fold_obj = (fun _ -> default);
      ty_fold_chan = (fun _ -> default);
      ty_fold_port = (fun _ -> default);
      ty_fold_task = (fun _ -> default);
      ty_fold_native = (fun _ -> default);
      ty_fold_param = (fun _ -> default);
      ty_fold_named = (fun _ -> default);
      ty_fold_type = (fun _ -> default);
      ty_fold_box = (fun _ -> default);
      ty_fold_mutable = (fun _ -> default);
      ty_fold_constrained = (fun _ -> default) }
;;

let ty_fold_rebuild (id:Ast.ty -> Ast.ty)
    : (Ast.ty, Ast.ty array, Ast.slot, Ast.slot array, Ast.ty_tag) ty_fold =
  let rebuild_fn ((islots, constrs, oslot), aux) =
    ({ Ast.sig_input_slots = islots;
       Ast.sig_input_constrs = constrs;
       Ast.sig_output_slot = oslot }, aux)
  in
    {
    ty_fold_tys = (fun ts -> ts);
    ty_fold_slot = (fun (mode, t) ->
                      { Ast.slot_mode = mode;
                        Ast.slot_ty = Some t });
    ty_fold_slots = (fun slots -> slots);
    ty_fold_tags = (fun htab -> htab);
    ty_fold_any = (fun _ -> id Ast.TY_any);
    ty_fold_nil = (fun _ -> id Ast.TY_nil);
    ty_fold_bool = (fun _ -> id Ast.TY_bool);
    ty_fold_mach = (fun m -> id (Ast.TY_mach m));
    ty_fold_int = (fun _ -> id Ast.TY_int);
    ty_fold_uint = (fun _ -> id Ast.TY_uint);
    ty_fold_char = (fun _ -> id Ast.TY_char);
    ty_fold_str = (fun _ -> id Ast.TY_str);
    ty_fold_tup =  (fun slots -> id (Ast.TY_tup slots));
    ty_fold_vec = (fun t -> id (Ast.TY_vec t));
    ty_fold_rec = (fun entries -> id (Ast.TY_rec entries));
    ty_fold_tag = (fun tag -> id (Ast.TY_tag tag));
    ty_fold_iso = (fun (i, tags) -> id (Ast.TY_iso { Ast.iso_index = i;
                                                     Ast.iso_group = tags }));
    ty_fold_idx = (fun i -> id (Ast.TY_idx i));
    ty_fold_fn = (fun t -> id (Ast.TY_fn (rebuild_fn t)));
    ty_fold_obj = (fun (eff,fns) ->
                     id (Ast.TY_obj
                           (eff, (htab_map fns
                                    (fun id fn -> (id, rebuild_fn fn))))));
    ty_fold_chan = (fun t -> id (Ast.TY_chan t));
    ty_fold_port = (fun t -> id (Ast.TY_port t));
    ty_fold_task = (fun _ -> id Ast.TY_task);
    ty_fold_native = (fun oid -> id (Ast.TY_native oid));
    ty_fold_param = (fun (i, mut) -> id (Ast.TY_param (i, mut)));
    ty_fold_named = (fun n -> id (Ast.TY_named n));
    ty_fold_type = (fun _ -> id (Ast.TY_type));
    ty_fold_box = (fun t -> id (Ast.TY_box t));
    ty_fold_mutable = (fun t -> id (Ast.TY_mutable t));
    ty_fold_constrained = (fun (t, constrs) ->
                             id (Ast.TY_constrained (t, constrs))) }
;;

let rebuild_ty_under_params
    (ty:Ast.ty)
    (params:Ast.ty_param array)
    (args:Ast.ty array)
    (resolve_names:bool)
    : Ast.ty =
  if (Array.length params) <> (Array.length args)
  then err None "mismatched type-params"
  else
    let nmap = Hashtbl.create (Array.length args) in
    let pmap = Hashtbl.create (Array.length args) in
    let _ =
      Array.iteri
        begin
          fun i (ident, param) ->
            htab_put pmap (Ast.TY_param param) args.(i);
            if resolve_names
            then
              htab_put nmap ident args.(i)
        end
        params
    in
    let rec rebuild_ty t =
      let base = ty_fold_rebuild (fun t -> t) in
      let ty_fold_param (i, mut) =
        let param = Ast.TY_param (i, mut) in
          match htab_search pmap param with
              None -> param
            | Some arg -> arg
      in
      let ty_fold_named n =
        let rec rebuild_name n =
          match n with
              Ast.NAME_base nb ->
                Ast.NAME_base (rebuild_name_base nb)
            | Ast.NAME_ext (n, nc) ->
                Ast.NAME_ext (rebuild_name n,
                              rebuild_name_component nc)

        and rebuild_name_base nb =
          match nb with
              Ast.BASE_ident i ->
                Ast.BASE_ident i
            | Ast.BASE_temp t ->
                Ast.BASE_temp t
            | Ast.BASE_app (i, tys) ->
                Ast.BASE_app (i, rebuild_tys tys)

        and rebuild_name_component nc =
          match nc with
              Ast.COMP_ident i ->
                Ast.COMP_ident i
            | Ast.COMP_app (i, tys) ->
                Ast.COMP_app (i, rebuild_tys tys)
            | Ast.COMP_idx i ->
                Ast.COMP_idx i

        and rebuild_tys tys =
          Array.map (fun t -> rebuild_ty t) tys
        in
        let n = rebuild_name n in
          match n with
              Ast.NAME_base (Ast.BASE_ident id)
                when resolve_names ->
                  begin
                    match htab_search nmap id with
                        None -> Ast.TY_named n
                      | Some arg -> arg
                  end
            | _ -> Ast.TY_named n
      in
      let fold =
        { base with
            ty_fold_param = ty_fold_param;
            ty_fold_named = ty_fold_named;
        }
      in
        fold_ty fold t
    in
      rebuild_ty ty
;;

let associative_binary_op_ty_fold
    (default:'a)
    (fn:'a -> 'a -> 'a)
    : 'a simple_ty_fold =
  let base = ty_fold_default default in
  let reduce ls =
    match ls with
        [] -> default
      | x::xs -> List.fold_left fn x xs
  in
  let reduce_fn ((islots, _, oslot), _) =
    fn islots oslot
  in
    { base with
        ty_fold_tys = (fun ts -> reduce (Array.to_list ts));
        ty_fold_slots = (fun slots -> reduce (Array.to_list slots));
        ty_fold_slot = (fun (_, a) -> a);
        ty_fold_tags = (fun tab -> reduce (htab_vals tab));
        ty_fold_tup = (fun a -> a);
        ty_fold_vec = (fun a -> a);
        ty_fold_rec = (fun sz ->
                         reduce (Array.to_list
                                   (Array.map (fun (_, s) -> s) sz)));
        ty_fold_tag = (fun a -> a);
        ty_fold_iso = (fun (_,iso) -> reduce (Array.to_list iso));
        ty_fold_fn = reduce_fn;
        ty_fold_obj = (fun (_,fns) ->
                         reduce (List.map reduce_fn (htab_vals fns)));
        ty_fold_chan = (fun a -> a);
        ty_fold_port = (fun a -> a);
        ty_fold_box = (fun a -> a);
        ty_fold_mutable = (fun a -> a);
        ty_fold_constrained = (fun (a, _) -> a) }

let ty_fold_bool_and (default:bool) : bool simple_ty_fold =
  associative_binary_op_ty_fold default (fun a b -> a & b)
;;

let ty_fold_bool_or (default:bool) : bool simple_ty_fold =
  associative_binary_op_ty_fold default (fun a b -> a || b)
;;

let ty_fold_int_max (default:int) : int simple_ty_fold =
  associative_binary_op_ty_fold default (fun a b -> max a b)
;;

let ty_fold_list_concat _ : ('a list) simple_ty_fold =
  associative_binary_op_ty_fold [] (fun a b -> a @ b)
;;

let type_is_structured (t:Ast.ty) : bool =
  let fold = ty_fold_bool_or false in
  let fold = { fold with
                 ty_fold_tup = (fun _ -> true);
                 ty_fold_vec = (fun _ -> true);
                 ty_fold_rec = (fun _ -> true);
                 ty_fold_tag = (fun _ -> true);
                 ty_fold_iso = (fun _ -> true);
                 ty_fold_idx = (fun _ -> true);
                 ty_fold_fn = (fun _ -> true);
                 ty_fold_obj = (fun _ -> true) }
  in
    fold_ty fold t
;;

(* Effect analysis. *)
let effect_le x y =
  match (x,y) with
      (Ast.UNSAFE, _) -> true
    | (Ast.STATE, Ast.PURE) -> true
    | (Ast.STATE, Ast.IO) -> true
    | (Ast.STATE, Ast.STATE) -> true
    | (Ast.IO, Ast.PURE) -> true
    | (Ast.IO, Ast.IO) -> true
    | (Ast.PURE, Ast.PURE) -> true
    | _ -> false
;;

let lower_effect_of x y =
  if effect_le x y then x else y
;;

let type_effect (t:Ast.ty) : Ast.effect =
  let fold_mutable _ = Ast.STATE in
  let fold = associative_binary_op_ty_fold Ast.PURE lower_effect_of in
  let fold = { fold with ty_fold_mutable = fold_mutable } in
    fold_ty fold t
;;

let type_has_state (t:Ast.ty) : bool =
  effect_le (type_effect t) Ast.STATE
;;


(* Various type analyses. *)

let is_prim_type (t:Ast.ty) : bool =
  match t with
      Ast.TY_int
    | Ast.TY_uint
    | Ast.TY_char
    | Ast.TY_mach _
    | Ast.TY_bool -> true
    | _ -> false
;;

let type_contains_chan (t:Ast.ty) : bool =
  let fold_chan _ = true in
  let fold = ty_fold_bool_or false in
  let fold = { fold with ty_fold_chan = fold_chan } in
    fold_ty fold t
;;


let type_is_unsigned_2s_complement t =
  match t with
      Ast.TY_mach TY_u8
    | Ast.TY_mach TY_u16
    | Ast.TY_mach TY_u32
    | Ast.TY_mach TY_u64
    | Ast.TY_char
    | Ast.TY_uint
    | Ast.TY_bool -> true
    | _ -> false
;;


let type_is_signed_2s_complement t =
  match t with
      Ast.TY_mach TY_i8
    | Ast.TY_mach TY_i16
    | Ast.TY_mach TY_i32
    | Ast.TY_mach TY_i64
    | Ast.TY_int -> true
    | _ -> false
;;


let type_is_2s_complement t =
  (type_is_unsigned_2s_complement t)
  || (type_is_signed_2s_complement t)
;;

let n_used_type_params t =
  let fold_param (i,_) = i+1 in
  let fold = ty_fold_int_max 0 in
  let fold = { fold with ty_fold_param = fold_param } in
    fold_ty fold t
;;



let check_concrete params thing =
  if Array.length params = 0
  then thing
  else bug () "unhandled parametric binding"
;;

let rec strip_mutable_or_constrained_ty (t:Ast.ty) : Ast.ty =
  match t with
      Ast.TY_mutable t
    | Ast.TY_constrained (t, _) -> strip_mutable_or_constrained_ty t
    | _ -> t
;;

let rec simplified_ty (t:Ast.ty) : Ast.ty =
  match strip_mutable_or_constrained_ty t with
      Ast.TY_box t -> simplified_ty t
    | t -> t
;;

let rec project_type
    (base_ty:Ast.ty)
    (comp:Ast.lval_component)
    : Ast.ty =
  match (base_ty, comp) with
      (Ast.TY_rec elts, Ast.COMP_named (Ast.COMP_ident id)) ->
        begin
          match atab_search elts id with
              Some ty -> ty
            | None -> err None "unknown record-member '%s'" id
        end

    | (Ast.TY_tup elts, Ast.COMP_named (Ast.COMP_idx i)) ->
        if 0 <= i && i < (Array.length elts)
        then elts.(i)
        else err None "out-of-range tuple index %d" i

    | (Ast.TY_vec ty, Ast.COMP_atom _) -> ty
    | (Ast.TY_str, Ast.COMP_atom _) -> (Ast.TY_mach TY_u8)
    | (Ast.TY_obj (_, fns), Ast.COMP_named (Ast.COMP_ident id)) ->
        (Ast.TY_fn (Hashtbl.find fns id))

    | (Ast.TY_box t, Ast.COMP_deref) -> t

    (* Box, mutable and constrained are transparent to the
     * other lval-ext forms: x.y and x.(y).
     *)
    | (Ast.TY_box t, _)
    | (Ast.TY_mutable t, _)
    | (Ast.TY_constrained (t, _), _) -> project_type t comp

    | (_,_) ->
        bug ()
          "project_ty: bad lval-ext: %s"
          (match comp with
               Ast.COMP_atom at ->
                 Printf.sprintf "%a.(%a)"
                   Ast.sprintf_ty base_ty
                   Ast.sprintf_atom at
             | Ast.COMP_named nc ->
                 Printf.sprintf "%a.%a"
                   Ast.sprintf_ty base_ty
                   Ast.sprintf_name_component nc
             | Ast.COMP_deref ->
                 Printf.sprintf "*(%a)"
                   Ast.sprintf_ty base_ty)
;;

let exports_permit (view:Ast.mod_view) (ident:Ast.ident) : bool =
  (Hashtbl.mem view.Ast.view_exports Ast.EXPORT_all_decls) ||
    (Hashtbl.mem view.Ast.view_exports (Ast.EXPORT_ident ident))
;;

(* NB: this will fail if lval is not an item. *)
let rec lval_item (cx:ctxt) (lval:Ast.lval) : Ast.mod_item =
  match lval with
      Ast.LVAL_base nb ->
        begin
          let referent = lval_to_referent cx nb.id in
            match htab_search cx.ctxt_all_defns referent with
                Some (DEFN_item item) -> {node=item; id=referent}
              | _ -> err (Some (lval_base_id lval))
                  "lval does not name an item"
        end
    | Ast.LVAL_ext (base, comp) ->
        let base_item = lval_item cx base in
        match base_item.node.Ast.decl_item with
            Ast.MOD_ITEM_mod (view, items) ->
              begin
                let i, args =
                  match comp with
                      Ast.COMP_named (Ast.COMP_ident i) -> (i, [||])
                    | Ast.COMP_named (Ast.COMP_app (i, args)) -> (i, args)
                    | _ ->
                        bug ()
                          "unhandled lval-component in '%a' in lval_item"
                          Ast.sprintf_lval lval
                in
                  match htab_search items i with
                    | Some sub when exports_permit view i ->
                        assert
                          ((Array.length sub.node.Ast.decl_params) =
                              (Array.length args));
                        check_concrete base_item.node.Ast.decl_params sub
                    | _ -> err (Some (lval_base_id lval))
                        "unknown module item '%s'" i
              end
          | _ -> err (Some (lval_base_id lval))
              "lval base %a does not name a module" Ast.sprintf_lval base
;;

let lval_is_slot (cx:ctxt) (lval:Ast.lval) : bool =
  match resolve_lval cx lval with
      DEFN_slot _ -> true
    | _ -> false
;;

let lval_is_item (cx:ctxt) (lval:Ast.lval) : bool =
  match resolve_lval cx lval with
      DEFN_item _ -> true
    | _ -> false
;;

let lval_is_direct_fn (cx:ctxt) (lval:Ast.lval) : bool =
  let defn = resolve_lval cx lval in
    (defn_is_static defn) && (defn_is_callable defn)
;;

let lval_is_direct_mod (cx:ctxt) (lval:Ast.lval) : bool =
  let defn = resolve_lval cx lval in
    if not (defn_is_static defn)
    then false
    else
      match defn with
          DEFN_item { Ast.decl_item = Ast.MOD_ITEM_mod _ } -> true
        | _ -> false
;;

(* 
 * FIXME: this function is a bad idea and exists only as a workaround
 * for other logic that is even worse. Untangle.
 *)
let rec project_lval_ty_from_slot (cx:ctxt) (lval:Ast.lval) : Ast.ty =
  match lval with
      Ast.LVAL_base nbi ->
        let referent = lval_to_referent cx nbi.id in
          if lval_is_slot cx lval
          then slot_ty (get_slot cx referent)
          else Hashtbl.find cx.ctxt_all_item_types nbi.id
    | Ast.LVAL_ext (base, comp) ->
        let base_ty = project_lval_ty_from_slot cx base in
          project_type base_ty comp
;;


let lval_ty (cx:ctxt) (lval:Ast.lval) : Ast.ty =
  (*
    FIXME: The correct definition of this function is just: 
    
       Hashtbl.find cx.ctxt_all_lval_types (lval_base_id lval)
    
    However, since the typechecker is not presently handling
    every stmt, we have a fallback mode to "pick out the slot
    type and hope for the best".
  *)
  match htab_search cx.ctxt_all_lval_types (lval_base_id lval) with
      Some t -> t
    | None -> project_lval_ty_from_slot cx lval
;;

let lval_is_static (cx:ctxt) (lval:Ast.lval) : bool =
  defn_is_static (resolve_lval cx lval)
;;

let lval_is_callable (cx:ctxt) (lval:Ast.lval) : bool =
  defn_is_callable (resolve_lval cx lval)
;;

let lval_is_obj_vtbl (cx:ctxt) (lval:Ast.lval) : bool =
  if lval_is_slot cx lval
  then
    match lval with
        Ast.LVAL_ext (base, _) ->
          begin
            match (simplified_ty (project_lval_ty_from_slot cx base)) with
                Ast.TY_obj _ -> true
              | _ -> false
          end
      | _ -> false
  else false
;;

let rec atom_type (cx:ctxt) (at:Ast.atom) : Ast.ty =
  match at with
      Ast.ATOM_literal {node=(Ast.LIT_int _); id=_} -> Ast.TY_int
    | Ast.ATOM_literal {node=(Ast.LIT_uint _); id=_} -> Ast.TY_uint
    | Ast.ATOM_literal {node=(Ast.LIT_bool _); id=_} -> Ast.TY_bool
    | Ast.ATOM_literal {node=(Ast.LIT_char _); id=_} -> Ast.TY_char
    | Ast.ATOM_literal {node=(Ast.LIT_nil); id=_} -> Ast.TY_nil
    | Ast.ATOM_literal {node=(Ast.LIT_mach (m,_,_)); id=_} -> Ast.TY_mach m
    | Ast.ATOM_lval lv -> lval_ty cx lv
;;

let expr_type (cx:ctxt) (e:Ast.expr) : Ast.ty =
  match e with
      Ast.EXPR_binary (op, a, _) ->
        begin
          match op with
              Ast.BINOP_eq | Ast.BINOP_ne | Ast.BINOP_lt  | Ast.BINOP_le
            | Ast.BINOP_ge | Ast.BINOP_gt -> Ast.TY_bool
            | _ -> atom_type cx a
        end
    | Ast.EXPR_unary (Ast.UNOP_not, _) -> Ast.TY_bool
    | Ast.EXPR_unary (_, a) -> atom_type cx a
    | Ast.EXPR_atom a -> atom_type cx a
;;

(* Mappings between mod items and their respective types. *)

let arg_slots (slots:Ast.header_slots) : Ast.slot array =
  Array.map (fun (sid,_) -> sid.node) slots
;;

let tup_slots (slots:Ast.header_tup) : Ast.slot array =
  Array.map (fun sid -> sid.node) slots
;;

let ty_fn_of_fn (fn:Ast.fn) : Ast.ty_fn =
  ({ Ast.sig_input_slots = arg_slots fn.Ast.fn_input_slots;
     Ast.sig_input_constrs = fn.Ast.fn_input_constrs;
     Ast.sig_output_slot = fn.Ast.fn_output_slot.node },
   fn.Ast.fn_aux )
;;

let ty_obj_of_obj (obj:Ast.obj) : Ast.ty_obj =
  (obj.Ast.obj_effect,
   htab_map obj.Ast.obj_fns (fun i f -> (i, ty_fn_of_fn f.node)))
;;

let ty_of_mod_item ((*inside*)_:bool) (item:Ast.mod_item) : Ast.ty =
  match item.node.Ast.decl_item with
      Ast.MOD_ITEM_type _ -> Ast.TY_type
    | Ast.MOD_ITEM_fn f -> (Ast.TY_fn (ty_fn_of_fn f))
    | Ast.MOD_ITEM_mod _ -> bug () "Semant.ty_of_mod_item on mod"
    | Ast.MOD_ITEM_obj ob ->
        let taux = { Ast.fn_effect = Ast.PURE;
                     Ast.fn_is_iter = false }
        in
        let tobj = Ast.TY_obj (ty_obj_of_obj ob) in
        let tsig = { Ast.sig_input_slots = arg_slots ob.Ast.obj_state;
                     Ast.sig_input_constrs = ob.Ast.obj_constrs;
                     Ast.sig_output_slot = local_slot tobj }
        in
          (Ast.TY_fn (tsig, taux))

    | Ast.MOD_ITEM_tag (htup, ttag, _) ->
        let taux = { Ast.fn_effect = Ast.PURE;
                     Ast.fn_is_iter = false }
        in
        let tsig = { Ast.sig_input_slots = tup_slots htup;
                     Ast.sig_input_constrs = [| |];
                     Ast.sig_output_slot = local_slot (Ast.TY_tag ttag) }
        in
          (Ast.TY_fn (tsig, taux))
;;

(* Scopes and the visitor that builds them. *)

type scope =
    SCOPE_block of node_id
  | SCOPE_mod_item of Ast.mod_item
  | SCOPE_obj_fn of (Ast.fn identified)
  | SCOPE_crate of Ast.crate
;;

let id_of_scope (sco:scope) : node_id =
  match sco with
      SCOPE_block id -> id
    | SCOPE_mod_item i -> i.id
    | SCOPE_obj_fn f -> f.id
    | SCOPE_crate c -> c.id
;;

let scope_stack_managing_visitor
    (scopes:(scope list) ref)
    (inner:Walk.visitor)
    : Walk.visitor =
  let push s =
    scopes := s :: (!scopes)
  in
  let pop _ =
    scopes := List.tl (!scopes)
  in
  let visit_block_pre b =
    push (SCOPE_block b.id);
    inner.Walk.visit_block_pre b
  in
  let visit_block_post b =
    inner.Walk.visit_block_post b;
    pop();
  in
  let visit_mod_item_pre n p i =
    push (SCOPE_mod_item i);
    inner.Walk.visit_mod_item_pre n p i
  in
  let visit_mod_item_post n p i =
    inner.Walk.visit_mod_item_post n p i;
    pop();
  in
  let visit_obj_fn_pre obj ident fn =
    push (SCOPE_obj_fn fn);
    inner.Walk.visit_obj_fn_pre obj ident fn
  in
  let visit_obj_fn_post obj ident fn =
    inner.Walk.visit_obj_fn_post obj ident fn;
    pop();
  in
  let visit_crate_pre c =
    push (SCOPE_crate c);
    inner.Walk.visit_crate_pre c
  in
  let visit_crate_post c =
    inner.Walk.visit_crate_post c;
    pop()
  in
    { inner with
        Walk.visit_block_pre = visit_block_pre;
        Walk.visit_block_post = visit_block_post;
        Walk.visit_mod_item_pre = visit_mod_item_pre;
        Walk.visit_mod_item_post = visit_mod_item_post;
        Walk.visit_obj_fn_pre = visit_obj_fn_pre;
        Walk.visit_obj_fn_post = visit_obj_fn_post;
        Walk.visit_crate_pre = visit_crate_pre;
        Walk.visit_crate_post = visit_crate_post; }
;;

let unreferenced_required_item_ignoring_visitor
    (cx:ctxt)
    (inner:Walk.visitor)
    : Walk.visitor =

  let inhibition = ref 0 in

  let directly_inhibited i =
    (Hashtbl.mem cx.ctxt_required_items i.id) &&
      (not (Hashtbl.mem cx.ctxt_node_referenced i.id))
  in

  let indirectly_inhibited _ =
    (!inhibition) <> 0
  in

  let should_visit i =
    not ((directly_inhibited i) || (indirectly_inhibited()))
  in

  let inhibit_pre i =
    if directly_inhibited i
    then incr inhibition
  in

  let inhibit_post i =
    if directly_inhibited i
    then decr inhibition
  in

  let visit_mod_item_pre n p i =
    if should_visit i
    then inner.Walk.visit_mod_item_pre n p i;
    inhibit_pre i
  in

  let visit_mod_item_post n p i =
    if should_visit i
    then inner.Walk.visit_mod_item_post n p i;
    inhibit_post i
  in

  let visit_obj_fn_pre oid ident fn =
    if not (indirectly_inhibited())
    then inner.Walk.visit_obj_fn_pre oid ident fn;
  in

  let visit_obj_fn_post oid ident fn =
    if not (indirectly_inhibited())
    then inner.Walk.visit_obj_fn_post oid ident fn;
  in

  let visit_obj_drop_pre oid d =
    if not (indirectly_inhibited())
    then inner.Walk.visit_obj_drop_pre oid d;
  in

  let visit_obj_drop_post oid d =
    if not (indirectly_inhibited())
    then inner.Walk.visit_obj_drop_post oid d;
  in

  let visit_constr_pre n c =
    if not (indirectly_inhibited())
    then inner.Walk.visit_constr_pre n c;
  in

  let visit_constr_post n c =
    if not (indirectly_inhibited())
    then inner.Walk.visit_constr_post n c;
  in

  let wrap1 fn =
    fun x ->
      if not (indirectly_inhibited())
      then fn x
  in

    { inner with
        Walk.visit_stmt_pre = wrap1 inner.Walk.visit_stmt_pre;
        Walk.visit_stmt_post = wrap1 inner.Walk.visit_stmt_post;
        Walk.visit_slot_identified_pre =
        wrap1 inner.Walk.visit_slot_identified_pre;
        Walk.visit_slot_identified_post =
        wrap1 inner.Walk.visit_slot_identified_post;
        Walk.visit_expr_pre = wrap1 inner.Walk.visit_expr_pre;
        Walk.visit_expr_post = wrap1 inner.Walk.visit_expr_post;
        Walk.visit_ty_pre = wrap1 inner.Walk.visit_ty_pre;
        Walk.visit_ty_post = wrap1 inner.Walk.visit_ty_post;
        Walk.visit_constr_pre = visit_constr_pre;
        Walk.visit_constr_post = visit_constr_post;
        Walk.visit_pat_pre = wrap1 inner.Walk.visit_pat_pre;
        Walk.visit_pat_post = wrap1 inner.Walk.visit_pat_post;
        Walk.visit_block_pre = wrap1 inner.Walk.visit_block_pre;
        Walk.visit_block_post = wrap1 inner.Walk.visit_block_post;
        Walk.visit_lit_pre = wrap1 inner.Walk.visit_lit_pre;
        Walk.visit_lit_post = wrap1 inner.Walk.visit_lit_post;
        Walk.visit_lval_pre = wrap1 inner.Walk.visit_lval_pre;
        Walk.visit_lval_post = wrap1 inner.Walk.visit_lval_post;
        Walk.visit_mod_item_pre = visit_mod_item_pre;
        Walk.visit_mod_item_post = visit_mod_item_post;
        Walk.visit_obj_fn_pre = visit_obj_fn_pre;
        Walk.visit_obj_fn_post = visit_obj_fn_post;
        Walk.visit_obj_drop_pre = visit_obj_drop_pre;
        Walk.visit_obj_drop_post = visit_obj_drop_post; }
;;

let rec name_of ncs =
  match ncs with
      [] -> bug () "Walk.name_of_ncs: empty path"
    | [(Ast.COMP_ident i)] -> Ast.NAME_base (Ast.BASE_ident i)
    | [(Ast.COMP_app x)] -> Ast.NAME_base (Ast.BASE_app x)
    | [(Ast.COMP_idx _)] ->
        bug () "Walk.name_of_ncs: path-name contains COMP_idx"
    | nc::ncs -> Ast.NAME_ext (name_of ncs, nc)
;;

let path_to_name
    (path:Ast.name_component Stack.t)
    : Ast.name =
  name_of (stk_elts_from_top path)
;;

let mod_item_logging_visitor
    (cx:ctxt)
    (log_flag:bool)
    (log:ctxt -> ('a, unit, string, unit) format4 -> 'a)
    (pass:int)
    (path:Ast.name_component Stack.t)
    (inner:Walk.visitor)
    : Walk.
visitor =
  let entering _ =
    if log_flag
    then
      log cx "pass %d: entering %a"
        pass Ast.sprintf_name (path_to_name path)
  in
  let entered _ =
    if log_flag
    then
      log cx "pass %d: entered %a"
        pass Ast.sprintf_name (path_to_name path)
  in
  let leaving _ =
    if log_flag
    then
      log cx "pass %d: leaving %a"
        pass Ast.sprintf_name (path_to_name path)
  in
  let left _ =
    if log_flag
    then
      log cx "pass %d: left %a"
        pass Ast.sprintf_name (path_to_name path)
  in

  let visit_mod_item_pre name params item =
    entering();
    inner.Walk.visit_mod_item_pre name params item;
    entered();
  in
  let visit_mod_item_post name params item =
    leaving();
    inner.Walk.visit_mod_item_post name params item;
    left();
  in
  let visit_obj_fn_pre obj ident fn =
    entering();
    inner.Walk.visit_obj_fn_pre obj ident fn;
    entered();
  in
  let visit_obj_fn_post obj ident fn =
    leaving();
    inner.Walk.visit_obj_fn_post obj ident fn;
    left();
  in
  let visit_obj_drop_pre obj b =
    entering();
    inner.Walk.visit_obj_drop_pre obj b;
    entered();
  in
  let visit_obj_drop_post obj fn =
    leaving();
    inner.Walk.visit_obj_drop_post obj fn;
    left();
  in
    { inner with
        Walk.visit_mod_item_pre = visit_mod_item_pre;
        Walk.visit_mod_item_post = visit_mod_item_post;
        Walk.visit_obj_fn_pre = visit_obj_fn_pre;
        Walk.visit_obj_fn_post = visit_obj_fn_post;
        Walk.visit_obj_drop_pre = visit_obj_drop_pre;
        Walk.visit_obj_drop_post = visit_obj_drop_post;
    }
;;



(* Generic lookup, used for slots, items, types, etc. *)

type resolved = ((scope list * node_id) option) ;;

let get_mod_item
    (cx:ctxt)
    (node:node_id)
    : (Ast.mod_view * Ast.mod_items) =
  match get_item cx node with
      { Ast.decl_item = Ast.MOD_ITEM_mod md } -> md
    | _ -> bugi cx node "defn is not a mod"
;;

let get_name_comp_ident
    (comp:Ast.name_component)
    : Ast.ident =
  match comp with
      Ast.COMP_ident i -> i
    | Ast.COMP_app (i, _) -> i
    | Ast.COMP_idx i -> string_of_int i
;;

let get_name_base_ident
    (comp:Ast.name_base)
    : Ast.ident =
  match comp with
      Ast.BASE_ident i -> i
    | Ast.BASE_app (i, _) -> i
    | Ast.BASE_temp _ ->
        bug () "get_name_base_ident on BASE_temp"
;;

type loop_check = (node_id * Ast.ident) list;;

let rec project_ident_from_items
    (cx:ctxt)
    (lchk:loop_check)
    (scopes:scope list)
    (scope_id:node_id)
    ((view:Ast.mod_view),(items:Ast.mod_items))
    (ident:Ast.ident)
    (inside:bool)
    : resolved =

  let lchk =
    if List.mem (scope_id, ident) lchk
    then err (Some scope_id) "cyclic import for ident %s" ident
    else (scope_id, ident)::lchk
  in

  if not (inside || (exports_permit view ident))
  then None
  else
    match htab_search items ident with
        Some i ->
          found cx scopes i.id
      | None ->
          match htab_search view.Ast.view_imports ident with
              None -> None
            | Some name ->
                lookup_by_name cx lchk scopes name

and found cx scopes id =
  Hashtbl.replace cx.ctxt_node_referenced id ();
  Some (scopes, id)

and project_name_comp_from_resolved
    (cx:ctxt)
    (lchk:loop_check)
    (mod_res:resolved)
    (ext:Ast.name_component)
    : resolved =
  match mod_res with
      None -> None
    | Some (scopes, id) ->
        let scope = (SCOPE_mod_item {id=id; node=get_item cx id}) in
        let scopes = scope :: scopes in
        let ident = get_name_comp_ident ext in
        let md = get_mod_item cx id in
          Hashtbl.replace cx.ctxt_node_referenced id ();
          project_ident_from_items cx lchk scopes id md ident false

and lookup_by_name
    (cx:ctxt)
    (lchk:loop_check)
    (scopes:scope list)
    (name:Ast.name)
    : resolved =
  assert (Ast.sane_name name);
  match name with
      Ast.NAME_base nb ->
        let ident = get_name_base_ident nb in
          lookup_by_ident cx lchk scopes ident
    | Ast.NAME_ext (name, ext) ->
        let base_res = lookup_by_name cx lchk scopes name in
          project_name_comp_from_resolved cx lchk base_res ext

and lookup_by_ident
    (cx:ctxt)
    (lchk:loop_check)
    (scopes:scope list)
    (ident:Ast.ident)
    : resolved =

  let check_slots scopes islots =
    arr_search islots
      (fun _ (sloti,ident') ->
         if ident = ident'
         then found cx scopes sloti.id
         else None)
  in

  let check_params scopes params =
    arr_search params
      (fun _ {node=(i,_); id=id} ->
         if i = ident
         then found cx scopes id
         else None)
  in

  let passed_capture_scope = ref false in

  let would_capture r =
    match r with
        None -> None
      | Some _ ->
          if !passed_capture_scope
          then err None "attempted dynamic environment-capture"
          else r
  in

  let check_scope scopes scope =
    match scope with
        SCOPE_block block_id ->
          let block_slots = Hashtbl.find cx.ctxt_block_slots block_id in
          let block_items = Hashtbl.find cx.ctxt_block_items block_id in
            begin
              match htab_search block_slots (Ast.KEY_ident ident) with
                  Some id -> would_capture (found cx scopes id)
                | None ->
                    match htab_search block_items ident with
                        Some id -> found cx scopes id
                      | None -> None
            end

      | SCOPE_crate crate ->
          project_ident_from_items
            cx lchk scopes crate.id crate.node.Ast.crate_items ident true

      | SCOPE_obj_fn fn ->
          would_capture (check_slots scopes fn.node.Ast.fn_input_slots)

      | SCOPE_mod_item item ->
          begin
            let item_match =
              match item.node.Ast.decl_item with
                  Ast.MOD_ITEM_fn f ->
                    check_slots scopes f.Ast.fn_input_slots

                | Ast.MOD_ITEM_obj obj ->
                    begin
                      match htab_search obj.Ast.obj_fns ident with
                          Some fn -> found cx scopes fn.id
                        | None -> check_slots scopes obj.Ast.obj_state
                    end

                | Ast.MOD_ITEM_mod md ->
                    project_ident_from_items cx lchk
                      scopes item.id md ident true

                | _ -> None
            in
              match item_match with
                  Some _ -> item_match
                | None ->
                    would_capture
                      (check_params scopes item.node.Ast.decl_params)
          end
  in
  let rec search scopes =
    match scopes with
        [] -> None
      | scope::rest ->
          match check_scope scopes scope with
              None ->
                begin
                  let is_ty_item i =
                    match i.node.Ast.decl_item with
                        Ast.MOD_ITEM_type _ -> true
                      | _ -> false
                  in
                    match scope with
                        SCOPE_block _
                      | SCOPE_obj_fn _ ->
                          search rest

                      | SCOPE_mod_item item when is_ty_item item ->
                          search rest

                      | _ ->
                          passed_capture_scope := true;
                          search rest
                end
            | x -> x
  in
    search scopes
;;

let lookup_by_temp
    (cx:ctxt)
    (scopes:scope list)
    (temp:temp_id)
    : ((scope list * node_id) option) =
  let passed_item_scope = ref false in
  let check_scope scope =
    if !passed_item_scope
    then None
    else
      match scope with
          SCOPE_block block_id ->
            let block_slots = Hashtbl.find cx.ctxt_block_slots block_id in
              htab_search block_slots (Ast.KEY_temp temp)
        | _ ->
            passed_item_scope := true;
            None
  in
    list_search_ctxt scopes check_scope
;;

let lookup
    (cx:ctxt)
    (scopes:scope list)
    (key:Ast.slot_key)
    : ((scope list * node_id) option) =
  match key with
      Ast.KEY_temp temp -> lookup_by_temp cx scopes temp
    | Ast.KEY_ident ident -> lookup_by_ident cx [] scopes ident
;;


let run_passes
    (cx:ctxt)
    (name:string)
    (path:Ast.name_component Stack.t)
    (passes:Walk.visitor array)
    (log_flag:bool)
    (log:ctxt -> ('a, unit, string, unit) format4 -> 'a)
    (crate:Ast.crate)
    : unit =
  let do_pass i pass =
      Walk.walk_crate
        (Walk.path_managing_visitor path
           (mod_item_logging_visitor cx log_flag log i path pass))
        crate
  in
  let sess = cx.ctxt_sess in
    if sess.Session.sess_failed
    then ()
    else
      try
        Session.time_inner name sess
          (fun _ -> Array.iteri do_pass passes)
      with
          Semant_err (ido, str) -> report_err cx ido str
;;

(* Rust type -> IL type conversion. *)

let word_sty (abi:Abi.abi) : Il.scalar_ty =
  Il.ValTy abi.Abi.abi_word_bits
;;

let word_rty (abi:Abi.abi) : Il.referent_ty =
  Il.ScalarTy (word_sty abi)
;;

let tydesc_rty (abi:Abi.abi) : Il.referent_ty =
  (* 
   * NB: must match corresponding tydesc structure
   * in trans and offsets in ABI exactly.
   *)
  Il.StructTy
    [|
      word_rty abi;                      (* Abi.tydesc_field_first_param   *)
      word_rty abi;                      (* Abi.tydesc_field_size          *)
      word_rty abi;                      (* Abi.tydesc_field_align         *)
      Il.ScalarTy (Il.AddrTy Il.CodeTy); (* Abi.tydesc_field_copy_glue     *)
      Il.ScalarTy (Il.AddrTy Il.CodeTy); (* Abi.tydesc_field_drop_glue     *)
      Il.ScalarTy (Il.AddrTy Il.CodeTy); (* Abi.tydesc_field_free_glue     *)
      Il.ScalarTy (Il.AddrTy Il.CodeTy); (* Abi.tydesc_field_sever_glue    *)
      Il.ScalarTy (Il.AddrTy Il.CodeTy); (* Abi.tydesc_field_mark_glue     *)
      Il.ScalarTy (Il.AddrTy Il.CodeTy); (* Abi.tydesc_field_obj_drop_glue *)
    |]
;;

let obj_closure_rty (abi:Abi.abi) : Il.referent_ty =
  Il.StructTy [| word_rty abi;
                 Il.ScalarTy (Il.AddrTy (tydesc_rty abi));
                 word_rty abi (* A lie: it's opaque, but this permits
                               * GEP'ing to it. *)
              |]
;;

let rec referent_type (abi:Abi.abi) (t:Ast.ty) : Il.referent_ty =
  let s t = Il.ScalarTy t in
  let v b = Il.ValTy b in
  let p t = Il.AddrTy t in
  let sv b = s (v b) in
  let sp t = s (p t) in

  let word = word_rty abi in
  let ptr = sp Il.OpaqueTy in
  let rc_ptr = sp (Il.StructTy [| word; Il.OpaqueTy |]) in
  let codeptr = sp Il.CodeTy in
  let tup ttup = Il.StructTy (Array.map (referent_type abi) ttup) in
  let tag ttag =
    let union =
      Il.UnionTy
        (Array.map
           (fun key -> tup (Hashtbl.find ttag key))
           (sorted_htab_keys ttag))
    in
    let discriminant = word in
      Il.StructTy [| discriminant; union |]
  in

    match t with
        Ast.TY_any -> Il.StructTy [| word;  ptr |]
      | Ast.TY_nil -> Il.NilTy
      | Ast.TY_int
      | Ast.TY_uint -> word

      | Ast.TY_bool -> sv Il.Bits8

      | Ast.TY_mach (TY_u8)
      | Ast.TY_mach (TY_i8) -> sv Il.Bits8

      | Ast.TY_mach (TY_u16)
      | Ast.TY_mach (TY_i16) -> sv Il.Bits16

      | Ast.TY_mach (TY_u32)
      | Ast.TY_mach (TY_i32)
      | Ast.TY_mach (TY_f32)
      | Ast.TY_char -> sv Il.Bits32

      | Ast.TY_mach (TY_u64)
      | Ast.TY_mach (TY_i64)
      | Ast.TY_mach (TY_f64) -> sv Il.Bits64

      | Ast.TY_str -> sp (Il.StructTy [| word; word; word; ptr |])
      | Ast.TY_vec _ -> sp (Il.StructTy [| word; word; word; ptr |])
      | Ast.TY_tup tt -> tup tt
      | Ast.TY_rec tr -> tup (Array.map snd tr)

      | Ast.TY_fn _ ->
          let fn_closure_ptr = sp (Il.StructTy [| word; Il.OpaqueTy |]) in
            Il.StructTy [| codeptr; fn_closure_ptr |]

      | Ast.TY_obj _ ->
          let obj_closure_ptr = sp (obj_closure_rty abi) in
            Il.StructTy [| ptr; obj_closure_ptr |]

      | Ast.TY_tag ttag -> tag ttag
      | Ast.TY_iso tiso -> tag tiso.Ast.iso_group.(tiso.Ast.iso_index)

      | Ast.TY_idx _ -> word (* A lie, but permits GEP'ing to it. *)

      | Ast.TY_chan _
      | Ast.TY_port _
      | Ast.TY_task -> rc_ptr

      | Ast.TY_type -> sp (tydesc_rty abi)

      | Ast.TY_native _ -> ptr

      | Ast.TY_box t ->
          sp (Il.StructTy [| word; referent_type abi t |])

      | Ast.TY_mutable t -> referent_type abi t

      | Ast.TY_param (i, _) -> Il.ParamTy i

      | Ast.TY_named _ -> bug () "named type in referent_type"
      | Ast.TY_constrained (t, _) -> referent_type abi t

and slot_referent_type (abi:Abi.abi) (sl:Ast.slot) : Il.referent_ty =
  let s t = Il.ScalarTy t in
  let p t = Il.AddrTy t in
  let sp t = s (p t) in

  let rty = referent_type abi (slot_ty sl) in
    match sl.Ast.slot_mode with
      | Ast.MODE_local -> rty
      | Ast.MODE_alias -> sp rty
;;

let task_rty (abi:Abi.abi) : Il.referent_ty =
  Il.StructTy
    begin
      Array.init
        Abi.n_visible_task_fields
        (fun _ -> word_rty abi)
    end
;;

let call_args_referent_type_full
    (abi:Abi.abi)
    (out_slot:Ast.slot)
    (n_ty_params:int)
    (in_slots:Ast.slot array)
    (iterator_arg_rtys:Il.referent_ty array)
    (indirect_arg_rtys:Il.referent_ty array)
    : Il.referent_ty =
  let out_slot_rty = slot_referent_type abi out_slot in
  let out_ptr_rty = Il.ScalarTy (Il.AddrTy out_slot_rty) in
  let task_ptr_rty = Il.ScalarTy (Il.AddrTy (task_rty abi)) in
  let ty_param_rtys =
    let td = Il.ScalarTy (Il.AddrTy (tydesc_rty abi)) in
      Il.StructTy (Array.init n_ty_params (fun _ -> td))
  in
  let arg_rtys = Il.StructTy (Array.map (slot_referent_type abi) in_slots) in
    (* 
     * NB: must match corresponding calltup structure in trans and
     * member indices in ABI exactly.
     *)
    Il.StructTy
      [|
        out_ptr_rty;                   (* Abi.calltup_elt_out_ptr       *)
        task_ptr_rty;                  (* Abi.calltup_elt_task_ptr      *)
        ty_param_rtys;                 (* Abi.calltup_elt_ty_params     *)
        arg_rtys;                      (* Abi.calltup_elt_args          *)
        Il.StructTy iterator_arg_rtys; (* Abi.calltup_elt_iterator_args *)
        Il.StructTy indirect_arg_rtys  (* Abi.calltup_elt_indirect_args *)
      |]
;;

let call_args_referent_type
    (cx:ctxt)
    (n_ty_params:int)
    (callee_ty:Ast.ty)
    (closure:Il.referent_ty option)
    : Il.referent_ty =
  let indirect_arg_rtys =
    match closure with
        None -> [| |]
      | Some c ->
          [|
            (* Abi.indirect_args_elt_closure *)
            Il.ScalarTy (Il.AddrTy c)
          |]
  in
  let iterator_arg_rtys _ =
    [|
      (* Abi.iterator_args_elt_loop_size *)
      Il.ScalarTy (Il.ValTy cx.ctxt_abi.Abi.abi_word_bits);
      (* Abi.iterator_args_elt_loop_info_ptr *)
      Il.ScalarTy (Il.AddrTy Il.OpaqueTy)
    |]
  in
    match simplified_ty callee_ty with
        Ast.TY_fn (tsig, taux) ->
          call_args_referent_type_full
            cx.ctxt_abi
            tsig.Ast.sig_output_slot
            n_ty_params
            tsig.Ast.sig_input_slots
            (if taux.Ast.fn_is_iter then (iterator_arg_rtys()) else [||])
            indirect_arg_rtys

      | _ -> bug cx
          "Semant.call_args_referent_type on non-callable type %a"
            Ast.sprintf_ty callee_ty
;;

let indirect_call_args_referent_type
    (cx:ctxt)
    (n_ty_params:int)
    (callee_ty:Ast.ty)
    (closure:Il.referent_ty)
    : Il.referent_ty =
  call_args_referent_type cx n_ty_params callee_ty (Some closure)
;;

let direct_call_args_referent_type
    (cx:ctxt)
    (callee_node:node_id)
    : Il.referent_ty =
  let ity = Hashtbl.find cx.ctxt_all_item_types callee_node in
  let n_ty_params =
    if item_is_obj_fn cx callee_node
    then 0
    else n_item_ty_params cx callee_node
  in
    call_args_referent_type cx n_ty_params ity None
;;

let ty_sz (abi:Abi.abi) (t:Ast.ty) : int64 =
  force_sz (Il.referent_ty_size abi.Abi.abi_word_bits (referent_type abi t))
;;

let ty_align (abi:Abi.abi) (t:Ast.ty) : int64 =
  force_sz (Il.referent_ty_align abi.Abi.abi_word_bits (referent_type abi t))
;;

let slot_sz (abi:Abi.abi) (s:Ast.slot) : int64 =
  force_sz (Il.referent_ty_size abi.Abi.abi_word_bits
              (slot_referent_type abi s))
;;

let word_slot (abi:Abi.abi) : Ast.slot =
  local_slot (Ast.TY_mach abi.Abi.abi_word_ty)
;;

let alias_slot (ty:Ast.ty) : Ast.slot =
  { Ast.slot_mode = Ast.MODE_alias;
    Ast.slot_ty = Some ty }
;;

let mutable_alias_slot (ty:Ast.ty) : Ast.slot =
  let ty =
    match ty with
        Ast.TY_mutable _ -> ty
      | _ -> Ast.TY_mutable ty
  in
    { Ast.slot_mode = Ast.MODE_alias;
      Ast.slot_ty = Some ty }
;;

let mk_ty_fn_or_iter
    (out_slot:Ast.slot)
    (arg_slots:Ast.slot array)
    (is_iter:bool)
    : Ast.ty =
  (* In some cases we don't care what aux or constrs are. *)
  let taux = { Ast.fn_effect = Ast.PURE;
               Ast.fn_is_iter = is_iter; }
  in
  let tsig = { Ast.sig_input_slots = arg_slots;
               Ast.sig_input_constrs = [| |];
               Ast.sig_output_slot = out_slot; }
  in
    Ast.TY_fn (tsig, taux)
;;

let mk_ty_fn
    (out_slot:Ast.slot)
    (arg_slots:Ast.slot array)
    : Ast.ty =
  mk_ty_fn_or_iter out_slot arg_slots false
;;

let mk_simple_ty_fn
    (arg_slots:Ast.slot array)
    : Ast.ty =
  (* In some cases we don't care what the output slot is. *)
  let out_slot = local_slot Ast.TY_nil in
    mk_ty_fn out_slot arg_slots
;;

let mk_simple_ty_iter
    (arg_slots:Ast.slot array)
    : Ast.ty =
  (* In some cases we don't care what the output slot is. *)
  let out_slot = local_slot Ast.TY_nil in
    mk_ty_fn_or_iter out_slot arg_slots true
;;


(* name mangling support. *)

let item_name (cx:ctxt) (id:node_id) : Ast.name =
  Hashtbl.find cx.ctxt_all_item_names id
;;

let item_str (cx:ctxt) (id:node_id) : string =
    string_of_name (item_name cx id)
;;

let ty_str (ty:Ast.ty) : string =
  let base = associative_binary_op_ty_fold "" (fun a b -> a ^ b) in
  let fold_slot (mode,ty) =
    (match mode with
         Ast.MODE_alias -> "a"
       | Ast.MODE_local -> "")
    ^ ty
  in
  let num n = (string_of_int n) ^ "$" in
  let len a = num (Array.length a) in
  let join az = Array.fold_left (fun a b -> a ^ b) "" az in
  let fold_slots slots =
    "t"
    ^ (len slots)
    ^ (join slots)
  in
  let fold_rec entries =
    "r"
    ^ (len entries)
    ^ (Array.fold_left
         (fun str (ident, s) -> str ^ "$" ^ ident ^ "$" ^ s)
         "" entries)
  in
  let fold_tags tags =
    "g"
    ^ (num (Hashtbl.length tags))
    ^ (Array.fold_left
         (fun str key -> str ^ (string_of_name key) ^ (Hashtbl.find tags key))
         "" (sorted_htab_keys tags))
  in
  let fold_iso (n, tags) =
    "G"
    ^ (num n)
    ^ (len tags)
    ^ (join tags)
  in
  let fold_mach m =
    match m with
        TY_u8 -> "U0"
      | TY_u16 -> "U1"
      | TY_u32 -> "U2"
      | TY_u64 -> "U3"
      | TY_i8 -> "I0"
      | TY_i16 -> "I1"
      | TY_i32 -> "I2"
      | TY_i64 -> "I3"
      | TY_f32 -> "F2"
      | TY_f64 -> "F3"
  in
  let fold =
     { base with
         (* Structural types. *)
         ty_fold_slot = fold_slot;
         ty_fold_slots = fold_slots;
         ty_fold_tags = fold_tags;
         ty_fold_rec = fold_rec;
         ty_fold_nil = (fun _ -> "n");
         ty_fold_bool = (fun _ -> "b");
         ty_fold_mach = fold_mach;
         ty_fold_int = (fun _ -> "i");
         ty_fold_uint = (fun _ -> "u");
         ty_fold_char = (fun _ -> "c");
         ty_fold_obj = (fun _ -> "o");
         ty_fold_str = (fun _ -> "s");
         ty_fold_vec = (fun s -> "v" ^ s);
         ty_fold_iso = fold_iso;
         ty_fold_idx = (fun i -> "x" ^ (string_of_int i));
         (* FIXME (issue #78): encode constrs, aux as well. *)
         ty_fold_fn = (fun ((ins,_,out),_) -> "f" ^ ins ^ out);

         (* Built-in special types. *)
         ty_fold_any = (fun _ -> "A");
         ty_fold_chan = (fun t -> "H" ^ t);
         ty_fold_port = (fun t -> "R" ^ t);
         ty_fold_task = (fun _ -> "T");
         ty_fold_native = (fun _ -> "N");
         ty_fold_param = (fun _ -> "P");
         ty_fold_type = (fun _ -> "Y");
         ty_fold_mutable = (fun t -> "M" ^ t);
         ty_fold_box = (fun t -> "B" ^ t);

         (* FIXME (issue #78): encode obj types. *)
         (* FIXME (issue #78): encode opaque and param numbers. *)
         ty_fold_named = (fun _ -> bug () "string-encoding named type");
         (* FIXME (issue #78): encode constrs as well. *)
         ty_fold_constrained = (fun (t,_)-> t) }
  in
    fold_ty fold ty
;;

let glue_str (cx:ctxt) (g:glue) : string =
  match g with
      GLUE_activate -> "glue$activate"
    | GLUE_yield -> "glue$yield"
    | GLUE_exit_main_task -> "glue$exit_main_task"
    | GLUE_exit_task -> "glue$exit_task"
    | GLUE_copy ty -> "glue$copy$" ^ (ty_str ty)
    | GLUE_drop ty -> "glue$drop$" ^ (ty_str ty)
    | GLUE_free ty -> "glue$free$" ^ (ty_str ty)
    | GLUE_sever ty -> "glue$sever$" ^ (ty_str ty)
    | GLUE_mark ty -> "glue$mark$" ^ (ty_str ty)
    | GLUE_clone ty -> "glue$clone$" ^ (ty_str ty)
    | GLUE_compare ty -> "glue$compare$" ^ (ty_str ty)
    | GLUE_hash ty -> "glue$hash$" ^ (ty_str ty)
    | GLUE_write ty -> "glue$write$" ^ (ty_str ty)
    | GLUE_read ty -> "glue$read$" ^ (ty_str ty)
    | GLUE_unwind -> "glue$unwind"
    | GLUE_gc -> "glue$gc"
    | GLUE_get_next_pc -> "glue$get_next_pc"
    | GLUE_mark_frame i -> "glue$mark_frame$" ^ (item_str cx i)
    | GLUE_drop_frame i -> "glue$drop_frame$" ^ (item_str cx i)
    | GLUE_reloc_frame i -> "glue$reloc_frame$" ^ (item_str cx i)
        (* 
         * FIXME (issue #78): the node_id here isn't an item, it's 
         * a statement; lookup bind target and encode bound arg 
         * tuple type.
         *)
    | GLUE_fn_binding i
      -> "glue$fn_binding$" ^ (string_of_int (int_of_node i))
    | GLUE_obj_drop oid
      -> (item_str cx oid) ^ ".drop"
    | GLUE_loop_body i
      -> "glue$loop_body$" ^ (string_of_int (int_of_node i))
    | GLUE_forward (id, oty1, oty2)
      -> "glue$forward$"
        ^ id
        ^ "$" ^ (ty_str (Ast.TY_obj oty1))
        ^ "$" ^ (ty_str (Ast.TY_obj oty2))
;;


(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
