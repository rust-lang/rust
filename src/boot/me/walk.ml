
open Common;;

(*
 * The purpose of this module is just to decouple the AST from the
 * various passes that are interested in visiting "parts" of it.
 * If the AST shifts, we have better odds of the shift only affecting
 * this module rather than all of its clients. Similarly if the
 * clients only need to visit part, they only have to define the
 * part of the walk they're interested in, making it cheaper to define
 * multiple passes.
 *)

type visitor =
    {
      visit_stmt_pre: Ast.stmt -> unit;
      visit_stmt_post: Ast.stmt -> unit;
      visit_slot_identified_pre: (Ast.slot identified) -> unit;
      visit_slot_identified_post: (Ast.slot identified) -> unit;
      visit_expr_pre: Ast.expr -> unit;
      visit_expr_post: Ast.expr -> unit;
      visit_pexp_pre: Ast.pexp -> unit;
      visit_pexp_post: Ast.pexp -> unit;
      visit_ty_pre: Ast.ty -> unit;
      visit_ty_post: Ast.ty -> unit;
      visit_constr_pre: node_id option -> Ast.constr -> unit;
      visit_constr_post: node_id option -> Ast.constr -> unit;
      visit_pat_pre: Ast.pat -> unit;
      visit_pat_post: Ast.pat -> unit;
      visit_block_pre: Ast.block -> unit;
      visit_block_post: Ast.block -> unit;

      visit_lit_pre: Ast.lit -> unit;
      visit_lit_post: Ast.lit -> unit;
      visit_lval_pre: Ast.lval -> unit;
      visit_lval_post: Ast.lval -> unit;
      visit_plval_pre: Ast.plval -> unit;
      visit_plval_post: Ast.plval -> unit;
      visit_mod_item_pre:
        (Ast.ident
         -> ((Ast.ty_param identified) array)
           -> Ast.mod_item
             -> unit);
      visit_mod_item_post:
        (Ast.ident
         -> ((Ast.ty_param identified) array)
           -> Ast.mod_item
             -> unit);
      visit_obj_fn_pre:
        (Ast.obj identified) -> Ast.ident -> (Ast.fn identified) -> unit;
      visit_obj_fn_post:
        (Ast.obj identified) -> Ast.ident -> (Ast.fn identified) -> unit;
      visit_obj_drop_pre:
        (Ast.obj identified) -> Ast.block -> unit;
      visit_obj_drop_post:
        (Ast.obj identified) -> Ast.block -> unit;
      visit_crate_pre: Ast.crate -> unit;
      visit_crate_post: Ast.crate -> unit;
    }
;;


let empty_visitor =
  { visit_stmt_pre = (fun _ -> ());
    visit_stmt_post = (fun _ -> ());
    visit_slot_identified_pre = (fun _ -> ());
    visit_slot_identified_post = (fun _ -> ());
    visit_expr_pre = (fun _ -> ());
    visit_expr_post = (fun _ -> ());
    visit_pexp_pre = (fun _ -> ());
    visit_pexp_post = (fun _ -> ());
    visit_ty_pre = (fun _ -> ());
    visit_ty_post = (fun _ -> ());
    visit_constr_pre = (fun _ _ -> ());
    visit_constr_post = (fun _ _ -> ());
    visit_pat_pre = (fun _ -> ());
    visit_pat_post = (fun _ -> ());
    visit_block_pre = (fun _ -> ());
    visit_block_post = (fun _ -> ());
    visit_lit_pre = (fun _ -> ());
    visit_lit_post = (fun _ -> ());
    visit_lval_pre = (fun _ -> ());
    visit_lval_post = (fun _ -> ());
    visit_plval_pre = (fun _ -> ());
    visit_plval_post = (fun _ -> ());
    visit_mod_item_pre = (fun _ _ _ -> ());
    visit_mod_item_post = (fun _ _ _ -> ());
    visit_obj_fn_pre = (fun _ _ _ -> ());
    visit_obj_fn_post = (fun _ _ _ -> ());
    visit_obj_drop_pre = (fun _ _ -> ());
    visit_obj_drop_post = (fun _ _ -> ());
    visit_crate_pre = (fun _ -> ());
    visit_crate_post = (fun _ -> ()); }
;;

let path_managing_visitor
    (path:Ast.name_component Stack.t)
    (inner:visitor)
    : visitor =
  let visit_mod_item_pre ident params item =
    Stack.push (Ast.COMP_ident ident) path;
    inner.visit_mod_item_pre ident params item
  in
  let visit_mod_item_post ident params item =
    inner.visit_mod_item_post ident params item;
    ignore (Stack.pop path)
  in
  let visit_obj_fn_pre obj ident fn =
    Stack.push (Ast.COMP_ident ident) path;
    inner.visit_obj_fn_pre obj ident fn
  in
  let visit_obj_fn_post obj ident fn =
    inner.visit_obj_fn_post obj ident fn;
    ignore (Stack.pop path)
  in
  let visit_obj_drop_pre obj b =
    Stack.push (Ast.COMP_ident "drop") path;
    inner.visit_obj_drop_pre obj b
  in
  let visit_obj_drop_post obj b =
    inner.visit_obj_drop_post obj b;
    ignore (Stack.pop path)
  in
    { inner with
        visit_mod_item_pre = visit_mod_item_pre;
        visit_mod_item_post = visit_mod_item_post;
        visit_obj_fn_pre = visit_obj_fn_pre;
        visit_obj_fn_post = visit_obj_fn_post;
        visit_obj_drop_pre = visit_obj_drop_pre;
        visit_obj_drop_post = visit_obj_drop_post;
    }
;;


let walk_bracketed
    (pre:'a -> unit)
    (children:unit -> unit)
    (post:'a -> unit)
    (x:'a)
    : unit =
  begin
    pre x;
    children ();
    post x
  end
;;


let walk_option
    (walker:'a -> unit)
    (opt:'a option)
    : unit =
  match opt with
      None -> ()
    | Some v -> walker v
;;


let rec walk_crate
    (v:visitor)
    (crate:Ast.crate)
    : unit =
    walk_bracketed
      v.visit_crate_pre
      (fun _ -> walk_mod_items v (snd crate.node.Ast.crate_items))
      v.visit_crate_post
      crate

and walk_mod_items
    (v:visitor)
    (items:Ast.mod_items)
    : unit =
  Hashtbl.iter (walk_mod_item v) items


and walk_mod_item
    (v:visitor)
    (name:Ast.ident)
    (item:Ast.mod_item)
    : unit =
  let children _ =
    match item.node.Ast.decl_item with
        Ast.MOD_ITEM_type (_, ty) -> walk_ty v ty
      | Ast.MOD_ITEM_const (ty, e) ->
          walk_ty v ty;
          walk_option (walk_expr v) e
      | Ast.MOD_ITEM_fn f -> walk_fn v f item.id
      | Ast.MOD_ITEM_tag (hdr, _, _) ->
          walk_header_slots v hdr
      | Ast.MOD_ITEM_mod (_, items) ->
          walk_mod_items v items
      | Ast.MOD_ITEM_obj ob ->
          walk_header_slots v ob.Ast.obj_state;
          walk_constrs v (Some item.id) ob.Ast.obj_constrs;
          let oid = { node = ob; id = item.id } in
            Hashtbl.iter (walk_obj_fn v oid) ob.Ast.obj_fns;
            match ob.Ast.obj_drop with
                None -> ()
              | Some d ->
                  v.visit_obj_drop_pre oid d;
                  walk_block v d;
                  v.visit_obj_drop_post oid d

  in
    walk_bracketed
      (v.visit_mod_item_pre name item.node.Ast.decl_params)
      children
      (v.visit_mod_item_post name item.node.Ast.decl_params)
      item


and walk_ty_tup v ttup = Array.iter (walk_ty v) ttup

and walk_ty
    (v:visitor)
    (ty:Ast.ty)
    : unit =
  let children _ =
    match ty with
        Ast.TY_tup ttup -> walk_ty_tup v ttup
      | Ast.TY_vec s -> walk_ty v s
      | Ast.TY_rec trec -> Array.iter (fun (_, s) -> walk_ty v s) trec
      | Ast.TY_fn tfn -> walk_ty_fn v tfn
      | Ast.TY_obj (_, fns) ->
          Hashtbl.iter (fun _ tfn -> walk_ty_fn v tfn) fns
      | Ast.TY_chan t -> walk_ty v t
      | Ast.TY_port t -> walk_ty v t
      | Ast.TY_constrained (t,cs) ->
          begin
            walk_ty v t;
            walk_constrs v None cs
          end
      | Ast.TY_named _ -> ()
      | Ast.TY_param _ -> ()
      | Ast.TY_tag _ -> ()
      | Ast.TY_native _ -> ()
      | Ast.TY_mach _ -> ()
      | Ast.TY_type -> ()
      | Ast.TY_str -> ()
      | Ast.TY_char -> ()
      | Ast.TY_int -> ()
      | Ast.TY_uint -> ()
      | Ast.TY_bool -> ()
      | Ast.TY_nil -> ()
      | Ast.TY_task -> ()
      | Ast.TY_any -> ()
      | Ast.TY_box m -> walk_ty v m
      | Ast.TY_mutable m -> walk_ty v m
  in
    walk_bracketed
      v.visit_ty_pre
      children
      v.visit_ty_post
      ty


and walk_ty_sig
    (v:visitor)
    (s:Ast.ty_sig)
    : unit =
  begin
    Array.iter (walk_slot v) s.Ast.sig_input_slots;
    walk_constrs v None s.Ast.sig_input_constrs;
    walk_slot v s.Ast.sig_output_slot;
  end


and walk_ty_fn
    (v:visitor)
    (tfn:Ast.ty_fn)
    : unit =
  let (tsig, _) = tfn in
  walk_ty_sig v tsig


and walk_constrs
    (v:visitor)
    (formal_base:node_id option)
    (cs:Ast.constrs)
    : unit =
  Array.iter (walk_constr v formal_base) cs

and walk_check_calls
    (v:visitor)
    (calls:Ast.check_calls)
    : unit =
  Array.iter
    begin
      fun (f, args) ->
        walk_lval v f;
        Array.iter (walk_atom v) args
    end
    calls


and walk_constr
    (v:visitor)
    (formal_base:node_id option)
    (c:Ast.constr)
    : unit =
  walk_bracketed
    (v.visit_constr_pre formal_base)
    (fun _ -> ())
    (v.visit_constr_post formal_base)
    c

and walk_header_slots
    (v:visitor)
    (hslots:Ast.header_slots)
    : unit =
  Array.iter (fun (s,_) -> walk_slot_identified v s) hslots

and walk_header_tup
    (v:visitor)
    (htup:Ast.header_tup)
    : unit =
  Array.iter (walk_slot_identified v) htup

and walk_obj_fn
    (v:visitor)
    (obj:Ast.obj identified)
    (ident:Ast.ident)
    (f:Ast.fn identified)
    : unit =
  v.visit_obj_fn_pre obj ident f;
  walk_fn v f.node f.id;
  v.visit_obj_fn_post obj ident f

and walk_fn
    (v:visitor)
    (f:Ast.fn)
    (id:node_id)
    : unit =
  walk_header_slots v f.Ast.fn_input_slots;
  walk_constrs v (Some id) f.Ast.fn_input_constrs;
  walk_slot_identified v f.Ast.fn_output_slot;
  walk_block v f.Ast.fn_body

and walk_slot_identified
    (v:visitor)
    (s:Ast.slot identified)
    : unit =
  walk_bracketed
    v.visit_slot_identified_pre
    (fun _ -> walk_slot v s.node)
    v.visit_slot_identified_post
    s


and walk_slot
    (v:visitor)
    (s:Ast.slot)
    : unit =
  walk_option (walk_ty v) s.Ast.slot_ty


and walk_stmt
    (v:visitor)
    (s:Ast.stmt)
    : unit =
  let walk_stmt_for
      (s:Ast.stmt_for)
      : unit =
    let (si,_) = s.Ast.for_slot in
    let lv = s.Ast.for_seq in
      walk_slot_identified v si;
      walk_lval v lv;
      walk_block v s.Ast.for_body
  in
  let walk_stmt_for_each
      (s:Ast.stmt_for_each)
      : unit =
    let (si,_) = s.Ast.for_each_slot in
    let (f,az) = s.Ast.for_each_call in
      walk_slot_identified v si;
      walk_lval v f;
      Array.iter (walk_atom v) az;
      walk_block v s.Ast.for_each_head
  in
  let walk_stmt_while
      (s:Ast.stmt_while)
      : unit =
    let (ss,e) = s.Ast.while_lval in
      Array.iter (walk_stmt v) ss;
      walk_expr v e;
      walk_block v s.Ast.while_body
  in
  let children _ =
    match s.node with
        Ast.STMT_log a | Ast.STMT_log_err a ->
          walk_atom v a

      | Ast.STMT_new_rec (lv, atab, base) ->
          walk_lval v lv;
          Array.iter (fun (_, _, a) -> walk_atom v a) atab;
          walk_option (walk_lval v) base;

      | Ast.STMT_new_vec (lv, _, atoms) ->
          walk_lval v lv;
          Array.iter (walk_atom v) atoms

      | Ast.STMT_new_tup (lv, mut_atoms) ->
          walk_lval v lv;
          Array.iter (fun (_, atom) -> walk_atom v atom) mut_atoms

      | Ast.STMT_new_str (lv, _) ->
          walk_lval v lv

      | Ast.STMT_new_port lv ->
          walk_lval v lv

      | Ast.STMT_new_chan (chan,port) ->
          walk_option (walk_lval v) port;
          walk_lval v chan;

      | Ast.STMT_new_box (dst, _, src) ->
          walk_lval v dst;
          walk_atom v src

      | Ast.STMT_for f ->
          walk_stmt_for f

      | Ast.STMT_for_each f ->
          walk_stmt_for_each f

      | Ast.STMT_while w ->
          walk_stmt_while w

      | Ast.STMT_do_while w ->
          walk_stmt_while w

      | Ast.STMT_if i ->
          begin
            walk_expr v i.Ast.if_test;
            walk_block v i.Ast.if_then;
            walk_option (walk_block v) i.Ast.if_else
          end

      | Ast.STMT_block b ->
          walk_block v b

      | Ast.STMT_copy (lv,e) ->
          walk_lval v lv;
          walk_expr v e

      | Ast.STMT_copy_binop (lv,_,a) ->
          walk_lval v lv;
          walk_atom v a

      | Ast.STMT_call (dst,f,az) ->
          walk_lval v dst;
          walk_lval v f;
          Array.iter (walk_atom v) az

      | Ast.STMT_bind (dst, f, az) ->
          walk_lval v dst;
          walk_lval v f;
          Array.iter (walk_opt_atom v) az

      | Ast.STMT_spawn (dst,_,_,p,az) ->
          walk_lval v dst;
          walk_lval v p;
          Array.iter (walk_atom v) az

      | Ast.STMT_ret ao ->
          walk_option (walk_atom v) ao

      | Ast.STMT_put at ->
          walk_option (walk_atom v) at

      | Ast.STMT_put_each (lv, ats) ->
          walk_lval v lv;
          Array.iter (walk_atom v) ats

      (* FIXME (issue #86): this should have a param array, and invoke the
       * visitors. 
       *)
      | Ast.STMT_decl (Ast.DECL_mod_item (id, mi)) ->
          walk_mod_item v id mi

      | Ast.STMT_decl (Ast.DECL_slot (_, slot)) ->
          walk_slot_identified v slot

      | Ast.STMT_break | Ast.STMT_cont | Ast.STMT_yield | Ast.STMT_fail ->
          ()

      | Ast.STMT_join task ->
          walk_lval v task

      | Ast.STMT_send (dst,src) ->
          walk_lval v dst;
          walk_lval v src

      | Ast.STMT_recv (dst,src) ->
          walk_lval v dst;
          walk_lval v src

      | Ast.STMT_be (lv, ats) ->
          walk_lval v lv;
          Array.iter (walk_atom v) ats

      | Ast.STMT_check_expr e ->
          walk_expr v e

      | Ast.STMT_check (cs, calls) ->
          walk_constrs v None cs;
          walk_check_calls v calls

      | Ast.STMT_check_if (cs,calls,b) ->
          walk_constrs v None cs;
          walk_check_calls v calls;
          walk_block v b

      | Ast.STMT_prove cs ->
          walk_constrs v None cs

      | Ast.STMT_alt_tag
          { Ast.alt_tag_lval = lval; Ast.alt_tag_arms = arms } ->
          walk_lval v lval;
            let walk_arm { node = (pat, block); id=_ } =
              walk_pat v pat;
              walk_block v block
            in
              Array.iter walk_arm arms

      (* FIXME (issue #20): finish this as needed. *)
      | Ast.STMT_slice _
      | Ast.STMT_note _
      | Ast.STMT_alt_type _
      | Ast.STMT_alt_port _ ->
          unimpl (Some s.id) "statement type in Walk.walk_stmt"
  in
    walk_bracketed
      v.visit_stmt_pre
      children
      v.visit_stmt_post
      s

and walk_unop
    (v:visitor)
    (unop:Ast.unop)
    : unit =
  match unop with
      Ast.UNOP_cast tyi ->
        walk_ty v tyi.node
    | _ -> ()


and walk_expr
    (v:visitor)
    (e:Ast.expr)
    : unit =
  let children _ =
    match e with
        Ast.EXPR_binary (_,aa,ab) ->
          walk_atom v aa;
          walk_atom v ab
      | Ast.EXPR_unary (unop,a) ->
          walk_atom v a;
          walk_unop v unop
      | Ast.EXPR_atom a ->
          walk_atom v a
  in
    walk_bracketed
      v.visit_expr_pre
      children
      v.visit_expr_post
      e

and walk_pexp
    (v:visitor)
    (p:Ast.pexp)
    : unit =
  let children _ =
    match p.node with
        Ast.PEXP_call (pexp, pexps) ->
          walk_pexp v pexp;
          Array.iter (walk_pexp v) pexps

      | Ast.PEXP_spawn (_, _, pexp)
      | Ast.PEXP_box (_, pexp) ->
          walk_pexp v pexp;

      | Ast.PEXP_unop (unop, pexp) ->
          walk_pexp v pexp;
          walk_unop v unop

      | Ast.PEXP_bind (pexp, pexp_opts) ->
          walk_pexp v pexp;
          Array.iter (walk_option (walk_pexp v)) pexp_opts

      | Ast.PEXP_rec (elts, base) ->
          let walk_elt (_, _, pexp) = walk_pexp v pexp in
            Array.iter walk_elt elts;
            walk_option (walk_pexp v) base

      | Ast.PEXP_tup elts ->
          let walk_elt (_, pexp) = walk_pexp v pexp in
            Array.iter walk_elt elts

      | Ast.PEXP_vec (_, pexps)
      | Ast.PEXP_custom (_, pexps, _) ->
          Array.iter (walk_pexp v) pexps

      | Ast.PEXP_chan po ->
          walk_option (walk_pexp v) po

      | Ast.PEXP_binop (_, a, b)
      | Ast.PEXP_lazy_and (a, b)
      | Ast.PEXP_lazy_or (a, b) ->
          walk_pexp v a;
          walk_pexp v b

      | Ast.PEXP_lval pl -> walk_plval v pl

      | Ast.PEXP_lit lit -> walk_lit v lit

      | Ast.PEXP_port
      | Ast.PEXP_str _ -> ()
  in
    walk_bracketed
      v.visit_pexp_pre
      children
      v.visit_pexp_post
      p

and walk_plval
    (v:visitor)
    (p:Ast.plval)
    : unit =
  let children _ =
    match p with
      | Ast.PLVAL_base (Ast.BASE_app (_, tys)) ->
          Array.iter (walk_ty v) tys
      | Ast.PLVAL_base _ -> ()
      | Ast.PLVAL_ext_name (pexp, _) ->
          walk_pexp v pexp
      | Ast.PLVAL_ext_pexp (a, b) ->
          walk_pexp v a;
          walk_pexp v b;
      | Ast.PLVAL_ext_deref pexp ->
          walk_pexp v pexp
  in
    walk_bracketed
      v.visit_plval_pre
      children
      v.visit_plval_post
      p

and walk_atom
    (v:visitor)
    (a:Ast.atom)
    : unit =
  match a with
      Ast.ATOM_literal ls -> walk_lit v ls.node
    | Ast.ATOM_lval lv -> walk_lval v lv
    | Ast.ATOM_pexp p -> walk_pexp v p


and walk_opt_atom
    (v:visitor)
    (ao:Ast.atom option)
    : unit =
  match ao with
      None -> ()
    | Some a -> walk_atom v a


and walk_lit
    (v:visitor)
    (li:Ast.lit)
    : unit =
  walk_bracketed
    v.visit_lit_pre
    (fun _ -> ())
    v.visit_lit_post
    li


and walk_lval
    (v:visitor)
    (lv:Ast.lval)
    : unit =
  walk_bracketed
    v.visit_lval_pre
    (fun _ -> ())
    v.visit_lval_post
    lv


and walk_pat
    (v:visitor)
    (p:Ast.pat)
    : unit =
  let walk p =
    match p with
        Ast.PAT_lit lit -> walk_lit v lit
      | Ast.PAT_tag (lv, pats) ->
          walk_lval v lv;
          Array.iter (walk_pat v) pats
      | Ast.PAT_slot (si, _) -> walk_slot_identified v si
      | Ast.PAT_wild -> ()
  in
    walk_bracketed
      v.visit_pat_pre
      (fun _ -> walk p)
      v.visit_pat_post
      p


and walk_block
    (v:visitor)
    (b:Ast.block)
    : unit =
  walk_bracketed
    v.visit_block_pre
    (fun _ -> (Array.iter (walk_stmt v) b.node))
    v.visit_block_post
    b
;;

(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
