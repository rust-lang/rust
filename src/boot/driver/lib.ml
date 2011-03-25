open Common;;
open Fmt;;

let log (sess:Session.sess) =
  Session.log "lib"
    sess.Session.sess_log_lib
    sess.Session.sess_log_out
;;

let iflog (sess:Session.sess) (thunk:(unit -> unit)) : unit =
  if sess.Session.sess_log_lib
  then thunk ()
  else ()
;;

(*
 * Stuff associated with 'crate interfaces'.
 * 
 * The interface of a crate used to be defined by the accompanying DWARF
 * structure in the object file. This was an experiment -- we talked to
 * DWARF hackers before hand and they thought it worth trying -- which did
 * work, and had the advantage of economy of metadata-emitting, but several
 * downsides:
 *
 *   - The reader -- which we want a copy of at runtime in the linker -- has
 *     to know how to read DWARF. It's not the simplest format.
 * 
 *   - The complexity of the encoding meant we didn't always keep pace with
 *     the AST, and maintaining any degree of inter-change compatibility was
 *     going ot be a serious challenge.
 * 
 *   - Diagnostic tools are atrocious, as is the definition of
 *     well-formedness. It's largely trial and error when talking to gdb,
 *     say.
 * 
 *   - Because it was doing double-duty as *driving linkage*, we were never
 *     going to get to the linkage efficiency of native symbols (hash
 *     lookup) anyway. Runtime linkage -- even when lazy -- really ought to
 *     be fast.
 * 
 *   - LLVM, our "main" backend (in rustc) does not really want to make
 *     promises about preserving dwarf.
 *
 *   - LLVM also *is* going to emit native symbols; complete with relocs and
 *     such.  We'd actually have to do *extra work* to inhibit that.
 *
 *   - Most tools are set up to think of DWARF as "debug", meaning
 *     "optional", and may well strip it or otherwise mangle it.
 *
 *   - Many tools want native symbols anyways, and don't know how to look at
 *     DWARF.
 * 
 *   - All the tooling arguments go double on win32. Pretty much only
 *     objdump and gdb understand DWARF-in-PE. Everything else is just blank
 *     stares.
 * 
 * For all these reasons we're moving to a self-made format for describing
 * our interfaces. This will be stored in the .note.rust section as we
 * presently store the meta tags. The encoding is ASCII-compatible (the set
 * of "numbers" to encode is small enough, especially compared to dwarf,
 * that we can just use a text form) and is very easy to read with a simple
 * byte-at-a-time parser.
 * 
 *)

(*
 * Encoding goals:
 *
 *   - Simple. Minimal state or read-ambiguity in reader.
 *
 *   - Compact. Shouldn't add a lot to the size of the binary to glue this
 *     on to it.
 *
 *   - Front-end-y. Doesn't need to contain much beyond parse-level of the
 *     crate's exported items; it'll be fed into the front-end of the
 *     pipeline anyway. No need to have all types or names resolved.
 *
 *   - Testable. Human-legible and easy to identify/fix/test errors in.
 *
 *   - Very fast to read the 'identifying' prefix (version, meta tags, hash)
 *
 *   - Tolerably fast to read in its entirety.
 *
 *   - Safe from version-drift (or at least able to notice it and abort).
 * 
 * Anti-goals:
 * 
 *   - Random access.
 * 
 *   - Generality to other languages.
 *
 * Structure:
 *
 *   - Line oriented.
 *
 *   - Whitespace-separated and whitespace-agnostic. Indent for legibility.
 *
 *   - Each line is a record. A record is either a full item, an item bracket,
 *     a comment, or metadata.
 *
 *     - First byte describes type of record, unless first byte is +, in which
 *       case it's oh-no-we-ran-out-of-tags and it's followed by 2 type-bytes.
 *       (Continue to +++ if you happen to run out *there* as well. You
 *       won't.)
 *
 *       - Metadata type is !
 *
 *       - Comment type is #
 *
 *       - Full item types are: y for type, c for const, f for fn, i for iter,
 *         g for tag constructor.
 *
 *       - Item brackets are those that open/close a scope of
 *         sub-records. These would be obj (o), mod (m), tag (t) to open. The
 *         closer is always '.'.  So a mod looks like:
 *
 *            m foo
 *                c bar
 *            .
 * 
 *     - After first byte of openers and full items is whitespace, then an
 *       ident.
 *
 *     - After that, if it's a ty, fn, iter, obj or tag, there may be [, a
 *        list of comma-separated ty param names, and ].
 *
 *     - After that, if it's a fn, iter, obj or tag constructor, there is a (,
 *       a list of comma-separated type-encoded slot/ident pairs, and a ).
 *
 *     - After that, if it's a fn or iter, there's a '->' and a type-encoded
 *       output.
 *
 *     - After that, a newline '\n'.
 *
 *     - Type encoding is a longer issue! We'll get to that.
 *)

let fmt_iface (ff:Format.formatter) (crate:Ast.crate) : unit =
  let fmt_ty_param ff (p:Ast.ty_param identified) : unit =
    fmt ff "%s" (fst p.node)
  in
  let rec fmt_ty ff (t:Ast.ty) : unit =
    match t with
        Ast.TY_any -> fmt ff "a"
      | Ast.TY_nil -> fmt ff "n"
      | Ast.TY_bool -> fmt ff "b"
      | Ast.TY_mach tm -> fmt ff "%s" (string_of_ty_mach tm)
      | Ast.TY_int -> fmt ff "i"
      | Ast.TY_uint -> fmt ff "u"
      | Ast.TY_char -> fmt ff "c"
      | Ast.TY_str -> fmt ff "s"

      | Ast.TY_tup ttup ->
          fmt_bracketed_arr_sep "(" ")" ","
            fmt_ty ff ttup
      | Ast.TY_vec ty ->
          fmt ff "v["; fmt_ty ff ty; fmt ff "]"
      | Ast.TY_chan ty ->
          fmt ff "C["; fmt_ty ff ty; fmt ff "]"

      | Ast.TY_port ty ->
          fmt ff "P["; fmt_ty ff ty; fmt ff "]"

      | Ast.TY_task ->
          fmt ff "T"

      | Ast.TY_named n -> fmt ff ":"; fmt_name ff n
      | Ast.TY_type -> fmt ff "Y"

      | Ast.TY_box t -> fmt ff "@@"; fmt_ty ff t
      | Ast.TY_mutable t -> fmt ff "~"; fmt_ty ff t

      (* FIXME: finish this. *)
      | Ast.TY_rec _
      | Ast.TY_tag _
      | Ast.TY_fn _
      | Ast.TY_obj _
      | Ast.TY_native _
      | Ast.TY_param _
      | Ast.TY_constrained _ -> fmt ff "Z"

  and fmt_name ff n =
    match n with
        Ast.NAME_base (Ast.BASE_ident id) -> fmt ff "%s" id
      | Ast.NAME_base (Ast.BASE_temp _) -> failwith "temp in fmt_name"
      | Ast.NAME_base (Ast.BASE_app (id, tys)) ->
          fmt ff "%s" id;
          fmt_bracketed_arr_sep "[" "]" ","
            fmt_ty ff tys;
      | Ast.NAME_ext (n, Ast.COMP_ident id) ->
          fmt_name ff n;
          fmt ff ".%s" id
      | Ast.NAME_ext (n, Ast.COMP_app (id, tys)) ->
          fmt_name ff n;
          fmt ff ".%s" id;
          fmt_bracketed_arr_sep "[" "]" ","
            fmt_ty ff tys;
      | Ast.NAME_ext (n, Ast.COMP_idx i) ->
          fmt_name ff n;
          fmt ff "._%d" i
  in
  let rec fmt_mod_item (id:Ast.ident) (mi:Ast.mod_item) : unit =
    let i c = fmt ff "@\n%c %s" c id in

    let o c = fmt ff "@\n"; fmt_obox ff; fmt ff "%c %s" c id in
    let p _ =
      if (Array.length mi.node.Ast.decl_params) <> 0
      then
        fmt_bracketed_arr_sep "[" "]" ","
          fmt_ty_param ff mi.node.Ast.decl_params
    in
    let c _ =  fmt_cbox ff; fmt ff "@\n." in
    match mi.node.Ast.decl_item with
        Ast.MOD_ITEM_type _ -> i 'y'; p()
      | Ast.MOD_ITEM_tag _ -> i 'g'; p()
      | Ast.MOD_ITEM_fn _ -> i 'f'; p();
      | Ast.MOD_ITEM_const _ -> i 'c'
      | Ast.MOD_ITEM_obj _ ->
          o 'o'; p();
          c ()
      | Ast.MOD_ITEM_mod (_, items) ->
          o 'm';
          fmt_mod_items items;
          c ()
  and fmt_mod_items items =
    sorted_htab_iter fmt_mod_item items
  in
  let (_,items) = crate.node.Ast.crate_items in
    fmt_mod_items items
;;

(* Mechanisms for scanning libraries. *)

(* FIXME (issue #67): move these to sess. *)
let ar_cache = Hashtbl.create 0 ;;
let sects_cache = Hashtbl.create 0;;
let meta_cache = Hashtbl.create 0;;
let die_cache = Hashtbl.create 0;;

let get_ar
    (sess:Session.sess)
    (filename:filename)
    : Asm.asm_reader option =
  htab_search_or_add ar_cache filename
    begin
      fun _ ->
        let sniff =
          match sess.Session.sess_targ with
              Win32_x86_pe -> Pe.sniff
            | MacOS_x86_macho -> Macho.sniff
            | Linux_x86_elf -> Elf.sniff
            | FreeBSD_x86_elf -> Elf.sniff
        in
          sniff sess filename
    end
;;


let get_sects
    (sess:Session.sess)
    (filename:filename) :
    (Asm.asm_reader * ((string,(int*int)) Hashtbl.t)) option =
  htab_search_or_add sects_cache filename
    begin
      fun _ ->
        match get_ar sess filename with
            None -> None
          | Some ar ->
              let get_sections =
                match sess.Session.sess_targ with
                    Win32_x86_pe -> Pe.get_sections
                  | MacOS_x86_macho -> Macho.get_sections
                  | Linux_x86_elf -> Elf.get_sections
                  | FreeBSD_x86_elf -> Elf.get_sections
              in
                Some (ar, (get_sections sess ar))
    end
;;

let get_meta
    (sess:Session.sess)
    (filename:filename)
    : Session.meta option =
  htab_search_or_add meta_cache filename
    begin
      fun _ ->
        match get_sects sess filename with
            None -> None
          | Some (ar, sects) ->
              match htab_search sects ".note.rust" with
                  Some (off, _) ->
                    ar.Asm.asm_seek off;
                    Some (Asm.read_rust_note ar)
                | None -> None
    end
;;

let get_dies_opt
    (sess:Session.sess)
    (filename:filename)
    : (Dwarf.rooted_dies option) =
  htab_search_or_add die_cache filename
    begin
      fun _ ->
        match get_sects sess filename with
            None -> None
          | Some (ar, sects) ->
              let debug_abbrev = Hashtbl.find sects ".debug_abbrev" in
              let debug_info = Hashtbl.find sects ".debug_info" in
              let abbrevs = Dwarf.read_abbrevs sess ar debug_abbrev in
              let dies = Dwarf.read_dies sess ar debug_info abbrevs in
                ar.Asm.asm_close ();
                Hashtbl.remove ar_cache filename;
                Some dies
    end
;;

let get_dies
    (sess:Session.sess)
    (filename:filename)
    : Dwarf.rooted_dies =
  match get_dies_opt sess filename with
      None ->
        Printf.fprintf stderr "Error: bad crate file: %s\n%!" filename;
        exit 1
    | Some dies -> dies
;;

let get_file_mod
    (sess:Session.sess)
    (abi:Abi.abi)
    (filename:filename)
    : Ast.mod_items =
  let dies = get_dies sess filename in
  let items = Hashtbl.create 0 in
  let nref = sess.Session.sess_node_id_counter in
  let oref = sess.Session.sess_opaque_id_counter in
    Dwarf.extract_mod_items nref oref abi items dies;
    items
;;

let get_mod
    (sess:Session.sess)
    (abi:Abi.abi)
    (meta:Ast.meta_pat)
    (use_id:node_id)
    (crate_item_cache:(crate_id, Ast.mod_items) Hashtbl.t)
    : (filename * Ast.mod_items) =
  let found = Queue.create () in
  let suffix =
    match sess.Session.sess_targ with
        Win32_x86_pe -> ".dll"
      | MacOS_x86_macho -> ".dylib"
      | Linux_x86_elf -> ".so"
      | FreeBSD_x86_elf -> ".so"
  in
  let rec meta_matches i f_meta =
    if i >= (Array.length meta)
    then true
    else
      match meta.(i) with
          (* FIXME (issue #68): bind the wildcards. *)
          (_, None) -> meta_matches (i+1) f_meta
        | (k, Some v) ->
            match atab_search f_meta k with
                None -> false
              | Some v' ->
                  if v = v'
                  then meta_matches (i+1) f_meta
                  else false
  in
  let file_matches file =
    log sess "searching for metadata in %s" file;
    match get_meta sess file with
        None -> false
      | Some f_meta ->
          log sess "matching metadata in %s" file;
          meta_matches 0 f_meta
  in
    iflog sess
      begin
        fun _ ->
          log sess "searching for library matching:";
          Array.iter
            begin
              fun (k,vo) ->
                match vo with
                    None -> ()
                  | Some v ->
                      log sess "%s = %S" k v
            end
            meta;
      end;
    Queue.iter
      begin
        fun dir ->
          let dh = Unix.opendir dir in
          let rec scan _ =
            try
              let basename = Unix.readdir dh in
              let file = dir ^ "/" ^ basename in
                log sess "considering file %s" file;
                if (Filename.check_suffix file suffix) &&
                  (file_matches file)
                then
                  begin
                    log sess "matched against library %s" file;

                    let meta = get_meta sess file in
                    let crate_id =
                      match meta with
                          None -> Session.make_crate_id sess
                        | Some meta ->
                            iflog sess begin fun _ ->
                              Array.iter
                                (fun (k, v) -> log sess "%s = %S" k v)
                                meta
                            end;
                            htab_search_or_default
                              sess.Session.sess_crate_meta
                              meta
                              (fun () -> Session.make_crate_id sess)
                    in
                    Queue.add (file, crate_id) found;
                  end;
                scan()
            with
                End_of_file -> ()
          in
            scan ()
      end
      sess.Session.sess_lib_dirs;
    match Queue.length found with
        0 -> Common.err (Some use_id) "unsatisfied 'use' clause"
      | 1 ->
          let (filename, crate_id) = Queue.pop found in
          let items =
              htab_search_or_default crate_item_cache crate_id
                (fun () -> get_file_mod sess abi filename)
          in
            (filename, items)
      | _ -> Common.err (Some use_id) "multiple crates match 'use' clause"
;;

let infer_lib_name
    (sess:Session.sess)
    (ident:filename)
    : filename =
  match sess.Session.sess_targ with
      Win32_x86_pe -> ident ^ ".dll"
    | MacOS_x86_macho -> "lib" ^ ident ^ ".dylib"
    | Linux_x86_elf -> "lib" ^ ident ^ ".so"
    | FreeBSD_x86_elf -> "lib" ^ ident ^ ".so"
;;


(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
