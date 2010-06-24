open Common;;

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

(* FIXME: move these to sess. *)
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
              in
                Some (ar, (get_sections sess ar))
    end
;;

let get_meta
    (sess:Session.sess)
    (filename:filename)
    : Ast.meta option =
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
    (nref:node_id ref)
    (oref:opaque_id ref)
    : Ast.mod_items =
  let dies = get_dies sess filename in
  let items = Hashtbl.create 0 in
    Dwarf.extract_mod_items nref oref abi items dies;
    items
;;

let get_mod
    (sess:Session.sess)
    (abi:Abi.abi)
    (meta:Ast.meta_pat)
    (use_id:node_id)
    (nref:node_id ref)
    (oref:opaque_id ref)
    : (filename * Ast.mod_items) =
  let found = Queue.create () in
  let suffix =
    match sess.Session.sess_targ with
        Win32_x86_pe -> ".dll"
      | MacOS_x86_macho -> ".dylib"
      | Linux_x86_elf -> ".so"
  in
  let rec meta_matches i f_meta =
    if i >= (Array.length meta)
    then true
    else
      match meta.(i) with
          (* FIXME: bind the wildcards. *)
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
              let file = Unix.readdir dh in
                log sess "considering file %s" file;
                if (Filename.check_suffix file suffix) &&
                  (file_matches file)
                then
                  begin
                    iflog sess
                      begin
                        fun _ ->
                          log sess "matched against library %s" file;
                          match get_meta sess file with
                              None -> ()
                            | Some meta ->
                                Array.iter
                                  (fun (k,v) -> log sess "%s = %S" k v)
                                  meta;
                      end;
                    Queue.add file found;
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
          let filename = Queue.pop found in
          let items = get_file_mod sess abi filename nref oref in
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
;;


(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
