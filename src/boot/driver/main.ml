
open Common;;

let _ =
  Gc.set { (Gc.get()) with
             Gc.space_overhead = 400; }
;;

let (targ:Common.target) =
  match Sys.os_type with
      "Unix" ->
        (* FIXME (issue #69): this is an absurd heuristic. *)
        if Sys.file_exists "/System/Library"
        then MacOS_x86_macho
        else Linux_x86_elf
    | "Win32" -> Win32_x86_pe
    | "Cygwin" -> Win32_x86_pe
    | _ -> Linux_x86_elf
;;

let (abi:Abi.abi) = X86.abi;;

let (sess:Session.sess) =
  {
    Session.sess_in = None;
    Session.sess_out = None;
    Session.sess_library_mode = false;
    Session.sess_alt_backend = false;
    (* FIXME (issue #69): need something fancier here for unix
     * sub-flavours.
     *)
    Session.sess_targ = targ;
    Session.sess_log_lex = false;
    Session.sess_log_parse = false;
    Session.sess_log_ast = false;
    Session.sess_log_resolve = false;
    Session.sess_log_type = false;
    Session.sess_log_effect = false;
    Session.sess_log_typestate = false;
    Session.sess_log_loop = false;
    Session.sess_log_alias = false;
    Session.sess_log_dead = false;
    Session.sess_log_layout = false;
    Session.sess_log_itype = false;
    Session.sess_log_trans = false;
    Session.sess_log_dwarf = false;
    Session.sess_log_ra = false;
    Session.sess_log_insn = false;
    Session.sess_log_asm = false;
    Session.sess_log_obj = false;
    Session.sess_log_lib = false;
    Session.sess_log_out = stdout;
    Session.sess_trace_block = false;
    Session.sess_trace_drop = false;
    Session.sess_trace_tag = false;
    Session.sess_trace_gc = false;
    Session.sess_failed = false;
    Session.sess_spans = Hashtbl.create 0;
    Session.sess_report_timing = false;
    Session.sess_report_gc = false;
    Session.sess_report_deps = false;
    Session.sess_timings = Hashtbl.create 0;
    Session.sess_lib_dirs = Queue.create ();
  }
;;

let default_output_filename (sess:Session.sess) : filename option =
  match sess.Session.sess_in with
      None -> None
    | Some fname ->
        let base = Filename.chop_extension (Filename.basename fname) in
        let out =
          if sess.Session.sess_library_mode
          then
            Lib.infer_lib_name sess base
          else
            base ^ (match sess.Session.sess_targ with
                        Linux_x86_elf -> ""
                      | MacOS_x86_macho -> ""
                      | Win32_x86_pe -> ".exe")
        in
          Some out
;;

let set_default_output_filename (sess:Session.sess) : unit =
  match sess.Session.sess_out with
      None -> (sess.Session.sess_out <- default_output_filename sess)
    | _ -> ()
;;


let dump_sig (filename:filename) : unit =
  let items =
    Lib.get_file_mod sess abi filename (ref (Node 0)) (ref (Opaque 0)) in
    Printf.fprintf stdout "%s\n" (Fmt.fmt_to_str Ast.fmt_mod_items items);
    exit 0
;;

let dump_meta (filename:filename) : unit =
  begin
    match Lib.get_meta sess filename with
        None -> Printf.fprintf stderr "Error: bad crate file: %s\n" filename
      | Some meta ->
          Array.iter
            begin
              fun (k,v) ->
                Printf.fprintf stdout "%s = %S\n" k v;
            end
            meta
  end;
  exit 0
;;

let flag f opt desc =
  (opt, Arg.Unit f, desc)
;;

let argspecs =
  [
    ("-t", Arg.Symbol (["linux-x86-elf"; "win32-x86-pe"; "macos-x86-macho"],
                       fun s -> (sess.Session.sess_targ <-
                                   (match s with
                                        "win32-x86-pe" -> Win32_x86_pe
                                      | "macos-x86-macho" -> MacOS_x86_macho
                                      | _ -> Linux_x86_elf))),
     (" target (default: " ^ (match sess.Session.sess_targ with
                                  Win32_x86_pe -> "win32-x86-pe"
                                | Linux_x86_elf -> "linux-x86-elf"
                                | MacOS_x86_macho -> "macos-x86-macho"
                             ) ^ ")"));
    ("-o", Arg.String (fun s -> sess.Session.sess_out <- Some s),
     "file to output (default: "
     ^ (Session.filename_of sess.Session.sess_out) ^ ")");
    ("-shared", Arg.Unit (fun _ -> sess.Session.sess_library_mode <- true),
     "compile a shared-library crate");
    ("-L", Arg.String (fun s -> Queue.add s sess.Session.sess_lib_dirs),
     "dir to add to library path");
    ("-litype", Arg.Unit (fun _ -> sess.Session.sess_log_itype <- true;
                            Il.log_iltypes := true), "log IL types");
    (flag (fun _ -> sess.Session.sess_log_lex <- true)
       "-llex"      "log lexing");
    (flag (fun _ -> sess.Session.sess_log_parse <- true)
       "-lparse"    "log parsing");
    (flag (fun _ -> sess.Session.sess_log_ast <- true)
       "-last"      "log AST");
    (flag (fun _ -> sess.Session.sess_log_resolve <- true)
       "-lresolve"  "log resolution");
    (flag (fun _ -> sess.Session.sess_log_type <- true)
       "-ltype"     "log type checking");
    (flag (fun _ -> sess.Session.sess_log_effect <- true)
       "-leffect"   "log effect checking");
    (flag (fun _ -> sess.Session.sess_log_typestate <- true)
       "-ltypestate" "log typestate pass");
    (flag (fun _ -> sess.Session.sess_log_loop <- true)
       "-lloop"      "log loop analysis");
    (flag (fun _ -> sess.Session.sess_log_alias <- true)
       "-lalias"      "log alias analysis");
    (flag (fun _ -> sess.Session.sess_log_dead <- true)
       "-ldead"       "log dead analysis");
    (flag (fun _ -> sess.Session.sess_log_layout <- true)
       "-llayout"     "log frame layout");
    (flag (fun _ -> sess.Session.sess_log_trans <- true)
       "-ltrans"      "log IR translation");
    (flag (fun _ -> sess.Session.sess_log_dwarf <- true)
       "-ldwarf"      "log DWARF generation");
    (flag (fun _ -> sess.Session.sess_log_ra <- true)
       "-lra"         "log register allocation");
    (flag (fun _ -> sess.Session.sess_log_insn <- true)
       "-linsn"       "log instruction selection");
    (flag (fun _ -> sess.Session.sess_log_asm <- true)
       "-lasm"        "log assembly");
    (flag (fun _ -> sess.Session.sess_log_obj <- true)
       "-lobj"        "log object-file generation");
    (flag (fun _ -> sess.Session.sess_log_lib <- true)
       "-llib"        "log library search");

    (flag (fun _ -> sess.Session.sess_trace_block <- true)
       "-tblock"      "emit block-boundary tracing code");
    (flag (fun _ -> sess.Session.sess_trace_drop <- true)
       "-tdrop"       "emit slot-drop tracing code");
    (flag (fun _ -> sess.Session.sess_trace_tag <- true)
       "-ttag"        "emit tag-construction tracing code");
    (flag (fun _ -> sess.Session.sess_trace_gc <- true)
       "-tgc"         "emit GC tracing code");

    ("-tall", Arg.Unit (fun _ ->
                          sess.Session.sess_trace_block <- true;
                          sess.Session.sess_trace_drop <- true;
                          sess.Session.sess_trace_tag <- true ),
     "emit all tracing code");

    (flag (fun _ -> sess.Session.sess_report_timing <- true)
       "-rtime"        "report timing of compiler phases");
    (flag (fun _ -> sess.Session.sess_report_gc <- true)
       "-rgc"          "report gc behavior of compiler");
    ("-rsig", Arg.String dump_sig,
     "report type-signature from DWARF info in compiled file, then exit");
    ("-rmeta", Arg.String dump_meta,
     "report metadata from DWARF info in compiled file, then exit");
    ("-rdeps", Arg.Unit (fun _ -> sess.Session.sess_report_deps <- true),
     "report dependencies of input, then exit");
  ] @ (Glue.alt_argspecs sess)
;;

let exit_if_failed _ =
  if sess.Session.sess_failed
  then exit 1
  else ()
;;

Arg.parse
  argspecs
  (fun arg -> sess.Session.sess_in <- (Some arg))
  ("usage: " ^ Sys.argv.(0) ^ " [options] (CRATE_FILE.rc|SOURCE_FILE.rs)\n")
;;

let _ = set_default_output_filename  sess
;;

let _ =
  if sess.Session.sess_out = None
  then (Printf.fprintf stderr "Error: no output file specified\n"; exit 1)
  else ()
;;

let _ =
  if sess.Session.sess_in = None
  then (Printf.fprintf stderr "Error: empty input filename\n"; exit 1)
  else ()
;;


let (crate:Ast.crate) =
  Session.time_inner "parse" sess
    begin
      fun _ ->
        let infile = Session.filename_of sess.Session.sess_in in
        let crate =
          if Filename.check_suffix infile ".rc"
          then
            Cexp.parse_crate_file sess
              (Lib.get_mod sess abi)
              (Lib.infer_lib_name sess)
          else
            if Filename.check_suffix infile ".rs"
            then
              Cexp.parse_src_file sess
                (Lib.get_mod sess abi)
                (Lib.infer_lib_name sess)
            else
              begin
                Printf.fprintf stderr
                  "Error: unrecognized input file type: %s\n"
                  infile;
                exit 1
              end
        in
          if sess.Session.sess_report_deps
          then
            let outfile = (Session.filename_of sess.Session.sess_out) in
            let depfile =
              match sess.Session.sess_targ with
                  Linux_x86_elf
                | MacOS_x86_macho -> outfile ^ ".d"
                | Win32_x86_pe -> (Filename.chop_extension outfile) ^ ".d"
            in
              begin
                Array.iter
                  begin
                    fun out ->
                      Printf.fprintf stdout "%s: \\\n" out;
                      Hashtbl.iter
                        (fun _ file ->
                           Printf.fprintf stdout "    %s \\\n" file)
                        crate.node.Ast.crate_files;
                      Printf.fprintf stdout "\n"
                  end
                  [| outfile; depfile|];
                exit 0
              end
          else
            crate
    end
;;

exit_if_failed ()
;;

if sess.Session.sess_log_ast
then
  begin
    Printf.fprintf stdout "Post-parse AST:\n";
    Format.set_margin 80;
    Printf.fprintf stdout "%s\n" (Fmt.fmt_to_str Ast.fmt_crate crate)
  end

let list_to_seq ls = Asm.SEQ (Array.of_list ls);;
let select_insns (quads:Il.quads) : Asm.frag =
  Session.time_inner "insn" sess
    (fun _ -> X86.select_insns sess quads)
;;


(* Semantic passes. *)
let sem_cx = Semant.new_ctxt sess abi crate.node
;;


let main_pipeline _ =
  let _ =
    Array.iter
      (fun proc ->
         proc sem_cx crate;
         exit_if_failed ())
      [| Resolve.process_crate;
         Type.process_crate;
         Effect.process_crate;
         Typestate.process_crate;
         Loop.process_crate;
         Alias.process_crate;
         Dead.process_crate;
         Layout.process_crate;
         Trans.process_crate |]
  in

  (* Tying up various knots, allocating registers and selecting
   * instructions.
   *)
  let process_code _ (code:Semant.code) : Asm.frag =
    let frag =
      match code.Semant.code_vregs_and_spill with
          None -> select_insns code.Semant.code_quads
        | Some (n_vregs, spill_fix) ->
            let (quads', n_spills) =
              (Session.time_inner "RA" sess
                 (fun _ ->
                    Ra.reg_alloc sess
                      code.Semant.code_quads
                      n_vregs abi))
            in
            let insns = select_insns quads' in
              begin
                spill_fix.fixup_mem_sz <-
                  Some (Int64.mul
                          (Int64.of_int n_spills)
                          abi.Abi.abi_word_sz);
                insns
              end
    in
      Asm.ALIGN_FILE (Abi.general_code_alignment,
                      Asm.DEF (code.Semant.code_fixup, frag))
  in

  let (file_frags:Asm.frag) =
    let process_file file_id frag_code =
      let file_fix = Hashtbl.find sem_cx.Semant.ctxt_file_fixups file_id in
        Asm.DEF (file_fix,
                 list_to_seq (reduce_hash_to_list process_code frag_code))
    in
      list_to_seq (reduce_hash_to_list
                     process_file sem_cx.Semant.ctxt_file_code)
  in

    exit_if_failed ();
    let (glue_frags:Asm.frag) =
      list_to_seq (reduce_hash_to_list
                     process_code sem_cx.Semant.ctxt_glue_code)
    in

      exit_if_failed ();
      let code = Asm.SEQ [| file_frags; glue_frags |] in
      let data = list_to_seq (reduce_hash_to_list
                                (fun _ (_, i) -> i) sem_cx.Semant.ctxt_data)
      in
      (* Emitting Dwarf and PE/ELF/Macho. *)
      let (dwarf:Dwarf.debug_records) =
        Session.time_inner "dwarf" sess
          (fun _ -> Dwarf.process_crate sem_cx crate)
      in

        exit_if_failed ();
        let emitter =
          match sess.Session.sess_targ with
              Win32_x86_pe -> Pe.emit_file
            | MacOS_x86_macho -> Macho.emit_file
            | Linux_x86_elf -> Elf.emit_file
        in
          Session.time_inner "emit" sess
            (fun _ -> emitter sess crate code data sem_cx dwarf);
          exit_if_failed ()
;;

if sess.Session.sess_alt_backend
then Glue.alt_pipeline sess sem_cx crate
else main_pipeline ()
;;

if sess.Session.sess_report_timing
then
  begin
    Printf.fprintf stdout "timing:\n\n";
    Array.iter
      begin
        fun name ->
          Printf.fprintf stdout "%20s: %f\n" name
            (Hashtbl.find sess.Session.sess_timings name)
      end
      (sorted_htab_keys sess.Session.sess_timings)
  end;
;;

if sess.Session.sess_report_gc
then Gc.print_stat stdout;;


(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
