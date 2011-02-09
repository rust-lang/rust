
open Common;;

let _ =
  Gc.set { (Gc.get()) with
             Gc.space_overhead = 400; }
;;

let (targ:Common.target) =
  match Sys.os_type with

    | "Win32"
    | "Cygwin" -> Win32_x86_pe

    | "Unix"
        when Unix.system "test `uname -s` = 'Linux'" = Unix.WEXITED 0 ->
        Linux_x86_elf
    | "Unix"
        when Unix.system "test `uname -s` = 'Darwin'" = Unix.WEXITED 0 ->
        MacOS_x86_macho
    | "Unix"
        when Unix.system "test `uname -s` = 'FreeBSD'" = Unix.WEXITED 0 ->
        FreeBSD_x86_elf
    | _ ->
        Linux_x86_elf
;;

let (abi:Abi.abi) = X86.abi;;

let (sess:Session.sess) =
  {
    Session.sess_in = None;
    Session.sess_out = None;
    Session.sess_library_mode = false;
    Session.sess_alt_backend = false;
    Session.sess_minimal = false;
    Session.sess_use_pexps = false;
    (* FIXME (issue #69): need something fancier here for unix
     * sub-flavours.
     *)
    Session.sess_targ = targ;
    Session.sess_log_lex = false;
    Session.sess_log_parse = false;
    Session.sess_log_ast = false;
    Session.sess_log_sig = false;
    Session.sess_log_passes = false;
    Session.sess_log_resolve = false;
    Session.sess_log_type = false;
    Session.sess_log_simplify = false;
    Session.sess_log_layer = false;
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
    Session.sess_log_path = None;
    Session.sess_log_out = stdout;
    Session.sess_log_err = stderr;
    Session.sess_trace_block = false;
    Session.sess_trace_drop = false;
    Session.sess_trace_tag = false;
    Session.sess_trace_gc = false;
    Session.sess_failed = false;
    Session.sess_spans = Hashtbl.create 0;
    Session.sess_report_timing = false;
    Session.sess_report_quads = false;
    Session.sess_report_gc = false;
    Session.sess_report_deps = false;
    Session.sess_next_crate_id = 0;
    Session.sess_fuzz_item_count = 5;
    Session.sess_timings = Hashtbl.create 0;
    Session.sess_quad_counts = Hashtbl.create 0;
    Session.sess_lib_dirs = Queue.create ();
    Session.sess_crate_meta = Hashtbl.create 0;
    Session.sess_node_id_counter = ref (Node 0);
    Session.sess_opaque_id_counter = ref (Opaque 0);
    Session.sess_temp_id_counter = ref (Temp 0);
  }
;;

let exit_if_failed _ =
  if sess.Session.sess_failed
  then exit 1
  else ()
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
                      | FreeBSD_x86_elf -> ""
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
    Lib.get_file_mod sess abi filename in
    Printf.fprintf stdout "%s\n" (Fmt.fmt_to_str Ast.fmt_mod_items items);
    exit_if_failed ();
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

let print_version _ =
  Printf.fprintf stdout "rustboot %s\n" Version.version;
  exit 0;
;;

let flag f opt desc =
  (opt, Arg.Unit f, desc)
;;

let argspecs =
  [
    ("-t", Arg.Symbol (["linux-x86-elf";
                        "win32-x86-pe";
                        "macos-x86-macho";
                        "freebsd-x86-elf"],
                       fun s -> (sess.Session.sess_targ <-
                                   (match s with
                                        "win32-x86-pe" -> Win32_x86_pe
                                      | "macos-x86-macho" -> MacOS_x86_macho
                                      | "freebsd-x86-elf" -> FreeBSD_x86_elf
                                      | _ -> Linux_x86_elf))),
     (" target (default: " ^ (match sess.Session.sess_targ with
                                  Win32_x86_pe -> "win32-x86-pe"
                                | Linux_x86_elf -> "linux-x86-elf"
                                | MacOS_x86_macho -> "macos-x86-macho"
                                | FreeBSD_x86_elf -> "freebsd-x86-elf"
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
    (flag (fun _ -> sess.Session.sess_log_sig <- true)
       "-lsig"      "log signature");
    (flag (fun _ -> sess.Session.sess_log_passes <- true)
       "-lpasses"  "log passes at high-level");
    (flag (fun _ -> sess.Session.sess_log_resolve <- true)
       "-lresolve"  "log resolution");
    (flag (fun _ -> sess.Session.sess_log_type <- true)
       "-ltype"     "log type checking");
    (flag (fun _ -> sess.Session.sess_log_simplify <- true)
       "-lsimplify" "log simplification");
    (flag (fun _ -> sess.Session.sess_log_layer <- true)
       "-llayer"  "log layer checking");
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

    ("-lpath", Arg.String
       (fun s -> sess.Session.sess_log_path <- Some (split_string '.' s)),
     "module path to restrict logging to");

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
    (flag (fun _ -> sess.Session.sess_report_quads <- true)
       "-rquads"       "report categories of quad emitted");
    (flag (fun _ -> sess.Session.sess_report_gc <- true)
       "-rgc"          "report gc behavior of compiler");
    ("-rsig", Arg.String dump_sig,
     "report type-signature from DWARF info in compiled file, then exit");
    ("-rmeta", Arg.String dump_meta,
     "report metadata from DWARF info in compiled file, then exit");
    ("-rdeps", Arg.Unit (fun _ -> sess.Session.sess_report_deps <- true),
     "report dependencies of input, then exit");
    ("-version", Arg.Unit (fun _ -> print_version()),
     "print version information, then exit");

    (flag (fun _ -> sess.Session.sess_use_pexps <- true)
       "-pexp"         "use pexp portion of AST");

    (flag (fun _ -> sess.Session.sess_minimal <- true)
       "-minimal"     ("reduce code size by disabling various features"
                       ^ " (use at own risk)"));

    ("-zc", Arg.Int (fun i -> sess.Session.sess_fuzz_item_count <- i),
     "count of items to generate when fuzzing");

    ("-zs", Arg.Int (fun i -> Fuzz.fuzz (Some i) sess),
     "run fuzzer with given seed");

    (flag (fun _ -> Fuzz.fuzz None sess)
       "-z" "run fuzzer with random seed")

  ] @ (Glue.alt_argspecs sess)
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


let parse_input_crate
    (crate_cache:(crate_id, Ast.mod_items) Hashtbl.t)
    : Ast.crate =
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
              crate_cache
          else
            if Filename.check_suffix infile ".rs"
            then
              Cexp.parse_src_file sess
                (Lib.get_mod sess abi)
                (Lib.infer_lib_name sess)
                crate_cache
            else
              begin
                Printf.fprintf stderr
                  "Error: unrecognized input file type: %s\n"
                  infile;
                exit 1
              end
        in
          exit_if_failed();
          if sess.Session.sess_report_deps
          then
            let outfile = (Session.filename_of sess.Session.sess_out) in
            let depfile =
              match sess.Session.sess_targ with
                  Linux_x86_elf
                | FreeBSD_x86_elf
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

let (crate:Ast.crate) =
  try
    let crate_cache = Hashtbl.create 1 in
    parse_input_crate crate_cache
  with
      Not_implemented (ido, str) ->
        Session.report_err sess ido str;
        { node = Ast.empty_crate'; id = Common.Node 0 }
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
;;

if sess.Session.sess_log_sig
then
  begin
    Printf.fprintf stdout "Post-parse signature:\n";
    Format.set_margin 80;
    Printf.fprintf stdout "%s\n" (Fmt.fmt_to_str Lib.fmt_iface crate);
  end
;;


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
         Simplify.process_crate;
         Type.process_crate;
         Typestate.process_crate;
         Layer.process_crate;
         Effect.process_crate;
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
          None ->
            X86.log sess "selecting insns for %s"
              code.Semant.code_fixup.fixup_name;
            select_insns code.Semant.code_quads
        | Some (n_vregs, spill_fix) ->
            let (quads', n_spills) =
              (Session.time_inner "RA" sess
                 (fun _ ->
                    Ra.reg_alloc sess
                      code.Semant.code_quads
                      n_vregs abi))
            in
            let _ =
              X86.log sess "selecting insns for %s"
                code.Semant.code_fixup.fixup_name
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
            | FreeBSD_x86_elf -> Elf.emit_file
        in
          Session.time_inner "emit" sess
            (fun _ -> emitter sess crate code data sem_cx dwarf);
          exit_if_failed ()
;;

try
  if sess.Session.sess_alt_backend
  then Glue.alt_pipeline sess sem_cx crate
  else main_pipeline ()
with
    Not_implemented (ido, str) ->
      Session.report_err sess ido str
;;

exit_if_failed ()
;;

if sess.Session.sess_report_timing
then
  begin
    let cumulative = ref 0.0 in
    Printf.fprintf stdout "timing:\n\n";
    Array.iter
      begin
        fun name ->
          let t = Hashtbl.find sess.Session.sess_timings name in
          Printf.fprintf stdout "%20s: %f\n" name t;
            cumulative := (!cumulative) +. t
      end
      (sorted_htab_keys sess.Session.sess_timings);
    Printf.fprintf stdout "\n%20s: %f\n" "cumulative" (!cumulative)
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
