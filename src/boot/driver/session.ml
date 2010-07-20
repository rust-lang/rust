(*
 * This module goes near the bottom of the dependency DAG, and holds option,
 * and global-state machinery for a single run of the compiler.
 *)

open Common;;

type sess =
{
  mutable sess_in: filename option;
  mutable sess_out: filename option;
  mutable sess_library_mode: bool;
  mutable sess_alt_backend: bool;
  mutable sess_targ: target;
  mutable sess_log_lex: bool;
  mutable sess_log_parse: bool;
  mutable sess_log_ast: bool;
  mutable sess_log_resolve: bool;
  mutable sess_log_type: bool;
  mutable sess_log_effect: bool;
  mutable sess_log_typestate: bool;
  mutable sess_log_dead: bool;
  mutable sess_log_loop: bool;
  mutable sess_log_alias: bool;
  mutable sess_log_layout: bool;
  mutable sess_log_trans: bool;
  mutable sess_log_itype: bool;
  mutable sess_log_dwarf: bool;
  mutable sess_log_ra: bool;
  mutable sess_log_insn: bool;
  mutable sess_log_asm: bool;
  mutable sess_log_obj: bool;
  mutable sess_log_lib: bool;
  mutable sess_log_out: out_channel;
  mutable sess_trace_block: bool;
  mutable sess_trace_drop: bool;
  mutable sess_trace_tag: bool;
  mutable sess_trace_gc: bool;
  mutable sess_failed: bool;
  mutable sess_report_timing: bool;
  mutable sess_report_gc: bool;
  mutable sess_report_deps: bool;
  sess_timings: (string, float) Hashtbl.t;
  sess_spans: (node_id,span) Hashtbl.t;
  sess_lib_dirs: filename Queue.t;
}
;;

let add_time sess name amt =
  let existing =
    if Hashtbl.mem sess.sess_timings name
    then Hashtbl.find sess.sess_timings name
    else 0.0
  in
    (Hashtbl.replace sess.sess_timings name (existing +. amt))
;;

let time_inner name sess thunk =
  let t0 = Unix.gettimeofday() in
  let x = thunk() in
  let t1 = Unix.gettimeofday() in
    add_time sess name (t1 -. t0);
    x
;;

let get_span sess id =
  if Hashtbl.mem sess.sess_spans id
  then (Some (Hashtbl.find sess.sess_spans id))
  else None
;;

let log name flag chan =
  let k1 s =
    Printf.fprintf chan "%s: %s\n%!" name s
  in
  let k2 _ = () in
    Printf.ksprintf (if flag then k1 else k2)
;;

let fail sess =
  sess.sess_failed <- true;
  Printf.fprintf sess.sess_log_out
;;


let string_of_pos (p:pos) =
  let (filename, line, col) = p in
  Printf.sprintf "%s:%d:%d" filename line col
;;


let string_of_span (s:span) =
    let (filename, line0, col0) = s.lo in
    let (_, line1, col1) = s.hi in
    Printf.sprintf "%s:%d:%d - %d:%d" filename line0 col0 line1 col1
;;

let filename_of (fo:filename option) : filename =
  match fo with
      None -> "<none>"
    | Some f -> f
;;

let report_err sess ido str =
  let spano = match ido with
      None -> None
    | Some id -> get_span sess id
  in
    match spano with
        None ->
          fail sess "Error: %s\n%!" str
      | Some span ->
          fail sess "%s:E:Error: %s\n%!"
            (string_of_span span) str
;;

(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)
