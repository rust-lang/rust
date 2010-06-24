(*
 * LLVM emitter.
 *)

(* The top-level interface to the LLVM translation subsystem. *)
let trans_and_process_crate
    (sess:Session.sess)
    (sem_cx:Semant.ctxt)
    (crate:Ast.crate)
    : unit =
  let llcontext = Llvm.create_context () in
  let emit_file (llmod:Llvm.llmodule) : unit =
    let filename = Session.filename_of sess.Session.sess_out in
    if not (Llvm_bitwriter.write_bitcode_file llmod filename)
    then raise (Failure ("failed to write the LLVM bitcode '" ^ filename
      ^ "'"))
  in
  let llmod = Lltrans.trans_crate sem_cx llcontext sess crate in
  begin
    try
      emit_file llmod
    with e -> Llvm.dispose_module llmod; raise e
  end;
  Llvm.dispose_module llmod;
  Llvm.dispose_context llcontext
;;

(*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 *)

